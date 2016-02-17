/*
package apps

import java.io._
import org.apache.commons.io.FileUtils

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

import libs._
import loaders._
import preprocessing._

// this app pulls Cifar10 and creates LevelDB databases which Caffe reads from, you
// need to run $SPARKNET_HOME/caffe/data/cifar10/get_cifar10.sh to get the cifar
// data before running this app
object CifarDBApp {
  val trainBatchSize = 100
  val testBatchSize = 100
  val channels = 3
  val width = 32
  val height = 32
  val imShape = Array(channels, height, width)
  val size = imShape.product

  val workerStore = new WorkerStore()

  def main(args: Array[String]) {
    val numWorkers = args(0).toInt
    val conf = new SparkConf()
      .setAppName("CifarDB")
      .set("spark.driver.maxResultSize", "5G")
      .set("spark.task.maxFailures", "1")
    val sc = new SparkContext(conf)

    val sparkNetHome = sys.env("SPARKNET_HOME")

    // information for logging
    val startTime = System.currentTimeMillis()
    val trainingLog = new PrintWriter(new File(sparkNetHome + "/training_log_" + startTime.toString + ".txt" ))
    def log(message: String, i: Int = -1) {
      val elapsedTime = 1F * (System.currentTimeMillis() - startTime) / 1000
      if (i == -1) {
        trainingLog.write(elapsedTime.toString + ": "  + message + "\n")
      } else {
        trainingLog.write(elapsedTime.toString + ", i = " + i.toString + ": "+ message + "\n")
      }
      trainingLog.flush()
    }

    val loader = new CifarLoader(sparkNetHome + "/caffe/data/cifar10/")
    log("loading train data")
    var trainRDD = sc.parallelize(loader.trainImages.zip(loader.trainLabels))
    log("loading test data")
    var testRDD = sc.parallelize(loader.testImages.zip(loader.testLabels))

    log("repartition data")
    trainRDD = trainRDD.repartition(numWorkers)
    testRDD = testRDD.repartition(numWorkers)

    log("processing train data")
    val trainConverter = new ScaleAndConvert(trainBatchSize, height, width)
    var trainRDDConverted = trainConverter.scaleAndConvertWithoutCompression(trainRDD).persist()
    val numTrainData = trainRDDConverted.count()
    log("numTrainData = " + numTrainData.toString)

    log("processing test data")
    val testConverter = new ScaleAndConvert(testBatchSize, height, width)
    var testRDDConverted = testConverter.scaleAndConvertWithoutCompression(testRDD).persist()
    val numTestData = testRDDConverted.count()
    log("numTestData = " + numTestData.toString)

    val numTrainMinibatches = numTrainData / trainBatchSize
    val numTestMinibatches = numTestData / testBatchSize

    val trainPartitionSizes = trainRDDConverted.mapPartitions(iter => Array(iter.size / trainBatchSize).iterator).persist()
    val testPartitionSizes = testRDDConverted.mapPartitions(iter => Array(iter.size / testBatchSize).iterator).persist()
    log("trainPartitionSizes = " + trainPartitionSizes.collect().deep.toString)
    log("testPartitionSizes = " + testPartitionSizes.collect().deep.toString)

    val workers = sc.parallelize(Array.range(0, numWorkers), numWorkers)

    // initialize Caffe Libraries on workers
    workers.foreach(_ => {
      System.load(sparkNetHome + "/build/libccaffe.so")
      val caffeLib = CaffeLibrary.INSTANCE
      caffeLib.set_basepath(sparkNetHome + "/caffe/")
      workerStore.setLib(caffeLib)
    })
    System.load(sparkNetHome + "/build/libccaffe.so")
    val caffeLib = CaffeLibrary.INSTANCE
    caffeLib.set_basepath(sparkNetHome + "/caffe/")

    val trainDBFilename = sparkNetHome + "/caffe/examples/cifar10/cifar10_train_db"
    val testDBFilename = sparkNetHome + "/caffe/examples/cifar10/cifar10_test_db"

    log("write train data to DB")
    trainRDDConverted.mapPartitions(dataIt => {
      FileUtils.deleteDirectory(new File(trainDBFilename))
      val DBCreator = new CreateDB(workerStore.getLib, "leveldb")
      DBCreator.makeDBFromPartition(dataIt, trainDBFilename, height, width)
      Array(0).iterator
    }).foreach(_ => ())

    log("write test data to DB")
    testRDDConverted.mapPartitions(dataIt => {
      FileUtils.deleteDirectory(new File(testDBFilename))
      val DBCreator = new CreateDB(workerStore.getLib, "leveldb")
      DBCreator.makeDBFromPartition(dataIt, testDBFilename, height, width)
      Array(0).iterator
    }).foreach(_ => ())

    log("computing mean image")
    val meanImageFilename = sparkNetHome + "/caffe/examples/cifar10/mean.binaryproto"
    val meanImage = ComputeMean.computeMean(trainRDDConverted, imShape, numTrainData.toInt)
    log("saving mean image on master")
    new File(meanImageFilename).delete()
    ComputeMean.writeMeanToBinaryProto(caffeLib, meanImage, meanImageFilename)
    log("saving mean image on workers")
    workers.foreach(_ => {
      new File(meanImageFilename).delete()
      ComputeMean.writeMeanToBinaryProto(workerStore.getLib, meanImage, meanImageFilename)
    })

    log("finished creating databases")

    log("initialize nets on workers")
    workers.foreach(_ => {
      val solverParameter = ProtoLoader.loadSolverPrototxt(sparkNetHome + "/caffe/examples/cifar10/cifar10_full_solver.prototxt")
      val net = CaffeNet(workerStore.getLib, solverParameter)
      workerStore.setNet("net", net)
    })
    testPartitionSizes.foreach(size => workerStore.getNet("net").setNumTestBatches(size)) // tell each net how many test batches it has

    // initialize weights on master
    var netWeights = workers.map(_ => workerStore.getNet("net").getWeights()).collect()(0)

    var i = 0
    while (true) {
      log("broadcasting weights", i)
      val broadcastWeights = sc.broadcast(netWeights)
      log("setting weights on workers", i)
      workers.foreach(_ => workerStore.getNet("net").setWeights(broadcastWeights.value))

      /*
      if (i % 10 == 0) {
        net.setWeights(netWeights)
        net.saveWeightsToFile("/root/weights/" + i.toString + ".caffemodel")
      }
      */

      if (i % 10 == 0) {
        log("testing, i")
        val testScores = workers.map(_ => workerStore.getNet("net").test()).cache()
        val testScoresAggregate = testScores.reduce((a, b) => (a, b).zipped.map(_ + _))
        val accuracies = testScoresAggregate.map(v => 100F * v / numTestMinibatches)
        log("%.2f".format(accuracies(0)) + "% accuracy", i)
      }

      log("training", i)
      val syncInterval = 10
      workers.foreach(_ => workerStore.getNet("net").train(syncInterval))

      log("collecting weights", i)
      netWeights = workers.map(_ => { workerStore.getNet("net").getWeights() }).reduce((a, b) => WeightCollection.add(a, b))
      netWeights.scalarDivide(1F * numWorkers)
      i += 1
    }

    log("finished training")
  }
}
*/
