/*
package apps

import java.io._
import org.apache.commons.io.FileUtils

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.storage.StorageLevel

import libs._
import loaders._
import preprocessing._

object ImageNetCreateDBApp {
  val trainBatchSize = 256
  val testBatchSize = 50
  val channels = 3
  val fullWidth = 256
  val fullHeight = 256
  val croppedWidth = 227
  val croppedHeight = 227
  val fullImShape = Array(channels, fullHeight, fullWidth)
  val fullImSize = fullImShape.product

  val workerStore = new WorkerStore()

  def main(args: Array[String]) {
    val numWorkers = args(0).toInt
    val conf = new SparkConf()
      .setAppName("ImageNetCreateDB")
      .set("spark.driver.maxResultSize", "30G")
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

    val loader = new ImageNetLoader("sparknet")
    log("loading train data")
    var trainRDD = loader.apply(sc, "ILSVRC2012_train/", "train.txt")
    log("loading test data")
    val testRDD = loader.apply(sc, "ILSVRC2012_test/", "test.txt")

    log("processing train data")
    val trainConverter = new ScaleAndConvert(trainBatchSize, fullHeight, fullWidth)
    var trainMinibatchRDD = trainConverter.makeMinibatchRDDWithCompression(trainRDD).persist(StorageLevel.MEMORY_AND_DISK)
    val numTrainMinibatches = trainMinibatchRDD.count()
    log("numTrainMinibatches = " + numTrainMinibatches.toString)

    log("processing test data")
    val testConverter = new ScaleAndConvert(testBatchSize, fullHeight, fullWidth)
    var testMinibatchRDD = testConverter.makeMinibatchRDDWithCompression(testRDD).persist(StorageLevel.MEMORY_AND_DISK)
    val numTestMinibatches = testMinibatchRDD.count()
    log("numTestMinibatches = " + numTestMinibatches.toString)

    val numTrainData = numTrainMinibatches * trainBatchSize
    val numTestData = numTestMinibatches * testBatchSize

    log("coalescing") // if you want to shuffle your data, replace coalesce with repartition
    trainMinibatchRDD = trainMinibatchRDD.coalesce(numWorkers)
    testMinibatchRDD = testMinibatchRDD.coalesce(numWorkers)

    val trainPartitionSizes = trainMinibatchRDD.mapPartitions(iter => Array(iter.size).iterator).persist()
    val testPartitionSizes = testMinibatchRDD.mapPartitions(iter => Array(iter.size).iterator).persist()
    log("trainPartitionSizes = " + trainPartitionSizes.collect().deep.toString)
    log("testPartitionSizes = " + testPartitionSizes.collect().deep.toString)
    val testPartitionSizesFilename = sparkNetHome + "/infoFiles/imagenet_num_test_batches.txt"
    testPartitionSizes.foreach(size => {
      val file = new File(testPartitionSizesFilename)
      file.delete()
      file.getParentFile().mkdirs()
      val pw = new PrintWriter(file)
      pw.write(size.toString)
      pw.flush()
    })

    val trainDBFilename = sparkNetHome + "/caffe/examples/imagenet/ilsvrc12_train_db"
    val testDBFilename = sparkNetHome + "/caffe/examples/imagenet/ilsvrc12_val_db"
    val meanImageFilename = sparkNetHome + "/caffe/data/ilsvrc12/imagenet_mean.binaryproto"

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

    log("write train data to DB")
    trainMinibatchRDD.mapPartitions(minibatchIt => {
      FileUtils.deleteDirectory(new File(trainDBFilename))
      val DBCreator = new CreateDB(workerStore.getLib, "leveldb")
      DBCreator.makeDBFromMinibatchPartition(minibatchIt, trainDBFilename, fullHeight, fullWidth)
      Array(0).iterator
    }).foreach(_ => ())

    log("write test data to DB")
    testMinibatchRDD.mapPartitions(minibatchIt => {
      FileUtils.deleteDirectory(new File(testDBFilename))
      val DBCreator = new CreateDB(workerStore.getLib, "leveldb")
      DBCreator.makeDBFromMinibatchPartition(minibatchIt, testDBFilename, fullHeight, fullWidth)
      Array(0).iterator
    }).foreach(_ => ())

    log("computing mean image")
    val meanImage = ComputeMean.computeMeanFromMinibatches(trainMinibatchRDD, fullImShape, numTrainData.toInt)
    log("saving mean image on master")
    new File(meanImageFilename).delete()
    ComputeMean.writeMeanToBinaryProto(caffeLib, meanImage, meanImageFilename)
    log("saving mean image on workers")
    workers.foreach(_ => {
      new File(meanImageFilename).delete()
      ComputeMean.writeMeanToBinaryProto(workerStore.getLib, meanImage, meanImageFilename)
    })

    log("finished creating databases")
  }
}
*/
