/*
package apps

import java.io._

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

import libs._
import loaders._
import preprocessing._

// for this app to work, $SPARKNET_HOME should be the SparkNet root directory
// and you need to run $SPARKNET_HOME/caffe/data/cifar10/get_cifar10.sh
object FeaturizerApp {
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
      .setAppName("Featurizer")
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

    log("repartition data")
    trainRDD = trainRDD.repartition(numWorkers)

    log("processing train data")
    val trainConverter = new ScaleAndConvert(trainBatchSize, height, width)
    var trainMinibatchRDD = trainConverter.makeMinibatchRDDWithoutCompression(trainRDD).persist()
    val numTrainMinibatches = trainMinibatchRDD.count()
    log("numTrainMinibatches = " + numTrainMinibatches.toString)

    val numTrainData = numTrainMinibatches * trainBatchSize

    val trainPartitionSizes = trainMinibatchRDD.mapPartitions(iter => Array(iter.size).iterator).persist()
    log("trainPartitionSizes = " + trainPartitionSizes.collect().deep.toString)

    val workers = sc.parallelize(Array.range(0, numWorkers), numWorkers)

    // initialize nets on workers
    workers.foreach(_ => {
      System.load(sparkNetHome + "/build/libccaffe.so")
      val caffeLib = CaffeLibrary.INSTANCE
      var netParameter = ProtoLoader.loadNetPrototxt(sparkNetHome + "/caffe/examples/cifar10/cifar10_full_train_test.prototxt")
      netParameter = ProtoLoader.replaceDataLayers(netParameter, trainBatchSize, testBatchSize, channels, height, width)
      val solverParameter = ProtoLoader.loadSolverPrototxtWithNet(sparkNetHome + "/caffe/examples/cifar10/cifar10_full_solver.prototxt", netParameter, None)
      val net = CaffeNet(caffeLib, solverParameter)
      workerStore.setNet("net", net)
    })

    // initialize weights on master
    var netWeights = workers.map(_ => workerStore.getNet("net").getWeights()).collect()(0)

    log("broadcasting weights")
    val broadcastWeights = sc.broadcast(netWeights)
    log("setting weights on workers")
    workers.foreach(_ => workerStore.getNet("net").setWeights(broadcastWeights.value))

    log("extracting features")
    val featureBatchRDD = trainPartitionSizes.zipPartitions(trainMinibatchRDD) (
      (lenIt, trainMinibatchIt) => {
        assert(lenIt.hasNext && trainMinibatchIt.hasNext)
        val len = lenIt.next
        assert(!lenIt.hasNext)
        val minibatchSampler = new MinibatchSampler(trainMinibatchIt, len, len)
        workerStore.getNet("net").setTrainData(minibatchSampler, None)
        val featureBatch = new Array[NDArray](len)
        for (i <- 0 to len - 1) {
          workerStore.getNet("net").forward()
          featureBatch(i) = workerStore.getNet("net").getData()("ip1")
        }
        featureBatch.iterator
      }
    )
    featureBatchRDD.foreachPartition(_ => ())

    log("finished featurizing")
  }
}
*/
