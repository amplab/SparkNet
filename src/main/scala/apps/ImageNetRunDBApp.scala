/*
package apps

import java.io._
import org.apache.commons.io.FileUtils
import scala.io.Source

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.storage.StorageLevel

import libs._
import loaders._
import preprocessing._

object ImageNetRunDBApp {
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
      .setAppName("ImageNetRunDB")
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

    val workers = sc.parallelize(Array.range(0, numWorkers), numWorkers)

    val testPartitionSizesFilename = sparkNetHome + "/infoFiles/imagenet_num_test_batches.txt"
    val testPartitionSizes = workers.map(_ => {
      Source.fromFile(testPartitionSizesFilename).getLines.mkString.toInt
    })
    log("testPartitionSizes = " + testPartitionSizes.collect.deep.toString)
    val numTestMinibatches = testPartitionSizes.sum()

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

    log("initialize nets on workers")
    workers.foreach(_ => {
      val solverParameter = ProtoLoader.loadSolverPrototxt(sparkNetHome + "/caffe/models/bvlc_reference_caffenet/solver.prototxt")
      val net = CaffeNet(workerStore.getLib, solverParameter)
      net.loadWeightsFromFile(sparkNetHome + "/caffe/models/bvlc_reference_caffenet/caffenet_train_iter_500.caffemodel")
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
      val syncInterval = 50
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
