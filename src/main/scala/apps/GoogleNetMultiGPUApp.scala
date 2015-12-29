package apps

import java.io._
import org.apache.commons.io.FileUtils
import scala.io.Source

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.storage.StorageLevel

import scala.concurrent._
import ExecutionContext.Implicits.global
import scala.concurrent.duration._

import libs._
import loaders._
import preprocessing._

object GoogleNetMultiGPUApp {
  val workerStore = new WorkerStore()

  def main(args: Array[String]) {
    val numWorkers = args(0).toInt
    val numGPUs = args(1).toInt
    val conf = new SparkConf()
      .setAppName("GoogleNetMultiGPU")
      .set("spark.driver.maxResultSize", "30G")
      .set("spark.task.maxFailures", "1")
      .set("spark.eventLog.enabled", "true")
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
    log("numWorkers = " + numWorkers.toString + ", numGPUs = " + numGPUs.toString)

    val workers = sc.parallelize(Array.range(0, numWorkers), numWorkers)

    log("initialize nets on workers")
    workers.foreach(_ => {
      System.load(sparkNetHome + "/build/libccaffe.so")
      val caffeLib = CaffeLibrary.INSTANCE
      caffeLib.set_basepath(sparkNetHome + "/caffe/")
      workerStore.setLib(caffeLib)
      var netParameter = ProtoLoader.loadNetPrototxt(sparkNetHome + "/caffe/models/bvlc_googlenet/train_val.prototxt")
      val solverParameter = ProtoLoader.loadSolverPrototxtWithNet(sparkNetHome + "/caffe/models/bvlc_googlenet/quick_solver.prototxt", netParameter, None)
      val net = CaffeNet(caffeLib, solverParameter, numGPUs)
      workerStore.setNet("net", net)
    })

    // initialize net on master
    System.load(sparkNetHome + "/build/libccaffe.so")
    val caffeLib = CaffeLibrary.INSTANCE
    caffeLib.set_basepath(sparkNetHome + "/caffe/")
    var netParameter = ProtoLoader.loadNetPrototxt(sparkNetHome + "/caffe/models/bvlc_googlenet/train_val.prototxt")
    val solverParameter = ProtoLoader.loadSolverPrototxtWithNet(sparkNetHome + "/caffe/models/bvlc_googlenet/quick_solver.prototxt", netParameter, None)
    val net = CaffeNet(caffeLib, solverParameter, numGPUs)
    net.loadWeightsFromFile(sparkNetHome + "/caffe/examples/imagenet/singlegpu_4000_init.caffemodel")

    var netWeights = net.getWeights()

    val testPartitionSizesFilename = sparkNetHome + "/infoFiles/imagenet_num_test_batches.txt"
    val numTestBatches = workers.map(_ => {
      Source.fromFile(testPartitionSizesFilename).getLines.mkString.toInt
    }).sum().toInt
    log("numTestBatches = " + numTestBatches.toString)
    var testAccuracy = None: Option[Future[Array[Float]]]

    var i = 0
    val syncInterval = 50
    while (true) {
      log("broadcasting weights", i)
      val broadcastWeights = sc.broadcast(netWeights)

      // TODO(pcmoritz): Currently, the weights are only updated/saved correctly
      // if they are saved twice as follows. I buy a six-pack of beer (or a
      // non-alcoholic drink of your choice) for whomever finds the problem.

      // save weights:
      if (i % 10 == 0) {
        if (!testAccuracy.isEmpty) {
          val testScores = Await.result(testAccuracy.get, Duration.Inf) // wait until testing finishes
          log("testScores = " + testScores.deep.toString, i)
          val accuracy = 100F * testScores(1) / numTestBatches // testScores(1) for GoogleNet and testScores(0) for AlexNet
          log("%.2f".format(accuracy) + "% accuracy", i - 10) // report the previous testing result
        }
        net.setWeights(netWeights)
        net.saveWeightsToFile("/root/weights/" + "%09d".format(i * syncInterval) + ".caffemodel")
        net.setNumTestBatches(numTestBatches)
        log("start testing", i)
        testAccuracy = Some(Future { net.test() }) // start testing asynchronously
        log("let testing run in background", i)
      }

      log("setting weights on workers", i)
      workers.foreach(_ => workerStore.getNet("net").setWeights(broadcastWeights.value))

      log("training", i)
      workers.foreachPartition(_ => workerStore.getNet("net").train(syncInterval))

      log("collecting weights", i)
      netWeights = workers.map(_ => workerStore.getNet("net").getWeights()).reduce((a, b) => WeightCollection.add(a, b))
      netWeights.scalarDivide(1F * numWorkers)

      // save weights:
      if (i % 10 == 0) {
        net.setWeights(netWeights)
        net.saveWeightsToFile("/root/weights/" + "%09d".format(i * syncInterval) + ".caffemodel2")
      }

      i += 1
    }

    log("finished training")
  }
}
