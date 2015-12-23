package apps

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.storage.StorageLevel

import libs._
import loaders._
import preprocessing._

object MultiGPUApp {
  val workerStore = new WorkerStore()

  def main(args: Array[String]) {
    val numWorkers = args(0).toInt
    val conf = new SparkConf()
      .setAppName("ImageNet")
      .set("spark.driver.maxResultSize", "30G")
      .set("spark.task.maxFailures", "1")
      .set("spark.eventLog.enabled", "true")
    val sc = new SparkContext(conf)

    val sparkNetHome = sys.env("SPARKNET_HOME")

    val workers = sc.parallelize(Array.range(0, numWorkers), numWorkers)

    // initialize nets on workers
    workers.foreach(_ => {
      System.load(sparkNetHome + "/build/libccaffe.so")
      val caffeLib = CaffeLibrary.INSTANCE
      caffeLib.set_basepath(sparkNetHome + "/caffe/")
      workerStore.setLib(caffeLib)
      var netParameter = ProtoLoader.loadNetPrototxt(sparkNetHome + "/caffe/models/bvlc_googlenet/train_val.prototxt")
      val solverParameter = ProtoLoader.loadSolverPrototxtWithNet(sparkNetHome + "/caffe/models/bvlc_googlenet/quick_solver.prototxt", netParameter, None)
      val net = CaffeNet(caffeLib, solverParameter, 4)
      workerStore.setNet("net", net)
    })

    // initialize net on master
    System.load(sparkNetHome + "/build/libccaffe.so")
    val caffeLib = CaffeLibrary.INSTANCE
    caffeLib.set_basepath(sparkNetHome + "/caffe/")
    var netParameter = ProtoLoader.loadNetPrototxt(sparkNetHome + "/caffe/models/bvlc_googlenet/train_val.prototxt")
    val solverParameter = ProtoLoader.loadSolverPrototxtWithNet(sparkNetHome + "/caffe/models/bvlc_googlenet/quick_solver.prototxt", netParameter, None)
    val net = CaffeNet(caffeLib, solverParameter, 4)

    var netWeights = net.getWeights()

    var i = 0
    val syncInterval = 50
    while (true) {
      val broadcastWeights = sc.broadcast(netWeights)

      // TODO(pcmoritz): Currently, the weights are only updated/saved correctly
      // if they are saved twice as follows. I buy a six-pack of beer (or a
      // non-alcoholic drink of your choice) for whomever finds the problem.

      // save weights:
      if (i % 10 == 0) {
        net.setWeights(netWeights)
        net.saveWeightsToFile("/imgnet/params/" + "%09d".format(i * syncInterval) + ".caffemodel")
      }

      workers.foreach(_ => net.setWeights(broadcastWeights.value))

      workers.foreachPartition(_ => net.train(syncInterval))

      netWeights = workers.map(_ => net.getWeights()).reduce((a, b) => WeightCollection.add(a, b))
      netWeights.scalarDivide(1F * numWorkers)

      // save weights:
      if (i % 10 == 0) {
        net.setWeights(netWeights)
        net.saveWeightsToFile("/imgnet/params/" + "%09d".format(i * syncInterval) + ".caffemodel")
      }

      i += 1
    }
  }
}
