package apps

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.storage.StorageLevel

import libs._
import loaders._
import preprocessing._

object MultiGPUApp {
  // initialize nets on workers
  val sparkNetHome = "/root/SparkNet"
  System.load(sparkNetHome + "/build/libccaffe.so")
	val caffeLib = CaffeLibrary.INSTANCE
	caffeLib.set_basepath(sparkNetHome + "/caffe/")
  var netParameter = ProtoLoader.loadNetPrototxt(sparkNetHome + "/caffe/models/bvlc_googlenet/train_val.prototxt")
  val solverParameter = ProtoLoader.loadSolverPrototxtWithNet(sparkNetHome + "/caffe/models/bvlc_googlenet/quick_solver.prototxt", netParameter, None)
  val net = CaffeNet(solverParameter, 4) // TODO: Fix 1 not working here

  def main(args: Array[String]) {
    val numWorkers = args(0).toInt
    val conf = new SparkConf()
      .setAppName("ImageNet")
      .set("spark.driver.maxResultSize", "30G")
      .set("spark.task.maxFailures", "1")
      .set("spark.eventLog.enabled", "true")
    val sc = new SparkContext(conf)

    var netWeights = net.getWeights()
    val workers = sc.parallelize(Array.range(0, numWorkers), numWorkers)

    val caffeLib = CaffeLibrary.INSTANCE
    val state = net.getState()

    var i = 0
    while (true) {
      val broadcastWeights = sc.broadcast(netWeights)

      workers.foreach(_ => net.setWeights(broadcastWeights.value))

      val syncInterval = 50
      workers.foreachPartition(
        _ => {
          net.train(syncInterval)
          ()
        }
      )

      netWeights = workers.map(_ => net.getWeights()).reduce((a, b) => WeightCollection.add(a, b))
      netWeights.scalarDivide(1F * numWorkers)

      // save weights:
      if (i % 500 == 0) {
        net.setWeights(netWeights)
        caffeLib.save_weights_to_file(state, "/imgnet/params/" + "%09d".format(i) + ".caffemodel")
      }

      i += 50
    }
  }
}
