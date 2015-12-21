import org.scalatest._
import libs._
import java.io._

class WeightCollectionSpec extends FlatSpec {
  val sparkNetHome = sys.env("SPARKNET_HOME")
  System.load(sparkNetHome + "/build/libccaffe.so")
  val caffeLib = CaffeLibrary.INSTANCE

  val batchsize = 256
  val channels = 3
  val imgSize = 227

  var netParameter = ProtoLoader.loadNetPrototxt(sparkNetHome + "/caffe/models/bvlc_reference_caffenet/train_val.prototxt")
  netParameter = ProtoLoader.replaceDataLayers(netParameter, batchsize, batchsize, channels, imgSize, imgSize)
  val solverParameter = ProtoLoader.loadSolverPrototxtWithNet(sparkNetHome + "/caffe/models/bvlc_reference_caffenet/solver.prototxt", netParameter, None)
  val net = CaffeNet(caffeLib, solverParameter)
  var netWeights = net.getWeights()

  for (i <- 1 to 3) {
    val startTime = System.currentTimeMillis()
    netWeights = net.getWeights()
    val endTime = System.currentTimeMillis()
    print("getWeights() took " + (1F * (endTime - startTime) / 1000).toString + "s\n")
  }

  for (i <- 1 to 3) {
    val startTime = System.currentTimeMillis()
    net.setWeights(netWeights)
    val endTime = System.currentTimeMillis()
    print("setWeights() took " + (1F * (endTime - startTime) / 1000).toString + "s\n")
  }
}
