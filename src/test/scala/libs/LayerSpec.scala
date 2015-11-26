import org.scalatest._
import libs._
import java.io._
import caffe._
import caffe.Caffe._

import com.sun.jna.Pointer;
import com.sun.jna.Memory;

class LayerSpec extends FlatSpec {
  "Layer definition" should "should be loadable" in {
    val sparkNetHome = sys.env("SPARKNET_HOME")
    System.load(sparkNetHome + "/build/libccaffe.so")
    val caffeLib = CaffeLibrary.INSTANCE

    val batchsize = 256

    val netParam = NetParam ("LeNet",
      RDDLayer("data", shape=List(batchsize, 1, 28, 28), None),
      RDDLayer("label", shape=List(batchsize, 1), None),
      ConvolutionLayer("conv1", List("data"), kernel=(5,5), numOutput=20),
      PoolingLayer("pool1", List("conv1"), pooling=Pooling.Max, kernel=(2,2), stride=(2,2)),
      ConvolutionLayer("conv2", List("pool1"), kernel=(5,5), numOutput=50),
      PoolingLayer("pool2", List("conv2"), pooling=Pooling.Max, kernel=(2,2), stride=(2,2)),
      InnerProductLayer("ip1", List("pool2"), numOutput=500),
      ReLULayer("relu1", List("ip1")),
      InnerProductLayer("ip2", List("relu1"), numOutput=10),
      SoftmaxWithLoss("loss", List("ip2", "label"))
    )

    print(netParam.toString)

    val lenetState = caffeLib.create_state()

    var byteArr = netParam.toByteArray()
    var ptr = new Memory(byteArr.length);
    ptr.write(0, byteArr, 0, byteArr.length)
    caffeLib.load_net_from_protobuf(lenetState, ptr, byteArr.length)

    var netParameter = ProtoLoader.loadNetPrototxt(sparkNetHome + "/caffe/models/bvlc_reference_caffenet/train_val.prototxt")
    netParameter = ProtoLoader.replaceDataLayers(netParameter, batchsize, 100, 3, 227, 227)
    val solverParameter = ProtoLoader.loadSolverPrototxtWithNet(sparkNetHome + "/caffe/models/bvlc_reference_caffenet/solver.prototxt", netParameter, None)

    val state = caffeLib.create_state()

    byteArr = solverParameter.toByteArray()
    ptr = new Memory(byteArr.length);
    ptr.write(0, byteArr, 0, byteArr.length)
    caffeLib.load_solver_from_protobuf(state, ptr, byteArr.length)
	}
}
