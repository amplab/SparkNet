import org.scalatest._
import loaders.CifarLoader
import libs.CaffeLibrary
import libs.CaffeNet
import libs.ProtoLoader
import libs.Net
import com.sun.jna.Pointer
import com.sun.jna.Memory

// for this test to work, $SPARKNET_HOME/caffe should be the caffe root directory
// and you need to run $SPARKNET_HOME/caffe/data/cifar10/get_cifar10.sh
class CifarFeaturizationSpec extends FlatSpec {
  "CifarNet" should "get chance digits right on randomly initialized net" in {
    val sparkNetHome = sys.env("SPARKNET_HOME")
    val loader = new CifarLoader(sparkNetHome + "/caffe/data/cifar10/")

    System.load(sparkNetHome + "/build/libccaffe.so")
    val caffeLib = CaffeLibrary.INSTANCE

    caffeLib.set_basepath(sparkNetHome + "/caffe/")
    // val net = caffeLib.make_solver_from_prototxt(sparkNetHome + "/caffe/examples/cifar10/cifar10_full_java_solver.prototxt")

    val state = caffeLib.create_state()
    val solver = ProtoLoader.loadSolverPrototxt(sparkNetHome + "/caffe/examples/cifar10/cifar10_full_java_solver.prototxt")

    val byteArr = solver.toByteArray()
    val ptr = new Memory(byteArr.length);
    ptr.write(0, byteArr, 0, byteArr.length)
    caffeLib.load_solver_from_protobuf(state, ptr, byteArr.length)

    val dtypeSize = caffeLib.get_dtype_size()
    val intSize = caffeLib.get_int_size()

    def makeImageCallback(images: Array[Array[Byte]]) : CaffeLibrary.java_callback_t = {
      new CaffeLibrary.java_callback_t() {
        var currImage = 0
        def invoke(data: Pointer, batch_size: Int, num_dims: Int, shape: Pointer) {
          var size = 1
          for(i <- 0 to num_dims - 1) {
            val dim = shape.getInt(i * intSize)
            size *= dim
          }
          for(j <- 0 to batch_size - 1) {
            assert(size == images(currImage).length)
            for(i <- 0 to size - 1) {
              data.setFloat((j * size + i) * dtypeSize, 1F * (images(currImage)(i) & 0xFF))
            }
            currImage += 1
            if(currImage == images.length) {
              currImage = 0
            }
          }
         }
      };
    }

    def makeLabelCallback(labels: Array[Int]) : CaffeLibrary.java_callback_t =  {
      new CaffeLibrary.java_callback_t() {
        var currImage = 0
        def invoke(data: Pointer, batch_size: Int, num_dims: Int, shape: Pointer) {
          for(j <- 0 to batch_size - 1) {
            assert(shape.getInt(0) == 1)
            data.setFloat(j * dtypeSize, 1F * labels(currImage))
            currImage += 1
            if(currImage == labels.length) {
              currImage = 0
            }
          }
        }
      };
    }

    val loadTrainImageFn = makeImageCallback(loader.trainImages)
    val loadTrainLabelFn = makeLabelCallback(loader.trainLabels)
    caffeLib.set_train_data_callback(state, 0, loadTrainImageFn)
    caffeLib.set_train_data_callback(state, 1, loadTrainLabelFn)

    val loadTestImageFn = makeImageCallback(loader.testImages)
    val loadTestLabelFn = makeLabelCallback(loader.testLabels)
    caffeLib.set_test_data_callback(state, 0, loadTestImageFn)
    caffeLib.set_test_data_callback(state, 1, loadTestLabelFn)

    val net = new CaffeNet(state, caffeLib)
    net.forward()

    val data = net.getData

    val sortedKeys = data.keys.toArray.sorted
    assert(sortedKeys(0) == "conv1")
    assert(sortedKeys(1) == "conv2")
    assert(sortedKeys(2) == "conv3")
    assert(sortedKeys(3) == "data")
    assert(sortedKeys(4) == "ip1")
    assert(sortedKeys(5) == "label")
    assert(sortedKeys(6) == "loss")
    assert(sortedKeys(7) == "norm1")
    assert(sortedKeys(8) == "norm2")

    val shape = data("conv1").shape.deep
    assert(shape(0) == 100)
    assert(shape(1) == 32)
    assert(shape(2) == 32)
    assert(shape(3) == 32)
  }
}
