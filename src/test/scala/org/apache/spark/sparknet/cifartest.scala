import org.scalatest._
import org.apache.spark.sparknet.loaders.CifarLoader
import org.apache.spark.sparknet.CaffeLibrary
import com.sun.jna.Pointer

// for this test to work, $SPARKNET_HOME/caffe should be the caffe root directory
// and you need to run $SPARKNET_HOME/caffe/data/cifar10/get_cifar10.sh
class CifarSpec extends FlatSpec {
	ignore should "get chance digits right on randomly initialized net" in {
		val sparkNetHome = sys.env("SPARKNET_HOME")
		val loader = new CifarLoader(sparkNetHome + "/caffe/data/cifar10/")

		System.load(sparkNetHome + "/build/libccaffe.so")
		val caffeLib = CaffeLibrary.INSTANCE

		caffeLib.set_basepath(sparkNetHome + "/caffe/")
		val net = caffeLib.make_solver_from_prototxt(sparkNetHome + "/caffe/examples/cifar10/cifar10_full_java_solver.prototxt")

		val dtypeSize = caffeLib.get_dtype_size()
		val intSize = caffeLib.get_int_size()

		def makeImageCallback(images: Array[Array[Float]]) : CaffeLibrary.java_callback_t = {
  		return new CaffeLibrary.java_callback_t() {
  			var currImage = 0
				def invoke(data: Pointer, batch_size: Int, num_dims: Int, shape: Pointer) {
					var size = 1
					for(i <- 0 to num_dims-1) {
						val dim = shape.getInt(i * intSize)
						size *= dim
					}
					for(j <- 0 to batch_size-1) {
						assert(size == images(currImage).length)
						for(i <- 0 to size-1) {
							data.setFloat((j * size + i) * dtypeSize, images(currImage)(i))
						}
						currImage += 1
						if(currImage == images.length) {
							currImage = 0
						}
					}
	 			}
			};
		}

		def makeLabelCallback(labels: Array[Float]) : CaffeLibrary.java_callback_t =  {
			return new CaffeLibrary.java_callback_t() {
				var currImage = 0
				def invoke(data: Pointer, batch_size: Int, num_dims: Int, shape: Pointer) {
	  			for(j <- 0 to batch_size-1) {
						assert(shape.getInt(0) == 1)
						data.setFloat(j * dtypeSize, labels(currImage))
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
		caffeLib.set_train_data_callback(net, 0, loadTrainImageFn)
		caffeLib.set_train_data_callback(net, 1, loadTrainLabelFn)

		val loadTestImageFn = makeImageCallback(loader.testImages)
		val loadTestLabelFn = makeLabelCallback(loader.testLabels)
		caffeLib.set_test_data_callback(net, 0, loadTestImageFn)
		caffeLib.set_test_data_callback(net, 1, loadTestLabelFn)

		caffeLib.solver_test(net, 10) // TODO: shouldn't be hard coded

		val numTestScores = caffeLib.num_test_scores(net)

		val testScores = new Array[Float](numTestScores)

		// perform test on random net
		for (i <- 0 to numTestScores - 1) {
			testScores(i) = caffeLib.get_test_score(net, i) * 100 // TODO: this batch size shouldn't be hard coded
		}

		assert(70.0 <= testScores(0) && testScores(0) <= 130.0)
	}
}
