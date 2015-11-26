package libs

import scala.util.Random

import com.sun.jna.Pointer
import com.sun.jna.Memory
import scala.collection.mutable.Map
import scala.collection.mutable.MutableList

import caffe._
import caffe.Caffe._

import libs.CaffeLibrary
import libs.NDArray
import libs.MinibatchSampler

class WeightCollection(val allWeights: Map[String, MutableList[NDArray]], val layerNames: List[String]) extends java.io.Serializable {
  val numLayers = layerNames.length

  def scalarDivide(v: Float) = {
    for (i <- 0 to numLayers - 1) {
      for (j <- 0 to allWeights(layerNames(i)).length - 1) {
        allWeights(layerNames(i))(j).scalarDivide(v)
      }
    }
  }
}

object WeightCollection extends java.io.Serializable {
  def add(wc1: WeightCollection, wc2: WeightCollection): WeightCollection = {
    assert(wc1.layerNames == wc2.layerNames)
    val layerNames = wc1.layerNames
    //check that the WeightCollection objects have the same shape
    for (i <- 0 to wc1.numLayers - 1) {
      assert(wc1.allWeights(layerNames(i)).length == wc2.allWeights(layerNames(i)).length)
      for (j <- 0 to wc1.allWeights(layerNames(i)).length - 1) {
        assert(wc1.allWeights(layerNames(i))(j).shape.deep == wc2.allWeights(layerNames(i))(j).shape.deep)
      }
    }
    // add the WeightCollection objects together
    var newWeights = Map[String, MutableList[NDArray]]()
    for (i <- 0 to wc1.numLayers - 1) {
      newWeights += (layerNames(i) -> MutableList())
      for (j <- 0 to wc1.allWeights(wc1.layerNames(i)).length - 1) {
        newWeights(layerNames(i)) += NDArray.plus(wc1.allWeights(layerNames(i))(j), wc2.allWeights(layerNames(i))(j))
      }
    }
    return new WeightCollection(newWeights, layerNames)
  }
}

trait Net {
  def setTrainData(minibatchSampler: MinibatchSampler, trainPreprocessing: Option[NDArray => NDArray] = None)

  def setTestData(minibatchSampler: MinibatchSampler, len: Int, testPreprocessing: Option[NDArray => NDArray] = None)

  def train(numSteps: Int)

  def test(): Array[Float]

  def setWeights(weights: WeightCollection)

  def getWeights(): WeightCollection
}

class CaffeNet(state: Pointer, library: CaffeLibrary) extends Net {
  val numLayers = library.num_layers(state)
  val layerNames = List.range(0, numLayers).map(i => library.layer_name(state, i))
  val layerNumBlobs = List.range(0, numLayers).map(i => library.num_layer_weights(state, i))

  // store callbacks to save them from garbage collection
  var imageTrainCallback : Option[CaffeLibrary.java_callback_t] = None
  var labelTrainCallback : Option[CaffeLibrary.java_callback_t] = None
  var imageTestCallback : Option[CaffeLibrary.java_callback_t] = None
  var labelTestCallback : Option[CaffeLibrary.java_callback_t] = None

  val dtypeSize: Int = library.get_dtype_size()
  val intSize: Int = library.get_int_size()

  var numTestBatches = None: Option[Int]

  def setTrainData(minibatchSampler: MinibatchSampler, trainPreprocessing: Option[NDArray => NDArray] = None) = {
    imageTrainCallback = Some(makeImageCallback(minibatchSampler, trainPreprocessing))
    labelTrainCallback = Some(makeLabelCallback(minibatchSampler))
    library.set_train_data_callback(state, 0, imageTrainCallback.get)
    library.set_train_data_callback(state, 1, labelTrainCallback.get)
  }

  def setTestData(minibatchSampler: MinibatchSampler, len: Int, testPreprocessing: Option[NDArray => NDArray] = None) = {
    numTestBatches = Some(len)
    imageTestCallback = Some(makeImageCallback(minibatchSampler, testPreprocessing))
    labelTestCallback = Some(makeLabelCallback(minibatchSampler))
    library.set_test_data_callback(state, 0, imageTestCallback.get)
    library.set_test_data_callback(state, 1, labelTestCallback.get)
  }

  def train(numSteps: Int) = {
    library.set_mode_gpu()
    library.solver_step(state, numSteps)
  }

  def test(): Array[Float] = {
    library.set_mode_gpu()
    assert(!numTestBatches.isEmpty)
    library.solver_test(state, numTestBatches.get) // you must run this before running library.num_test_scores(state)
    val numTestScores = library.num_test_scores(state)
    val testScores = new Array[Float](numTestScores)
    for (i <- 0 to numTestScores - 1) {
      testScores(i) = library.get_test_score(state, i) // for accuracy layers, this returns the average accuracy over a minibatch
    }
    print("testScores = " + testScores.deep.toString + "\n")
    return testScores
  }

  def setWeights(allWeights: WeightCollection) = {
    assert(allWeights.numLayers == numLayers)
    for (i <- 0 to numLayers - 1) {
      assert(allWeights.allWeights(layerNames(i)).length == layerNumBlobs(i)) // check that we have the correct number of weights
      for (j <- 0 to layerNumBlobs(i) - 1) {
        val blob = library.get_weight_blob(state, i, j)
        val shape = getShape(blob)
        assert(shape.deep == allWeights.allWeights(layerNames(i))(j).shape.deep) // check that weights are the correct shape
        val flatWeights = allWeights.allWeights(layerNames(i))(j).toFlat() // todo: this allocation can be avoided
        val blob_pointer = library.get_data(blob)
        for (t <- 0 to shape.product - 1) {
          blob_pointer.setFloat(dtypeSize * t, flatWeights(t))
        }
      }
    }
  }

  def getWeights(): WeightCollection = {
    val allWeights = Map[String, MutableList[NDArray]]()
    for (i <- 0 to numLayers - 1) {
      val weightList = MutableList[NDArray]()
      for (j <- 0 to layerNumBlobs(i) - 1) {
        val blob = library.get_weight_blob(state, i, j)
        val shape = getShape(blob)
        val data = new Array[Float](shape.product)
        val blob_pointer = library.get_data(blob)
        for (t <- 0 to shape.product - 1) {
          data(t) = blob_pointer.getFloat(dtypeSize * t)
        }
        weightList += NDArray(data, shape)
      }
      allWeights += (layerNames(i) -> weightList)
    }
    return new WeightCollection(allWeights, layerNames)
  }

  private def makeImageCallback(minibatchSampler: MinibatchSampler, preprocessing: Option[NDArray => NDArray] = None): CaffeLibrary.java_callback_t = {
    return new CaffeLibrary.java_callback_t() {
      def invoke(data: Pointer, batchSize: Int, numDims: Int, shape: Pointer) {
        val currentImageBatch = minibatchSampler.nextImageMinibatch()
        assert(currentImageBatch.length == batchSize)

        // figure out what shape images Caffe expects
        val arrayShape = new Array[Int](numDims)
        for (i <- 0 to numDims - 1) {
          val dim = shape.getInt(i * intSize)
          arrayShape(i) = dim
        }

        for (j <- 0 to batchSize - 1) {
          val currentImage = currentImageBatch(j)
          val processedImage = {
            if (preprocessing.isEmpty) {
              currentImage
            } else {
              preprocessing.get(currentImage) // apply the preprocessing closure to the current image
            }
          }

          assert(arrayShape.deep == processedImage.shape.deep) // check that the image shape matches the shape Caffe expects
          val flatImage = processedImage.toFlat()
          val flatSize = flatImage.length
          for (i <- 0 to flatSize - 1) {
            data.setFloat((j * flatSize + i) * dtypeSize, flatImage(i))
          }
        }
      }
    }
  }

  def makeLabelCallback(minibatchSampler: MinibatchSampler): CaffeLibrary.java_callback_t = {
    return new CaffeLibrary.java_callback_t {
      def invoke(data: Pointer, batchSize: Int, numDims: Int, shape: Pointer) {
        val currentLabelBatch = minibatchSampler.nextLabelMinibatch()
        assert(currentLabelBatch.length == batchSize)
        for (j <- 0 to batchSize - 1) {
          data.setFloat(j * dtypeSize, 1F * currentLabelBatch(j))
        }
      }
    }
  }

  def loadWeightsFromFile(filename: String) {
    library.load_weights_from_file(state, filename)
  }

  private def getShape(blob: Pointer): Array[Int] = {
    val numAxes = library.get_num_axes(blob)
    val shape = new Array[Int](numAxes)
    for (k <- 0 to numAxes - 1) {
      shape(k) = library.get_axis_shape(blob, k)
    }
    return shape
  }
}

object CaffeNet {
  def apply(solverParameter: SolverParameter): CaffeNet = {
    val caffeLib = CaffeLibrary.INSTANCE
    val state = caffeLib.create_state()
    val byteArr = solverParameter.toByteArray()
    val ptr = new Memory(byteArr.length)
    ptr.write(0, byteArr, 0, byteArr.length)
    caffeLib.load_solver_from_protobuf(state, ptr, byteArr.length)
    return new CaffeNet(state, caffeLib)
  }
}
