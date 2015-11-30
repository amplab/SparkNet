/*

Testing the speed of various ways to do callbacks; on pcm's laptop:

0.149: before empty callback
0.149: before copy callback
0.176: before simple callback
0.708: before byte image callback
0.78: before full callback
1.947: end

(these are cumulative numbers, so the elapsed times are the differences between
successive lines)

Takaway: We can copy stuff out of the array, but fancy indexing is very expensive,
and also setting floats via JNA in a loop is expensive

*/

import org.scalatest._
import java.io._
import scala.util.Random
import com.sun.jna.Pointer
import com.sun.jna.Memory


import libs._
import loaders._
import preprocessing._

class CallbackBenchmarkSpec extends FlatSpec {
  val trainBatchSize = 256
  val testBatchSize = 50
  val channels = 3
  val fullWidth = 256
  val fullHeight = 256
  val croppedWidth = 227
  val croppedHeight = 227

  val intSize = 4
  val dtypeSize = 4

  val startTime = System.currentTimeMillis()

  def log(message: String, i: Int = -1) {
    val elapsedTime = 1F * (System.currentTimeMillis() - startTime) / 1000
    if (i == -1) {
      print(elapsedTime.toString + ": "  + message + "\n")
    } else {
      print(elapsedTime.toString + ", i = " + i.toString + ": "+ message + "\n")
    }
  }

  val minibatch = Array.range(0, trainBatchSize).map(
    i => {
      ByteNDArray(new Array[Byte](channels * fullWidth * fullHeight), Array(channels, fullWidth, fullHeight))
    }
  ).toArray

  val byteImageMinibatch = Array.range(0, trainBatchSize).map(
    i => {
      new ByteImage(fullWidth, fullHeight)
    }
  ).toArray

  val emptyCallback = makeEmptyImageCallback(minibatch)
  val copyCallback = makeCopyImageCallback(minibatch)
  val simpleCallback = makeSimpleImageCallback(minibatch)
  val preprocessing = (image: ByteImage, buffer: Array[Float]) => {
    image.cropInto(buffer, Array[Int](0, 0), Array[Int](227, 227))
    var i = 0
    while (i < 227 * 227 * 3) {
      buffer(i) -= 100.0F
      i += 1
    }
    ()
  }
  val byteImageCallback = makeByteImageCallback(byteImageMinibatch, Some(preprocessing))
  val fullCallback = makeImageCallback(minibatch)
  val data = new Memory(channels * fullWidth * fullHeight * trainBatchSize * dtypeSize);

  log("before empty callback")
  emptyCallback.invoke(data, trainBatchSize, 3, new Pointer(0))
  log("before copy callback")
  copyCallback.invoke(data, trainBatchSize, 3, new Pointer(0))
  log("before simple callback")
  simpleCallback.invoke(data, trainBatchSize, 3, new Pointer(0))
  log("before byte image callback")
  byteImageCallback.invoke(data, trainBatchSize, 3, new Pointer(0))
  log("before full callback")
  fullCallback.invoke(data, trainBatchSize, 3, new Pointer(0))
  log("end")

  private def makeEmptyImageCallback(minibatch: Array[ByteNDArray], preprocessing: Option[ByteNDArray => NDArray] = None): CaffeLibrary.java_callback_t = {
    return new CaffeLibrary.java_callback_t() {
      def invoke(data: Pointer, batchSize: Int, numDims: Int, shape: Pointer) {
      }
    }
  }

  private def makeCopyImageCallback(minibatch: Array[ByteNDArray], preprocessing: Option[ByteNDArray => NDArray] = None): CaffeLibrary.java_callback_t = {
    return new CaffeLibrary.java_callback_t() {
      def invoke(data: Pointer, batchSize: Int, numDims: Int, shape: Pointer) {
        val currentImageBatch = minibatch
        var flatArray = new Array[Float](channels * fullWidth * fullHeight)
        for (j <- 0 to batchSize - 1) {
          val currentImage = currentImageBatch(j)
          currentImage.copyBufferToFloatArray(flatArray)
        }
      }
    }
  }

  private def makeSimpleImageCallback(minibatch: Array[ByteNDArray], preprocessing: Option[ByteNDArray => NDArray] = None): CaffeLibrary.java_callback_t = {
    return new CaffeLibrary.java_callback_t() {
      def invoke(data: Pointer, batchSize: Int, numDims: Int, shape: Pointer) {
        for (j <- 0 to batchSize - 1) {
          var i = 0
          val flatSize = channels * fullWidth * fullHeight
          while (i < flatSize) {
            data.setFloat((j * flatSize + i) * dtypeSize, 0.0F)
            i += 1
          }
        }
      }
    }
  }

  private def makeByteImageCallback(minibatch: Array[ByteImage], preprocessing: Option[(ByteImage, Array[Float]) => Unit] = None): CaffeLibrary.java_callback_t = {
    return new CaffeLibrary.java_callback_t() {
      def invoke(data: Pointer, batchSize: Int, numDims: Int, shape: Pointer) {
        var buffer = new Array[Float](227 * 227 * 3)
        for (j <- 0 to batchSize - 1) {
          preprocessing.get(minibatch(j), buffer)
          data.write(j * 227 * 227 * 3, buffer, 0, 227 * 227 * 3);
        }
      }
    }
  }

  private def makeImageCallback(minibatch: Array[ByteNDArray], preprocessing: Option[ByteNDArray => NDArray] = None): CaffeLibrary.java_callback_t = {
    return new CaffeLibrary.java_callback_t() {
      def invoke(data: Pointer, batchSize: Int, numDims: Int, shape: Pointer) {
        val currentImageBatch = minibatch
        assert(currentImageBatch.length == batchSize)

        for (j <- 0 to batchSize - 1) {
          val currentImage = currentImageBatch(j)
          val processedImage = {
            if (preprocessing.isEmpty) {
              currentImage.toFloatNDArray() // didn't test this code path
            } else {
              preprocessing.get(currentImage)
            }
          }

          val flatImage = processedImage.toFlat() // this allocation could be avoided
          val flatSize = flatImage.length
          var i = 0
          while (i < flatSize) {
            data.setFloat((j * flatSize + i) * dtypeSize, flatImage(i))
            i += 1
          }
        }
      }
    }
  }
}
