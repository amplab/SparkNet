/*

Testing the speed of various ways to do callbacks; on pcm's laptop:

0.118: before empty callback
0.118: before copy callback
0.148: before simple callback
0.709: before full callback
1.753: end

Takaway: We can copy stuff, but fancy indexing is very expensive

*/

import org.scalatest._
import java.io._
import scala.util.Random
import com.sun.jna.Pointer
import com.sun.jna.Memory


import libs._
import loaders._
import preprocessing._

class ImageNetBenchmarkSpec extends FlatSpec {
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

  "ImageNetCallback" should "be fast" in {
    val minibatch = Array.range(0, trainBatchSize).map(
      i => {
        ByteNDArray(new Array[Byte](channels * fullWidth * fullHeight), Array(channels, fullWidth, fullHeight))
      }
    ).toArray

    val emptyCallback = makeEmptyImageCallback(minibatch)
    val copyCallback = makeCopyImageCallback(minibatch)
    val simpleCallback = makeSimpleImageCallback(minibatch)
    val fullCallback = makeImageCallback(minibatch)
    val data = new Memory(channels * fullWidth * fullHeight * trainBatchSize * dtypeSize);

    log("before empty callback")
    emptyCallback.invoke(data, trainBatchSize, 3, new Pointer(0))
    log("before copy callback")
    copyCallback.invoke(data, trainBatchSize, 3, new Pointer(0))
    log("before simple callback")
    simpleCallback.invoke(data, trainBatchSize, 3, new Pointer(0))
    log("before full callback")
    fullCallback.invoke(data, trainBatchSize, 3, new Pointer(0))
    log("end")
  }

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
          // currentImage.flatCopy(flatArray)
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
