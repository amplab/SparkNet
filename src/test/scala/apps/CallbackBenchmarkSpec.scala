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

  val byteImageMinibatch = Array.range(0, trainBatchSize).map(
    i => {
      new ByteImage(fullWidth, fullHeight)
    }
  ).toArray

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
  val data = new Memory(channels * fullWidth * fullHeight * trainBatchSize * dtypeSize);

  log("before byte image callback")
  byteImageCallback.invoke(data, trainBatchSize, 3, new Pointer(0))
  log("before full callback")

  private def makeByteImageCallback(minibatch: Array[ByteImage], preprocessing: Option[(ByteImage, Array[Float]) => Unit] = None): CaffeLibrary.java_callback_t = {
    return new CaffeLibrary.java_callback_t() {
      def invoke(data: Pointer, batchSize: Int, numDims: Int, shape: Pointer) {
        var buffer = new Array[Float](227 * 227 * 3 * batchSize)
        for (j <- 0 to batchSize - 1) {
          preprocessing.get(minibatch(j), buffer)
          val dtypeSize = 4
          data.write(j * 227 * 227 * 3 * dtypeSize, buffer, 0, 227 * 227 * 3);
        }
      }
    }
  }
}
