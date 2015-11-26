package preprocessing

import java.awt.image.DataBufferByte
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO

import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConversions._
import net.coobird.thumbnailator._

import org.apache.spark.rdd.RDD

import libs.NDArray

class ScaleAndConvert(batchsize: Int, height: Int, width: Int) extends java.io.Serializable {
  def convertImage(compressedImg : Array[Byte]) : Option[NDArray] = {
    val im = ImageIO.read(new ByteArrayInputStream(compressedImg))
    try {
      val resizedImage = Thumbnails.of(im).forceSize(width, height).asBufferedImage()
      val pixels = resizedImage.getRGB(0, 0, width, height, null: Array[Int], 0, width)
      val array = new Array[Float](3 * width * height)
      assert(3 * pixels.length == array.length, "pixels.length = " + pixels.length.toString + ", array.length = " + array.length.toString)

      var index = 0
      var row = 0
      while (row < height) {
        var col = 0
        while (col < width) {
          val rgb = pixels(row * width + col)
          array(0 * width * height + row * width + col) = (rgb >> 16) & 0xFF // red
          array(1 * width * height + row * width + col) = (rgb >> 8) & 0xFF // green
          array(2 * width * height + row * width + col) = (rgb) & 0xFF // blue
          col += 1
        }
        row += 1
      }

      Some(NDArray(array, Array[Int](3, width, height)))
    } catch {
      // If images can't be processed properly, just ignore them
      case e: java.lang.IllegalArgumentException => None
      case e: javax.imageio.IIOException => None
      case e: java.lang.NullPointerException => None
    }
  }

  // This method will drop examples so that the number of training examples is divisible by the batch size
  def makeMinibatchRDD(data: RDD[(Array[Byte], Int)]) : RDD[(Array[NDArray], Array[Int])] = {
    data.mapPartitions(
      it => {
        val accumulator = new ArrayBuffer[(Array[NDArray], Array[Int])]
        // loop over minibatches
        while (it.hasNext) {
          val imageMinibatchAccumulator = new ArrayBuffer[NDArray]
          val labelMinibatchAccumulator = new ArrayBuffer[Int]
          while (it.hasNext && imageMinibatchAccumulator.length != batchsize) {
            val (compressedImg, label) = it.next
            convertImage(compressedImg) match {
              case Some(image) => {
                imageMinibatchAccumulator += image
                labelMinibatchAccumulator += label
              }
              case None => {}
            }
          }
          if (imageMinibatchAccumulator.length == batchsize) {
            accumulator += ((imageMinibatchAccumulator.toArray, labelMinibatchAccumulator.toArray))
          }
        }
        accumulator.iterator
      }
    )
  }

  def makeMinibatchRDDWithoutCompression(data: RDD[(Array[Float], Int)]) : RDD[(Array[NDArray], Array[Int])] = {
    data.mapPartitions(
      it => {
        val accumulator = new ArrayBuffer[(Array[NDArray], Array[Int])]
        while (it.hasNext) {
          val imageMinibatchAccumulator = new ArrayBuffer[NDArray]
          val labelMinibatchAccumulator = new ArrayBuffer[Int]
          while (it.hasNext && imageMinibatchAccumulator.length != batchsize) {
            val (image, label) = it.next
            imageMinibatchAccumulator += NDArray(image, Array[Int](3, width, height))
            labelMinibatchAccumulator += label
          }
          if (imageMinibatchAccumulator.length == batchsize) {
            accumulator += ((imageMinibatchAccumulator.toArray, labelMinibatchAccumulator.toArray))
          }
        }
        accumulator.iterator
      }
    )
  }

  def apply(data: RDD[(Array[Byte], Int)]) : RDD[(Array[NDArray], Array[Int])] = {
    makeMinibatchRDD(data)
  }
}
