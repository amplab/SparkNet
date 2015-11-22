package preprocessing

import libs.NDArray
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO
import net.coobird.thumbnailator._
import scala.collection.JavaConversions._

class ScaleAndConvert(batchsize: Int, height: Int, width: Int) extends java.io.Serializable {
  def convertImage(compressedImg : Array[Byte]) : Option[NDArray] = {
    val im = ImageIO.read(new ByteArrayInputStream(compressedImg))
    try {
      val resized_img = Thumbnails.of(im).forceSize(width, height).asBufferedImage()
      val array = new Array[Float](3 * width * height)
      for (col <- 0 to width - 1) {
        for (row <- 0 to height - 1) {
          val rgb = resized_img.getRGB(col, row)
          array(0 * width * height + row * width + col) = (rgb >> 16) & 0xFF // red
          array(1 * width * height + row * width + col) = (rgb >> 8) & 0xFF // green
          array(2 * width * height + row * width + col) = (rgb) & 0xFF // blue
        }
      }
      Some(NDArray(array, Array[Int](3, width, height)))
    } catch {
      // If images can't be processed properly, just ignore
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
        // loop over minibatches:
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
          accumulator += ((imageMinibatchAccumulator.toArray, labelMinibatchAccumulator.toArray))
        }
        accumulator.iterator
      }
    )
  }
  def apply(data: RDD[(Array[Byte], Int)]) : RDD[(Array[NDArray], Array[Int])] = {
    makeMinibatchRDD(data)
  }
}
