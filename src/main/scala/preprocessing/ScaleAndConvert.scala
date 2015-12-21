package preprocessing

import java.awt.image.DataBufferByte
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO

import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConversions._
import net.coobird.thumbnailator._

import org.apache.spark.rdd.RDD

import libs._

class ScaleAndConvert(batchsize: Int, height: Int, width: Int) extends java.io.Serializable {
  def convertImage(compressedImg : Array[Byte]) : Option[ByteImage] = {
    val im = ImageIO.read(new ByteArrayInputStream(compressedImg))
    try {
      val resizedImage = Thumbnails.of(im).forceSize(width, height).asBufferedImage()
      Some(new ByteImage(resizedImage))
    } catch {
      // If images can't be processed properly, just ignore them
      case e: java.lang.IllegalArgumentException => None
      case e: javax.imageio.IIOException => None
      case e: java.lang.NullPointerException => None
    }
  }

  def scaleAndConvertWithCompression(data: RDD[(Array[Byte], Int)]) : RDD[(ByteImage, Int)] = {
    data.flatMap{
      case (compressedImage, label) => {
        convertImage(compressedImage) match {
          case Some(image) => Seq((image, label))
          case None => Seq[(ByteImage, Int)]()
        }
      }
    }
  }

  def scaleAndConvertWithoutCompression(data: RDD[(Array[Byte], Int)]) : RDD[(ByteImage, Int)] = {
    data.map{ case (image, label) => (new ByteImage(image, height, width), label) }
  }

  // This method will drop examples so that the number of training examples is divisible by the batch size
  def makeMinibatchRDDWithCompression(data: RDD[(Array[Byte], Int)]) : RDD[(Array[ByteImage], Array[Int])] = {
    data.mapPartitions(
      it => {
        val accumulator = new ArrayBuffer[(Array[ByteImage], Array[Int])]
        // loop over minibatches
        while (it.hasNext) {
          val imageMinibatchAccumulator = new ArrayBuffer[ByteImage]
          val labelMinibatchAccumulator = new ArrayBuffer[Int]
          while (it.hasNext && imageMinibatchAccumulator.length != batchsize) {
            val (compressedImage, label) = it.next
            convertImage(compressedImage) match {
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

  def makeMinibatchRDDWithoutCompression(data: RDD[(Array[Byte], Int)]) : RDD[(Array[ByteImage], Array[Int])] = {
    data.mapPartitions(
      it => {
        val accumulator = new ArrayBuffer[(Array[ByteImage], Array[Int])]
        while (it.hasNext) {
          val imageMinibatchAccumulator = new ArrayBuffer[ByteImage]
          val labelMinibatchAccumulator = new ArrayBuffer[Int]
          while (it.hasNext && imageMinibatchAccumulator.length != batchsize) {
            val (image, label) = it.next
            imageMinibatchAccumulator += new ByteImage(image, height, width)
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
}
