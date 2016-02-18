package preprocessing

import java.awt.image.DataBufferByte
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO

import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConversions._
import net.coobird.thumbnailator._

import org.apache.spark.rdd.RDD

import libs._

object ScaleAndConvert {
  def BufferedImageToByteArray(image: java.awt.image.BufferedImage) : Array[Byte] = {
    val height = image.getHeight()
    val width = image.getWidth()
    val pixels = image.getRGB(0, 0, width, height, null, 0, width)
    val result = new Array[Byte](3 * height * width)
    var row = 0
    while (row < height) {
      var col = 0
      while (col < width) {
        val rgb = pixels(row * width + col)
        result(0 * height * width + row * width + col) = ((rgb >> 16) & 0xFF).toByte
        result(1 * height * width + row * width + col) = ((rgb >> 8) & 0xFF).toByte
        result(2 * height * width + row * width + col) = (rgb & 0xFF).toByte
        col += 1
      }
      row += 1
    }
    result
  }

  def decompressImageAndResize(compressedImage: Array[Byte], height: Int, width: Int) : Option[Array[Byte]] = {
    // this method takes a JPEG, decompresses it, and resizes it
    try {
      val im = ImageIO.read(new ByteArrayInputStream(compressedImage))
      val resizedImage = Thumbnails.of(im).forceSize(width, height).asBufferedImage()
      Some(BufferedImageToByteArray(resizedImage))
    } catch {
      // If images can't be processed properly, just ignore them
      case e: java.lang.IllegalArgumentException => None
      case e: javax.imageio.IIOException => None
      case e: java.lang.NullPointerException => None
    }
  }
}
