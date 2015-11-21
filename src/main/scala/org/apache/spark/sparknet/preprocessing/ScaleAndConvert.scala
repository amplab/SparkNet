package org.apache.spark.sparknet.preprocessing

import org.apache.spark.sparknet.NDArray
import org.apache.spark.rdd.RDD

import java.io.ByteArrayInputStream
import javax.imageio.ImageIO
import net.coobird.thumbnailator._
import scala.collection.JavaConversions._

class ScaleAndConvert(height: Int, width: Int) extends java.io.Serializable {
	def apply(data: RDD[(Array[Byte], Int)]) : RDD[(NDArray, Int)] = {
		data.flatMap {
			case (compressed_img, label) => {
				val im = ImageIO.read(new ByteArrayInputStream(compressed_img))
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
					val ndarray = NDArray(array, Array[Int](3, 256, 256))
					Some((ndarray, label))
				} catch {
					// If images can't be processed properly, just ignore
					case e: java.lang.IllegalArgumentException => None
					case e: javax.imageio.IIOException => None
					case e: java.lang.NullPointerException => None
				}
			}
		}
	}
}
