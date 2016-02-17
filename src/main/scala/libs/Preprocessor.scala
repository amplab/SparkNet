package libs

import scala.util.Random

import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import scala.collection.mutable.ArrayBuffer

trait Preprocessor {
  def convert(name: String, shape: Array[Int]): Any => NDArray
}

class DefaultPreprocessor(schema: StructType) extends Preprocessor {
  def convert(name: String, shape: Array[Int]): Any => NDArray = {
    schema(name).dataType match {
      case FloatType => (element: Any) => {
        NDArray(Array[Float](element.asInstanceOf[Float]), shape)
      }
      case DoubleType => (element: Any) => {
        NDArray(Array[Float](element.asInstanceOf[Double].toFloat), shape)
      }
      case IntegerType => (element: Any) => {
        NDArray(Array[Float](element.asInstanceOf[Int].toFloat), shape)
      }
      case BinaryType => (element: Any) => {
        NDArray(element.asInstanceOf[Array[Byte]].map(e => e.toFloat), shape)
      }
      case ArrayType(FloatType, true) => (element: Any) => {
        NDArray(element.asInstanceOf[Seq[Float]].toArray, shape)
      }
    }
  }
}

class ImageNetPreprocessor(schema: StructType, meanImage: Array[Float], fullHeight: Int = 256, fullWidth: Int = 256, croppedHeight: Int = 227, croppedWidth: Int = 227) extends Preprocessor {
  def convert(name: String, shape: Array[Int]): Any => NDArray = {
    schema(name).dataType match {
      case IntegerType => (element: Any) => {
        NDArray(Array[Float](element.asInstanceOf[Int].toFloat), shape)
      }
      case BinaryType => (element: Any) => {
        val heightOffset = Random.nextInt(fullHeight - croppedHeight)
        val widthOffset = Random.nextInt(fullWidth - croppedWidth)
        val imageMinusMean = (element.asInstanceOf[Array[Byte]].map(e => e.toFloat), meanImage).zipped.map(_ - _)
        NDArray(imageMinusMean, shape).subarray(Array[Int](0, heightOffset, widthOffset), Array[Int](shape(0), heightOffset + croppedHeight, widthOffset + croppedWidth))
      }
    }
  }
}
