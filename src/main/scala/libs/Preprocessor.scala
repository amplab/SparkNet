package libs

import scala.util.Random

import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import scala.collection.mutable._

trait CaffePreprocessor {
  def convert(name: String, shape: Array[Int]): Any => NDArray
}

trait TensorFlowPreprocessor {
  def convert(name: String, shape: Array[Int]): Any => Any
}

// The convert method in DefaultPreprocessor is used to convert data extracted
// from a dataframe into an NDArray, which can then be passed into a net. The
// implementation in DefaultPreprocessor is slow and does unnecessary
// allocation. This is designed to be easier to understand, whereas the
// ImageNetPreprocessor is designed to be faster.
class DefaultPreprocessor(schema: StructType) extends CaffePreprocessor {
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
      case LongType => (element: Any) => {
        NDArray(Array[Float](element.asInstanceOf[Long].toFloat), shape)
      }
      case BinaryType => (element: Any) => {
        NDArray(element.asInstanceOf[Array[Byte]].map(e => (e & 0xFF).toFloat), shape)
      }
      // case ArrayType(IntegerType, true) => (element: Any) => {} // TODO(rkn): implement
      case ArrayType(FloatType, true) => (element: Any) => {
        element match {
          case element: Array[Float] => NDArray(element.asInstanceOf[Array[Float]], shape)
          case element: WrappedArray[Float] => NDArray(element.asInstanceOf[WrappedArray[Float]].toArray, shape)
          case element: ArrayBuffer[Float] => NDArray(element.asInstanceOf[ArrayBuffer[Float]].toArray, shape)
        }
      }
      // case ArrayType(DoubleType, true) => (element: Any) => {} // TODO(rkn): implement
      // case ArrayType(LongType, true) => (element: Any) => {} // TODO(rkn): implement
    }
  }
}

class ImageNetPreprocessor(schema: StructType, meanImage: Array[Float], fullHeight: Int = 256, fullWidth: Int = 256, croppedHeight: Int = 227, croppedWidth: Int = 227) extends CaffePreprocessor {
  def convert(name: String, shape: Array[Int]): Any => NDArray = {
    schema(name).dataType match {
      case IntegerType => (element: Any) => {
        NDArray(Array[Float](element.asInstanceOf[Int].toFloat), shape)
      }
      case BinaryType => {
        if (shape(0) != 3) {
          throw new IllegalArgumentException("Expecting input image to have 3 channels.")
        }
        val buffer = new Array[Float](3 * fullHeight * fullWidth)
        (element: Any) => {
          element match {
            case element: Array[Byte] => {
              var index = 0
              while (index < 3 * fullHeight * fullWidth) {
                buffer(index) = (element(index) & 0XFF).toFloat - meanImage(index)
                index += 1
              }
            }
          }
          val heightOffset = Random.nextInt(fullHeight - croppedHeight + 1)
          val widthOffset = Random.nextInt(fullWidth - croppedWidth + 1)
          NDArray(buffer.clone, Array[Int](shape(0), fullHeight, fullWidth)).subarray(Array[Int](0, heightOffset, widthOffset), Array[Int](shape(0), heightOffset + croppedHeight, widthOffset + croppedWidth))
          // TODO(rkn): probably don't want to call buffer.clone
        }
      }
    }
  }
}

class DefaultTensorFlowPreprocessor(schema: StructType) extends TensorFlowPreprocessor {
  def convert(name: String, shape: Array[Int]): Any => Any = {
    schema(name).dataType match {
      case FloatType => (element: Any) => {
        Array[Float](element.asInstanceOf[Float])
      }
      case DoubleType => (element: Any) => {
        Array[Double](element.asInstanceOf[Double])
      }
      case IntegerType => (element: Any) => {
        Array[Int](element.asInstanceOf[Int])
      }
      case LongType => (element: Any) => {
        Array[Long](element.asInstanceOf[Long])
      }
      case BinaryType => (element: Any) => {
        element.asInstanceOf[Array[Byte]]
      }
      case ArrayType(FloatType, true) => (element: Any) => {
        element match {
          case element: Array[Float] => element.asInstanceOf[Array[Float]]
          case element: WrappedArray[Float] => element.asInstanceOf[WrappedArray[Float]].toArray
          case element: ArrayBuffer[Float] => element.asInstanceOf[ArrayBuffer[Float]].toArray
        }
      }
    }
  }
}
