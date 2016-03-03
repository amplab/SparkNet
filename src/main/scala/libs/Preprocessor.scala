package libs

import scala.util.Random

import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import scala.collection.mutable._

trait Preprocessor {
  def convert(name: String, shape: Array[Int]): Any => NDArray
}

trait TensorFlowPreprocessor {
  def convert(name: String, shape: Array[Int]): (Any, Any) => Unit
}

// The convert method in DefaultPreprocessor is used to convert data extracted
// from a dataframe into an NDArray, which can then be passed into a net. The
// implementation in DefaultPreprocessor is slow and does unnecessary
// allocation. This is designed to be easier to understand, whereas the
// ImageNetPreprocessor is designed to be faster.
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

class ImageNetPreprocessor(schema: StructType, meanImage: Array[Float], fullHeight: Int = 256, fullWidth: Int = 256, croppedHeight: Int = 227, croppedWidth: Int = 227) extends Preprocessor {
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
                buffer(index) = (element(index) & 0xFF).toFloat - meanImage(index)
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
  def convert(name: String, shape: Array[Int]): (Any, Any) => Unit = {
    schema(name).dataType match {
      case FloatType => (element: Any, buffer: Any) => {
        val e = element.asInstanceOf[Float]
        val b = buffer.asInstanceOf[Array[Float]]
        b(0) = e
      }
      case DoubleType => (element: Any, buffer: Any) => {
        val e = element.asInstanceOf[Double]
        val b = buffer.asInstanceOf[Array[Double]]
        b(0) = e
      }
      case IntegerType => (element: Any, buffer: Any) => {
        val e = element.asInstanceOf[Int]
        val b = buffer.asInstanceOf[Array[Int]]
        b(0) = e
      }
      case LongType => (element: Any, buffer: Any) => {
        val e = element.asInstanceOf[Long]
        val b = buffer.asInstanceOf[Array[Long]]
        b(0) = e
      }
      case BinaryType => (element: Any, buffer: Any) => {
        val e = element.asInstanceOf[Array[Byte]]
        val b = buffer.asInstanceOf[Array[Byte]]
        var i = 0
        while (i < b.length) {
          b(i) = e(i)
          i += 1
        }
      }
      case ArrayType(FloatType, true) => (element: Any, buffer: Any) => {
        val b = buffer.asInstanceOf[Array[Float]]
        element match {
          case element: Array[Float] => {
            val e = element.asInstanceOf[Array[Float]]
            var i = 0
            while (i < b.length) {
              b(i) = e(i)
              i += 1
            }
          }
          case element: WrappedArray[Float] => {
            val e = element.asInstanceOf[WrappedArray[Float]]
            var i = 0
            while (i < b.length) {
              b(i) = e(i)
              i += 1
            }
          }
          case element: ArrayBuffer[Float] => {
            val e = element.asInstanceOf[ArrayBuffer[Float]]
            var i = 0
            while (i < b.length) {
              b(i) = e(i)
              i += 1
            }
          }
        }
      }
    }
  }
}

class ImageNetTensorFlowPreprocessor(schema: StructType, meanImage: Array[Float], fullHeight: Int = 256, fullWidth: Int = 256, croppedHeight: Int = 227, croppedWidth: Int = 227) extends TensorFlowPreprocessor {
  def convert(name: String, shape: Array[Int]): (Any, Any) => Unit = {
    if (name == "label") {
      (element: Any, buffer: Any) => {
        val e = element.asInstanceOf[Int]
        val b = buffer.asInstanceOf[Array[Int]]
        b(0) = e
      }
    } else if (name == "data") {
      val tempBuffer = new Array[Float](fullHeight * fullWidth * 3)
      (element: Any, buffer: Any) => {
        val e = element.asInstanceOf[Array[Byte]]
        val b = buffer.asInstanceOf[Array[Float]]
        var index = 0
        while (index < fullHeight * fullWidth) {
          tempBuffer(3 * index + 0) = (e(0 * fullHeight * fullWidth + index) & 0xFF).toFloat - meanImage(0 * fullHeight * fullWidth + index)
          tempBuffer(3 * index + 1) = (e(1 * fullHeight * fullWidth + index) & 0xFF).toFloat - meanImage(1 * fullHeight * fullWidth + index)
          tempBuffer(3 * index + 2) = (e(2 * fullHeight * fullWidth + index) & 0xFF).toFloat - meanImage(2 * fullHeight * fullWidth + index)
          index += 1
        }
        val heightOffset = Random.nextInt(fullHeight - croppedHeight + 1)
        val widthOffset = Random.nextInt(fullWidth - croppedWidth + 1)
        NDArray(tempBuffer, Array[Int](fullHeight, fullWidth, shape(2))).subarray(Array[Int](heightOffset, widthOffset, 0), Array[Int](heightOffset + croppedHeight, widthOffset + croppedWidth, shape(2))).flatCopy(b)
      }
    } else {
      throw new Exception("The name is not `label` or `data`, name = " + name + "\n")
    }
  }
}
