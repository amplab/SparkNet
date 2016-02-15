package libs

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
      case ArrayType(FloatType, true) => (element: Any) => {
        NDArray(element.asInstanceOf[Seq[Float]].toArray, shape)
      }
    }
  }
}
