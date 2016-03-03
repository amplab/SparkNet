package libs

import java.nio.FloatBuffer

import scala.collection.mutable._
import org.bytedeco.javacpp.tensorflow._

object TensorFlowUtils {
  def checkStatus(s: Status) = {
    if (!s.ok) {
      throw new Exception("TensorFlow error:\n" + s.error_message.getString)
    }
  }

  def getTensorShape(t: Tensor): Array[Int] = {
    Array.range(0, t.dims).map(i => t.shape.dim_sizes.get(i).toInt)
  }

  def getTensorShape(sp: TensorShapeProto): Array[Int] = {
    Array.range(0, sp.dim_size).map(i => sp.dim(i).size.toInt)
  }

  def getNodeType(node: NodeDef): Int = {
    val attrMap = getAttributeMap(node)
    attrMap("dtype").`type` // type is a Scala keyword, so we need to use the backticks
  }

  def getNodeShape(node: NodeDef): Array[Int] = {
    val attrMap = getAttributeMap(node)
    getTensorShape(attrMap("shape").shape)
  }

  def getAttributeMap(node: NodeDef): Map[String, AttrValue] = {
    val attributes = node.attr
    val result = Map[String, AttrValue]()
    var curr = attributes.begin
    for (i <- 0 to node.attr_size - 1) {
      result += (curr.first.getString -> curr.second)
      curr = curr.increment
    }
    result
  }

  def newBuffer(dtype: Int, size: Int): Any = {
    dtype match {
      case DT_FLOAT => new Array[Float](size)
      case DT_INT32 => new Array[Int](size)
      case DT_INT64 => new Array[Long](size)
      case DT_DOUBLE => new Array[Double](size)
      case DT_UINT8 => new Array[Byte](size)
    }
  }

  def tensorToNDArray(t: Tensor): NDArray = {
    val shape = getTensorShape(t)
    val data = new Array[Float](shape.product)
    val buffer = TensorFlowHelper.createFloatBuffer(t)
    var i = 0
    while (i < data.length) {
      data(i) = buffer.get(i)
      i += 1
    }
    NDArray(data, shape)
  }

  def tensorFromNDArray(t: Tensor, array: NDArray) = {
    if (getTensorShape(t).deep != array.shape.deep) {
      throw new Exception("The shape of `t` does not match the shape of `array`. `t` has shape " + getTensorShape(t).deep.toString + " and array has shape " + array.shape.deep.toString + "\n")
    }
    val buffer = TensorFlowHelper.createFloatBuffer(t)
    val flatArray = array.toFlat() // TODO(rkn): this is inefficient, fix it
    var i = 0
    while (i < flatArray.length) {
      buffer.put(i, flatArray(i))
      i += 1
    }
  }

  def tensorFromFlatArray(t: Tensor, a: Any, offsetInT: Int = 0, offsetInA: Int = 0, length: Int = -1) = {
    // Copy `array` starting at index `offsetInA` into `t` starting at the offset `offsetInT` in `t` for length `length`.
    t.dtype match {
      case DT_FLOAT =>
        try {
          a.asInstanceOf[Array[Float]]
        } catch {
          case e: Exception => throw new Exception("Tensor t has type DT_FLOAT, but `a` cannot be cast to Array[Float], `a` has type ???")
        }
      case DT_INT32 =>
        try {
          a.asInstanceOf[Array[Int]]
        } catch {
          case e: Exception => throw new Exception("Tensor t has type DT_INT32, but `a` cannot be cast to Array[Int], `a` has type ???")
        }
      case DT_INT64 =>
        try {
          a.asInstanceOf[Array[Long]]
        } catch {
          case e: Exception => throw new Exception("Tensor t has type DT_INT64, but `a` cannot be cast to Array[Long], `a` has type ???")
        }
      case DT_DOUBLE =>
        try {
          a.asInstanceOf[Array[Double]]
        } catch {
          case e: Exception => throw new Exception("Tensor t has type DT_DOUBLE, but `a` cannot be cast to Array[Double], `a` has type ???")
        }
      case DT_UINT8 =>
        try {
          a.asInstanceOf[Array[Byte]]
        } catch {
          case e: Exception => throw new Exception("Tensor t has type DT_UINT8, but `a` cannot be cast to Array[Byte], `a` has type ???")
        }
    }

    val len = t.dtype match {
      case DT_FLOAT => a.asInstanceOf[Array[Float]].length
      case DT_INT32 => a.asInstanceOf[Array[Int]].length
      case DT_INT64 => a.asInstanceOf[Array[Long]].length
      case DT_DOUBLE => a.asInstanceOf[Array[Double]].length
      case DT_UINT8 => a.asInstanceOf[Array[Byte]].length
    }

    val size = if (length == -1) len else length
    val tShape = getTensorShape(t)
    if (offsetInA + size > len) {
      throw new Exception("`offsetInA` + `size` exceeds the size of `a`. offsetInA = " + offsetInA.toString + ", size = " + size.toString + ", and a.length = " + len.toString + "\n")
    }
    if (offsetInT + size > tShape.product) {
      throw new Exception("`offsetInT` + `size` exceeds the size of `t`. offsetInT = " + offsetInT.toString + ", size = " + size.toString + ", and the size of `t` is " + tShape.product.toString + "\n")
    }

    t.dtype match {
      case DT_FLOAT => {
        val array = a.asInstanceOf[Array[Float]]
        val buffer = TensorFlowHelper.createFloatBuffer(t)
        var i = 0
        while (i < size) {
          buffer.put(offsetInT + i, array(offsetInA + i))
          i += 1
        }
      }
      case DT_INT32 => {
        val array = a.asInstanceOf[Array[Int]]
        val buffer = TensorFlowHelper.createIntBuffer(t)
        var i = 0
        while (i < size) {
          buffer.put(offsetInT + i, array(offsetInA + i))
          i += 1
        }
      }
      case DT_INT64 => {
        val array = a.asInstanceOf[Array[Long]]
        val buffer = TensorFlowHelper.createLongBuffer(t)
        var i = 0
        while (i < size) {
          buffer.put(offsetInT + i, array(offsetInA + i))
          i += 1
        }
      }
      case DT_DOUBLE => {
        val array = a.asInstanceOf[Array[Double]]
        val buffer = TensorFlowHelper.createDoubleBuffer(t)
        var i = 0
        while (i < size) {
          buffer.put(offsetInT + i, array(offsetInA + i))
          i += 1
        }
      }
      case DT_UINT8 => {
        val array = a.asInstanceOf[Array[Byte]]
        val buffer = TensorFlowHelper.createByteBuffer(t)
        var i = 0
        while (i < size) {
          buffer.put(offsetInT + i, array(offsetInA + i))
          i += 1
        }
      }
    }

  }

}
