package libs

class NDArray private(val javaArray: JavaNDArray) extends java.io.Serializable {
  val dim = javaArray.dim
  val shape = javaArray.shape

  def subarray(lowerOffsets: Array[Int], upperOffsets: Array[Int]): NDArray = {
    new NDArray(javaArray.subArray(lowerOffsets, upperOffsets))
  }

  def slice(axis: Int, index: Int): NDArray = {
    new NDArray(javaArray.slice(axis, index))
  }

  def get(indices: Array[Int]): Float = {
    javaArray.get(indices:_*)
  }

  def set(indices: Array[Int], value: Float) = {
    javaArray.set(indices, value)
  }

  def flatCopy(result: Array[Float]) = {
    javaArray.flatCopy(result)
  }

  def flatCopySlow(result: Array[Float]) = {
    javaArray.flatCopySlow(result)
  }

  def toFlat(): Array[Float] = {
    javaArray.toFlat()
  }

  def getBuffer(): Array[Float] = {
    javaArray.getBuffer()
  }

  def add(that: NDArray) = {
    javaArray.add(that.javaArray)
  }

  def subtract(that: NDArray) = {
    javaArray.subtract(that.javaArray)
  }

  def scalarDivide(v: Float) = {
    javaArray.scalarDivide(v)
  }

  override def toString() = {
    javaArray.toString()
  }
}

object NDArray {
  def apply(data: Array[Float], shape: Array[Int]) = {
    if (data.length != shape.product) {
      throw new IllegalArgumentException("The data and shape arguments are not compatible, data.length = " + data.length.toString + " and shape = " + shape.deep + ".\n")
    }
    new NDArray(new JavaNDArray(data, shape:_*))
  }

  def zeros(shape: Array[Int]) = new NDArray(new JavaNDArray(new Array[Float](shape.product), shape:_*))

  def plus(v1: NDArray, v2: NDArray): NDArray = {
    val v = new NDArray(new JavaNDArray(v1.toFlat(), v1.shape:_*))
    v.add(v2)
    v
  }

  def checkEqual(v1: NDArray, v2: NDArray, tol: Float): Boolean = {
    return v1.javaArray.equals(v2.javaArray, tol)
  }
}
