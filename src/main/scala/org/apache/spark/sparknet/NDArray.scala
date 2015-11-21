package org.apache.spark.sparknet

class NDArray private(val javaArray: JavaNDArray) extends java.io.Serializable {
  val dim = javaArray.dim
  val shape = javaArray.shape

  def subarray(lowerOffsets: Array[Int], upperOffsets: Array[Int]): NDArray = {
    return new NDArray(javaArray.subArray(lowerOffsets, upperOffsets))
  }

  def slice(axis: Int, index: Int): NDArray = {
    return new NDArray(javaArray.slice(axis, index))
  }

  def get(indices: Array[Int]): Float = {
    return javaArray.get(indices:_*)
  }

  def set(indices: Array[Int], value: Float) = {
    javaArray.set(indices, value)
  }

  def flatCopy(result: Array[Float]) = {
    javaArray.flatCopy(result)
  }

  def toFlat(): Array[Float] = {
    return javaArray.toFlat()
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
}

object NDArray {
  def apply(data: Array[Float], shape: Array[Int]) = {
    new NDArray(new JavaNDArray(data, shape:_*))
  }

  def zeros(shape: Array[Int]) = new NDArray(new JavaNDArray(new Array[Float](shape.product), shape:_*))

  def plus(v1: NDArray, v2: NDArray): NDArray = {
    val v = new NDArray(new JavaNDArray(v1.toFlat(), v1.shape:_*))
    v.add(v2)
    return v
  }
}
