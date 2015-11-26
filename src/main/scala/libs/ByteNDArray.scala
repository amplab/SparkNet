package libs

class ByteNDArray private(val javaByteArray: JavaByteNDArray) extends java.io.Serializable {
  val dim = javaByteArray.dim
  val shape = javaByteArray.shape

  def subarray(lowerOffsets: Array[Int], upperOffsets: Array[Int]): ByteNDArray = {
    return new ByteNDArray(javaByteArray.subArray(lowerOffsets, upperOffsets))
  }

  def slice(axis: Int, index: Int): ByteNDArray = {
    return new ByteNDArray(javaByteArray.slice(axis, index))
  }

  def get(indices: Array[Int]): Byte = {
    return javaByteArray.get(indices:_*)
  }

  def set(indices: Array[Int], value: Byte) = {
    javaByteArray.set(indices, value)
  }

  def flatCopy(result: Array[Byte]) = {
    javaByteArray.flatCopy(result)
  }

  def toFlat(): Array[Byte] = {
    return javaByteArray.toFlat()
  }

  def getBuffer(): Array[Byte] = {
    return javaByteArray.getBuffer()
  }

  def copyBufferToFloatArray(result: Array[Float]) = {
    javaByteArray.copyBufferToFloatArray(result)
  }

  // didn't test this
  def toFloatNDArray(): NDArray = {
    val floatData = new Array[Float](shape.product)
    javaByteArray.flatCopyFloat(floatData)
    return NDArray(floatData, shape)
  }
}

object ByteNDArray {
  def apply(data: Array[Byte], shape: Array[Int]) = {
    new ByteNDArray(new JavaByteNDArray(data, shape:_*))
  }

  def zeros(shape: Array[Int]) = new ByteNDArray(new JavaByteNDArray(new Array[Byte](shape.product), shape:_*))
}
