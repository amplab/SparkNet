package libs

import org.bytedeco.javacpp._
import org.bytedeco.javacpp.caffe._

object JavaCPPUtils {
  def floatBlobToNDArray(floatBlob: FloatBlob): NDArray = {
    val shape = getFloatBlobShape(floatBlob)
    val data = new Array[Float](shape.product)
    val pointer = floatBlob.cpu_data
    var i = 0
    while (i < shape.product) {
      data(i) = pointer.get(i)
      i += 1
    }
    NDArray(data, shape)
  }

  def getFloatBlobShape(floatBlob: FloatBlob): Array[Int] = {
    val numAxes = floatBlob.num_axes()
    val shape = new Array[Int](numAxes)
    for (k <- 0 to numAxes - 1) {
      shape(k) = floatBlob.shape.get(k)
    }
    shape
  }

  def getInputShape(netParam: NetParameter, i: Int): Array[Int] = {
    val numAxes = netParam.input_shape(i).dim_size
    val shape = new Array[Int](numAxes)
    for (j <- 0 to numAxes - 1) {
      shape(j) = netParam.input_shape(i).dim(j).toInt
    }
    shape
  }

  def arraysToFloatBlobVector(inputBuffer: Array[Array[Array[Float]]], inputs: FloatBlobVector, batchSize: Int, inputBufferSize: Array[Int], inputSize: Int) = {
    for (i <- 0 to inputSize - 1) {
      val blob = inputs.get(i)
      val buffer = blob.mutable_cpu_data()
      var batchIndex = 0
      while (batchIndex < batchSize) {
        var j = 0
        while (j < inputBufferSize(i)) {
          // it'd be preferable to do this with one call, but JavaCPP's FloatPointer API has confusing semantics
          buffer.put(inputBufferSize(i) * batchIndex + j, inputBuffer(i)(batchIndex)(j))
          j += 1
        }
        batchIndex += 1
      }
    }
  }

  // this method is just for testing
  def arraysFromFloatBlobVector(inputs: FloatBlobVector, batchSize: Int, inputBufferSize: Array[Int], inputSize: Int): Array[Array[Array[Float]]] = {
    val result = new Array[Array[Array[Float]]](inputSize)
    for (i <- 0 to inputSize - 1) {
      result(i) = new Array[Array[Float]](batchSize)
      val blob = inputs.get(i)
      val buffer = blob.cpu_data()
      for (batchIndex <- 0 to batchSize - 1) {
        result(i)(batchIndex) = new Array[Float](inputBufferSize(i))
        var j = 0
        while (j < inputBufferSize(i)) {
          // it'd be preferable to do this with one call, but JavaCPP's FloatPointer API has confusing semantics
          result(i)(batchIndex)(j) = buffer.get(inputBufferSize(i) * batchIndex + j)
          j += 1
        }
      }
    }
    return result
  }

}
