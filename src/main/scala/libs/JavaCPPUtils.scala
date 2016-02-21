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


}
