package preprocessing

import org.apache.spark.rdd.RDD

import libs.NDArray
import libs.ByteNDArray

object ComputeMean {
  def apply(minibatchRDD: RDD[(Array[ByteNDArray], Array[Int])], shape: Array[Int], numData: Int): NDArray = {
    val size = shape.product
    val imageSum = minibatchRDD.mapPartitions(minibatchIt => {
      val runningImageSum = new Array[Long](size)
      while (minibatchIt.hasNext) {
        val currentImageMinibatch = minibatchIt.next._1
        val batchSize = currentImageMinibatch.length
        var i = 0
        while (i < batchSize) {
          val currentImageBuffer = currentImageMinibatch(i).getBuffer
          assert(currentImageBuffer.length == size)
          var j = 0
          while (j < size) {
            // scala Bytes are signed, so we and with 0xFF to get the unsigned
            // value (as an Int) and then add to runningImageSum, which casts it
            // to a Long
            runningImageSum(j) += (currentImageBuffer(j) & 0xFF)
            j += 1
          }
          i += 1
        }
      }
      Array(runningImageSum).iterator
    }).reduce((a, b) => (a, b).zipped.map(_ + _)).map(x => x.toFloat)
    var j = 0
    while (j < size) {
      imageSum(j) /= numData
      j += 1
    }
    return NDArray(imageSum, shape)
  }
}
