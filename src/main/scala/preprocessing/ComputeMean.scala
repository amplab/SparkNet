package preprocessing

import org.apache.spark.rdd.RDD

import libs._

object ComputeMean {
  def apply(minibatchRDD: RDD[(Array[ByteImage], Array[Int])], shape: Array[Int], numData: Int): NDArray = {
    val size = shape.product
    val imageSum = minibatchRDD.mapPartitions(minibatchIt => {
      val runningImageSum = new Array[Long](size)
      while (minibatchIt.hasNext) {
        val currentImageMinibatch = minibatchIt.next._1
        val batchSize = currentImageMinibatch.length
        var i = 0
        while (i < batchSize) {
          val currentImage = currentImageMinibatch(i)
          var j = 0
          val imSize = currentImage.height * currentImage.width
          assert(shape.length == 3 && shape(0) == 3 && shape(1) == currentImage.height && shape(2) == currentImage.width)
          while (j < imSize) {
            // scala Bytes are signed, so we and with 0xFF to get the unsigned
            // value (as an Int) and then add to runningImageSum, which casts it
            // to a Long
            runningImageSum(0 * imSize + j) += (currentImage.red(j) & 0xFF)
            runningImageSum(1 * imSize + j) += (currentImage.green(j) & 0xFF)
            runningImageSum(2 * imSize + j) += (currentImage.blue(j) & 0xFF)
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
