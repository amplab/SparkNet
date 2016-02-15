package preprocessing

import org.apache.spark.rdd.RDD

import libs._

object ComputeMean {
  def computeMean(dataRDD: RDD[(ByteImage, Int)], shape: Array[Int], numData: Int): NDArray = {
    val size = shape.product
    val imageSum = dataRDD.mapPartitions(dataIt => {
      val runningImageSum = new Array[Long](size)
      while (dataIt.hasNext) {
        val currentImage = dataIt.next._1
        var j = 0
        val height = currentImage.getHeight()
        val width = currentImage.getWidth()
        assert(shape.length == 3 && shape(0) == 3 && shape(1) == height && shape(2) == width)
        var row = 0
        while (row < height) {
          var col = 0
          while (col < width) {
            runningImageSum(0 * height * width + row * width + col) += (currentImage.getRed(row, col) & 0xFF)
            runningImageSum(1 * height * width + row * width + col) += (currentImage.getGreen(row, col) & 0xFF)
            runningImageSum(2 * height * width + row * width + col) += (currentImage.getBlue(row, col) & 0xFF)
            col += 1
          }
          row += 1
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

  def computeMeanFromMinibatches(minibatchRDD: RDD[(Array[ByteImage], Array[Int])], shape: Array[Int], numData: Int): NDArray = {
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
          val height = currentImage.getHeight()
          val width = currentImage.getWidth()
          assert(shape.length == 3 && shape(0) == 3 && shape(1) == height && shape(2) == width)
          var row = 0
          while (row < height) {
            var col = 0
            while (col < width) {
              runningImageSum(0 * height * width + row * width + col) += (currentImage.getRed(row, col) & 0xFF)
              runningImageSum(1 * height * width + row * width + col) += (currentImage.getGreen(row, col) & 0xFF)
              runningImageSum(2 * height * width + row * width + col) += (currentImage.getBlue(row, col) & 0xFF)
              col += 1
            }
            row += 1
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

  def writeMeanToBinaryProto(caffeLib: CaffeLibrary, meanImage: NDArray, filename: String) = {
    val state = caffeLib.create_state()
    val shape = meanImage.shape
    val channels = shape(0)
    val height = shape(1)
    val width = shape(2)
    caffeLib.save_mean_image(state, meanImage.toFlat(), channels, height, width, filename, filename.length)
  }
}
