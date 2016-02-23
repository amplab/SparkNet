package loaders

import java.io.File
import java.io.FileInputStream

import scala.util.Random

import libs._

/**
 * Loads images from the CIFAR-10 Dataset. The string path points to a directory where the files data_batch_1.bin, etc. are stored.
 *
 * TODO: Implement loading of test images, and distinguish between training and test data
 */
class CifarLoader(path: String) {
  // We hardcode this because these are properties of the CIFAR-10 dataset.
  val height = 32
  val width = 32
  val channels = 3
  val size = channels * height * width
  val batchSize = 10000
  val nBatches = 5
  val nData = nBatches * batchSize

  val trainImages = new Array[Array[Float]](nData)
  val trainLabels = new Array[Int](nData)

  val testImages = new Array[Array[Float]](batchSize)
  val testLabels = new Array[Int](batchSize)

  val r = new Random()
  // val perm = Vector() ++ r.shuffle(1 to (nData - 1) toIterable)
  val indices = Vector() ++ (0 to nData - 1) toIterable
  val trainPerm = Vector() ++ r.shuffle(indices)
  val testPerm = Vector() ++ ((0 to batchSize) toIterable)

  val d = new File(path)
  if (!d.exists) {
    throw new Exception("The path " + path + " does not exist.")
  }
  if (!d.isDirectory) {
    throw new Exception("The path " + path + " is not a directory.")
  }
  val cifar10Files = List("data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin", "test_batch.bin")
  for (filename <- cifar10Files) {
    if (!d.list.contains(filename)) {
      throw new Exception("The directory " + path + " does not contain all of the Cifar10 data. Please run `bash $SPARKNET_HOME/data/cifar10/get_cifar10.sh` to obtain the Cifar10 data.")
    }
  }

  val fullFileList = d.listFiles.filter(_.getName().split('.').last == "bin").toList
  val testFile = fullFileList.find(x => x.getName().split('/').last == "test_batch.bin").head
  val fileList = fullFileList diff List(testFile)

  for (i <- 0 to nBatches - 1) {
    readBatch(fileList(i), i, trainImages, trainLabels, trainPerm)
  }
  readBatch(testFile, 0, testImages, testLabels, testPerm)

  val meanImage = new Array[Float](size)

  for (i <- 0 to nData - 1) {
    for (j <- 0 to size - 1) {
      meanImage(j) += trainImages(i)(j).toFloat / nData
    }
  }

  def readBatch(file: File, batch: Int, images: Array[Array[Float]], labels: Array[Int], perm: Vector[Int]) {
    val buffer = new Array[Byte](1 + size)
    val inputStream = new FileInputStream(file)

    var i = 0
    var nRead = inputStream.read(buffer)

    while(nRead != -1) {
      assert(i < batchSize)
      labels(perm(batch * batchSize + i)) = (buffer(0) & 0xFF) // convert to unsigned
      images(perm(batch * batchSize + i)) = new Array[Float](size)
      var j = 0
      while (j < size) {
        // we access buffer(j + 1) because the 0th position holds the label
        images(perm(batch * batchSize + i))(j) = buffer(j + 1) & 0xFF
        j += 1
      }
      nRead = inputStream.read(buffer)
      i += 1
    }
  }
}
