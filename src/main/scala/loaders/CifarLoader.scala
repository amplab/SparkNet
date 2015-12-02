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

  val trainImages = new Array[Array[Byte]](nData)
  val trainLabels = new Array[Int](nData)

  val testImages = new Array[Array[Byte]](batchSize)
  val testLabels = new Array[Int](batchSize)

  val r = new Random()
  // val perm = Vector() ++ r.shuffle(1 to (nData - 1) toIterable)
  val indices = Vector() ++ (0 to nData - 1) toIterable
  val trainPerm = Vector() ++ r.shuffle(indices)
  val testPerm = Vector() ++ ((0 to batchSize) toIterable)

  def getListOfFiles(dir: String) : List[File] = {
    val d = new File(dir)
    if (d.exists && d.isDirectory) {
      d.listFiles.filter(_.getName().split('.').last == "bin").toList
    } else {
      List[File]()
    }
  }

  val fullFileList = getListOfFiles(path)

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

  def readBatch(file: File, batch: Int, images: Array[Array[Byte]], labels: Array[Int], perm: Vector[Int]) {
    val buffer = new Array[Byte](1 + size);
    val inputStream = new FileInputStream(file);

    var i = 0
    var nRead = inputStream.read(buffer)

    while(nRead != -1) {
      assert(i < batchSize)
      labels(perm(batch * batchSize + i)) = (buffer(0) & 0xFF) // convert to unsigned
      images(perm(batch * batchSize + i)) = new Array[Byte](size)
      var j = 0
      while (j < size) {
        // we access buffer(j + 1) because the 0th position holds the label
        images(perm(batch * batchSize + i))(j) = buffer(j + 1)
        j += 1
      }
      nRead = inputStream.read(buffer)
      i += 1
    }
  }
}
