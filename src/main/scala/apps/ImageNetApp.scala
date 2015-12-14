package apps

import java.io._
import scala.util.Random

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.storage.StorageLevel

import libs._
import loaders._
import preprocessing._

// to run this app, the ImageNet training and test data must be located on S3 at
// s3://sparknet/ILSVRC2012_train/ and s3://sparknet/ILSVRC2012_test/.
// Performance is best if the uncompressed data can fit in memory. If it cannot
// fit, you can replace persist() with persist(StorageLevel.MEMORY_AND_DISK).
// However, spilling the RDDs to disk can cause training to be much slower.
object ImageNetApp {
  val trainBatchSize = 256
  val testBatchSize = 50
  val channels = 3
  val fullWidth = 256
  val fullHeight = 256
  val croppedWidth = 227
  val croppedHeight = 227
  val fullImShape = Array(channels, fullHeight, fullWidth)
  val fullImSize = fullImShape.product

  // initialize nets on workers
  val sparkNetHome = "/root/SparkNet"
  System.load(sparkNetHome + "/build/libccaffe.so")
  var netParameter = ProtoLoader.loadNetPrototxt(sparkNetHome + "/caffe/models/bvlc_reference_caffenet/train_val.prototxt")
  netParameter = ProtoLoader.replaceDataLayers(netParameter, trainBatchSize, testBatchSize, channels, croppedHeight, croppedWidth)
  val solverParameter = ProtoLoader.loadSolverPrototxtWithNet(sparkNetHome + "/caffe/models/bvlc_reference_caffenet/solver.prototxt", netParameter, None)
  val net = CaffeNet(solverParameter)

  def main(args: Array[String]) {
    val numWorkers = args(0).toInt
    val conf = new SparkConf()
      .setAppName("ImageNet")
      .set("spark.driver.maxResultSize", "30G")
      .set("spark.task.maxFailures", "1")
      .set("spark.eventLog.enabled", "true")
    val sc = new SparkContext(conf)

    // information for logging
    val startTime = System.currentTimeMillis()
    val trainingLog = new PrintWriter(new File("training_log_" + startTime.toString + ".txt" ))
    def log(message: String, i: Int = -1) {
      val elapsedTime = 1F * (System.currentTimeMillis() - startTime) / 1000
      if (i == -1) {
        trainingLog.write(elapsedTime.toString + ": "  + message + "\n")
      } else {
        trainingLog.write(elapsedTime.toString + ", i = " + i.toString + ": "+ message + "\n")
      }
      trainingLog.flush()
    }

    var netWeights = net.getWeights()

    val loader = new ImageNetLoader("sparknet")
    log("loading train data")
    var trainRDD = loader.apply(sc, "ILSVRC2012_train/", "train.txt")
    log("loading test data")
    val testRDD = loader.apply(sc, "ILSVRC2012_test/", "test.txt")

    log("processing train data")
    val trainConverter = new ScaleAndConvert(trainBatchSize, fullHeight, fullWidth)
    var trainMinibatchRDD = trainConverter.makeMinibatchRDD(trainRDD).persist()
    val numTrainMinibatches = trainMinibatchRDD.count()
    log("numTrainMinibatches = " + numTrainMinibatches.toString)

    log("processing test data")
    val testConverter = new ScaleAndConvert(testBatchSize, fullHeight, fullWidth)
    var testMinibatchRDD = testConverter.makeMinibatchRDD(testRDD).persist()
    val numTestMinibatches = testMinibatchRDD.count()
    log("numTestMinibatches = " + numTestMinibatches.toString)

    val numTrainData = numTrainMinibatches * trainBatchSize
    val numTestData = numTestMinibatches * testBatchSize

    log("computing mean image")
    val meanImage = ComputeMean(trainMinibatchRDD, fullImShape, numTrainData.toInt)
    val meanImageBuffer = meanImage.getBuffer()
    val broadcastMeanImageBuffer = sc.broadcast(meanImageBuffer)

    log("coalescing") // if you want to shuffle your data, replace coalesce with repartition
    trainMinibatchRDD = trainMinibatchRDD.coalesce(numWorkers)
    testMinibatchRDD = testMinibatchRDD.coalesce(numWorkers)

    val trainPartitionSizes = trainMinibatchRDD.mapPartitions(iter => Array(iter.size).iterator).persist()
    val testPartitionSizes = testMinibatchRDD.mapPartitions(iter => Array(iter.size).iterator).persist()
    log("trainPartitionSizes = " + trainPartitionSizes.collect().deep.toString)
    log("testPartitionSizes = " + testPartitionSizes.collect().deep.toString)

    val workers = sc.parallelize(Array.range(0, numWorkers), numWorkers)

    var i = 0
    while (true) {
      log("broadcasting weights", i)
      val broadcastWeights = sc.broadcast(netWeights)
      log("setting weights on workers", i)
      workers.foreach(_ => net.setWeights(broadcastWeights.value))

      if (i % 10 == 0) {
        log("testing", i)
        val testScores = testPartitionSizes.zipPartitions(testMinibatchRDD) (
          (lenIt, testMinibatchIt) => {
            assert(lenIt.hasNext && testMinibatchIt.hasNext)
            val len = lenIt.next
            assert(!lenIt.hasNext)
            // imageNetTestPreprocessing describes the preprocessing that is
            // done to each image before it is passed to Caffe during testing.
            // We subtract the mean image and take the central 227x227 subimage.
            val meanImageBuff = broadcastMeanImageBuffer.value
            val imageNetTestPreprocessing = (im: ByteImage, buffer: Array[Float]) => {
              val heightOffset = 15
              val widthOffset = 15
              im.cropInto(buffer, Array(heightOffset, widthOffset), Array(heightOffset + croppedHeight, widthOffset + croppedWidth))
              var row = 0
              var col = 0
              while (row < croppedHeight) {
                while (col < croppedWidth) {
                  val index = (row + heightOffset) * fullWidth + (col + widthOffset)
                  buffer(index) -= meanImageBuff(index)
                  col += 1
                }
                row += 1
              }
            }
            val minibatchSampler = new MinibatchSampler(testMinibatchIt, len, len)
            net.setTestData(minibatchSampler, len, Some(imageNetTestPreprocessing))
            Array(net.test()).iterator // do testing
          }
        ).cache() // the function inside has side effects, so we need the cache to ensure we don't redo it
        // add up test accuracies (a and b are arrays in case there are multiple test layers)
        val testScoresAggregate = testScores.reduce((a, b) => (a, b).zipped.map(_ + _))
        val accuracies = testScoresAggregate.map(v => 100F * v / numTestMinibatches)
        log("%.2f".format(accuracies(0)) + "% accuracy", i)
      }

      log("training", i)
      val syncInterval = 50
      trainPartitionSizes.zipPartitions(trainMinibatchRDD) (
        (lenIt, trainMinibatchIt) => {
          assert(lenIt.hasNext && trainMinibatchIt.hasNext)
          val len = lenIt.next
          assert(!lenIt.hasNext)
          // imageNetTrainPreprocessing describes the preprocessing that is done
          // to each image before it is passed to Caffe during training. We
          // subtract the mean image and take a random 227x227 subimage.
          val trainPreprocessingBuffer = new Array[Float](fullImSize)
          val meanImageBuff = broadcastMeanImageBuffer.value
          val imageNetTrainPreprocessing = (im: ByteImage, buffer: Array[Float]) => {
            val heightOffset = Random.nextInt(fullHeight - croppedHeight)
            val widthOffset = Random.nextInt(fullWidth - croppedWidth)
            im.cropInto(buffer, Array(heightOffset, widthOffset), Array(heightOffset + croppedHeight, widthOffset + croppedWidth))
            var row = 0
            var col = 0
            while (row < croppedHeight) {
              while (col < croppedWidth) {
                val index = (row + heightOffset) * fullWidth + (col + widthOffset)
                buffer(index) -= meanImageBuff(index)
                col += 1
              }
              row += 1
            }
          }
          val minibatchSampler = new MinibatchSampler(trainMinibatchIt, len, syncInterval)
          net.setTrainData(minibatchSampler, Some(imageNetTrainPreprocessing))
          net.train(syncInterval) // train for syncInterval minibatches
          Array(0).iterator // give the closure the right signature
        }
      ).foreachPartition(_ => ())

      log("collecting weights", i)
      netWeights = workers.map(_ => net.getWeights()).treeReduce((a, b) => WeightCollection.add(a, b))
      netWeights.scalarDivide(1F * numWorkers)

      i += 1
    }

    log("finished training")
  }
}
