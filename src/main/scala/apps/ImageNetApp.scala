package apps

import java.io._
import scala.util.Random

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.storage.StorageLevel

import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import org.bytedeco.javacpp.caffe._

import libs._
import loaders._
import preprocessing._

// to run this app, the ImageNet training and validation data must be located on
// S3 at s3://sparknet/ILSVRC2012_img_train/ and s3://sparknet/ILSVRC2012_img_val/.
// Performance is best if the uncompressed data can fit in memory. If it cannot
// fit, you can replace persist() with persist(StorageLevel.MEMORY_AND_DISK).
// However, spilling the RDDs to disk can cause training to be much slower.
object ImageNetApp {
  val trainBatchSize = 256
  val testBatchSize = 50
  val channels = 3
  val fullHeight = 256
  val fullWidth = 256
  val croppedHeight = 227
  val croppedWidth = 227
  val fullImShape = Array(channels, fullHeight, fullWidth)
  val fullImSize = fullImShape.product

  val workerStore = new WorkerStore()

  def main(args: Array[String]) {
    val numWorkers = args(0).toInt
    val s3Bucket = args(1)
    val conf = new SparkConf()
      .setAppName("ImageNet")
      .set("spark.driver.maxResultSize", "30G")
      .set("spark.task.maxFailures", "1")
      .setExecutorEnv("LD_LIBRARY_PATH", sys.env("LD_LIBRARY_PATH"))

    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val sparkNetHome = sys.env("SPARKNET_HOME")
    val logger = new Logger(sparkNetHome + "/training_log_" + System.currentTimeMillis().toString + ".txt")

    val loader = new ImageNetLoader(s3Bucket)
    logger.log("loading train data")
    var trainRDD = loader.apply(sc, "ILSVRC2012_img_train/train.000", "train.txt", fullHeight, fullWidth)
    logger.log("loading test data")
    val testRDD = loader.apply(sc, "ILSVRC2012_img_val/val.00", "val.txt", fullHeight, fullWidth)

    // convert to dataframes
    val schema = StructType(StructField("data", BinaryType, false) :: StructField("label", IntegerType, false) :: Nil)
    var trainDF = sqlContext.createDataFrame(trainRDD.map{ case (a, b) => Row(a, b)}, schema)
    var testDF = sqlContext.createDataFrame(testRDD.map{ case (a, b) => Row(a, b)}, schema)

    val numTrainData = trainDF.count()
    logger.log("numTrainData = " + numTrainData.toString)
    val numTestData = testDF.count()
    logger.log("numTestData = " + numTestData.toString)

    logger.log("computing mean image")
    val meanImage = trainDF.map(row => row(0).asInstanceOf[Array[Byte]].map(e => (e & 0xFF).toLong))
                           .reduce((a, b) => (a, b).zipped.map(_ + _))
                           .map(e => (e.toDouble / numTrainData).toFloat)

    logger.log("coalescing") // if you want to shuffle your data, replace coalesce with repartition
    trainDF = trainDF.coalesce(numWorkers).cache()
    testDF = testDF.coalesce(numWorkers).cache()

    val workers = sc.parallelize(Array.range(0, numWorkers), numWorkers)

    trainDF.foreachPartition(iter => workerStore.put("trainPartitionSize", iter.size))
    testDF.foreachPartition(iter => workerStore.put("testPartitionSize", iter.size))
    logger.log("trainPartitionSizes = " + workers.map(_ => workerStore.get[Int]("trainPartitionSize")).collect().deep.toString)
    logger.log("testPartitionSizes = " + workers.map(_ => workerStore.get[Int]("testPartitionSize")).collect().deep.toString)

    // initialize nets on workers
    workers.foreach(_ => {
      val netParam = new NetParameter()
      ReadProtoFromTextFileOrDie(sparkNetHome + "/models/bvlc_reference_caffenet/train_val.prototxt", netParam)
      val solverParam = new SolverParameter()
      ReadSolverParamsFromTextFileOrDie(sparkNetHome + "/models/bvlc_reference_caffenet/solver.prototxt", solverParam)
      solverParam.clear_net()
      solverParam.set_allocated_net_param(netParam)
      Caffe.set_mode(Caffe.GPU)
      val solver = new CaffeSolver(solverParam, schema, new ImageNetPreprocessor(schema, meanImage, fullHeight, fullWidth, croppedHeight, croppedWidth))
      workerStore.put("netParam", netParam) // prevent netParam from being garbage collected
      workerStore.put("solverParam", solverParam) // prevent solverParam from being garbage collected
      workerStore.put("solver", solver)
    })

    // initialize weights on master
    var netWeights = workers.map(_ => workerStore.get[CaffeSolver]("solver").trainNet.getWeights()).collect()(0)

    var i = 0
    while (true) {
      logger.log("broadcasting weights", i)
      val broadcastWeights = sc.broadcast(netWeights)
      logger.log("setting weights on workers", i)
      workers.foreach(_ => workerStore.get[CaffeSolver]("solver").trainNet.setWeights(broadcastWeights.value))

      if (i % 10 == 0) {
        logger.log("testing", i)
        val testAccuracies = testDF.mapPartitions(
          testIt => {
            val numTestBatches = workerStore.get[Int]("testPartitionSize") / testBatchSize
            var accuracy = 0F
            for (j <- 0 to numTestBatches - 1) {
              val out = workerStore.get[CaffeSolver]("solver").trainNet.forward(testIt, List("accuracy"))
              accuracy += out("accuracy").get(Array())
            }
            Array[(Float, Int)]((accuracy, numTestBatches)).iterator
          }
        ).cache()
        val accuracies = testAccuracies.map{ case (a, b) => a }.sum
        val numTestBatches = testAccuracies.map{ case (a, b) => b }.sum
        val accuracy = accuracies / numTestBatches
        logger.log("%.2f".format(100F * accuracy) + "% accuracy", i)
      }

      logger.log("training", i)
      val syncInterval = 50
      trainDF.foreachPartition(
        trainIt => {
          val len = workerStore.get[Int]("trainPartitionSize")
          val startIdx = Random.nextInt(len - syncInterval * trainBatchSize)
          val it = trainIt.drop(startIdx)
          for (j <- 0 to syncInterval - 1) {
            workerStore.get[CaffeSolver]("solver").step(it)
          }
        }
      )

      logger.log("collecting weights", i)
      netWeights = workers.map(_ => { workerStore.get[CaffeSolver]("solver").trainNet.getWeights() }).reduce((a, b) => CaffeWeightCollection.add(a, b))
      CaffeWeightCollection.scalarDivide(netWeights, 1F * numWorkers)
      logger.log("weight = " + netWeights("conv1")(0).toFlat()(0).toString, i)
      i += 1
    }

    logger.log("finished training")
  }
}
