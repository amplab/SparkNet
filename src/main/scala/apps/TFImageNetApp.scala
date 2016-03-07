package apps

import java.io._
import scala.util.Random

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.storage.StorageLevel

import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import org.bytedeco.javacpp.tensorflow._

import libs._
import loaders._
import preprocessing._

object TFImageNetApp {
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
      .setAppName("TFImageNet")
      .set("spark.driver.maxResultSize", "30G")
      .set("spark.task.maxFailures", "1")
      .setExecutorEnv("LD_LIBRARY_PATH", sys.env("LD_LIBRARY_PATH"))

    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val sparkNetHome = sys.env("SPARKNET_HOME")
    val logger = new Logger(sparkNetHome + "/training_log_" + System.currentTimeMillis().toString + ".txt")

    val loader = new ImageNetLoader(s3Bucket)
    logger.log("loading train data")
    var trainRDD = loader.apply(sc, "ILSVRC2012_img_train/train.0000", "train.txt", fullHeight, fullWidth)
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
    trainDF = trainDF.coalesce(numWorkers)
    testDF = testDF.coalesce(numWorkers)

    val workers = sc.parallelize(Array.range(0, numWorkers), numWorkers)

    trainDF.foreachPartition(iter => workerStore.put("trainPartitionSize", iter.size))
    testDF.foreachPartition(iter => workerStore.put("testPartitionSize", iter.size))
    logger.log("trainPartitionSizes = " + workers.map(_ => workerStore.get[Int]("trainPartitionSize")).collect().deep.toString)
    logger.log("testPartitionSizes = " + workers.map(_ => workerStore.get[Int]("testPartitionSize")).collect().deep.toString)

    // initialize nets on workers
    workers.foreach(_ => {
      val graph = new GraphDef()
      val status = ReadBinaryProto(Env.Default(), sparkNetHome + "/models/tensorflow/alexnet/alexnet_graph.pb", graph)
      if (!status.ok) {
        throw new Exception("Failed to read " + sparkNetHome + "/models/tensorflow/alexnet/alexnet_graph.pb, try running `python alexnet_graph.py from that directory`")
      }
      val net = new TensorFlowNet(graph, schema, new ImageNetTensorFlowPreprocessor(schema, meanImage, fullHeight, fullWidth, croppedHeight, croppedWidth))
      workerStore.put("graph", graph) // prevent graph from being garbage collected
      workerStore.put("net", net) // prevent net from being garbage collected
    })

    // initialize weights on master
    var netWeights = workers.map(_ => workerStore.get[TensorFlowNet]("net").getWeights()).collect()(0)

    var i = 0
    while (true) {
      logger.log("broadcasting weights", i)
      val broadcastWeights = sc.broadcast(netWeights)
      logger.log("setting weights on workers", i)
      workers.foreach(_ => workerStore.get[TensorFlowNet]("net").setWeights(broadcastWeights.value))

      if (i % 5 == 0) {
        logger.log("testing", i)
        val testAccuracies = testDF.mapPartitions(
          testIt => {
            val numTestBatches = workerStore.get[Int]("testPartitionSize") / testBatchSize
            var accuracy = 0F
            for (j <- 0 to numTestBatches - 1) {
              val out = workerStore.get[TensorFlowNet]("net").forward(testIt, List("accuracy"))
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
      val syncInterval = 10
      trainDF.foreachPartition(
        trainIt => {
          val t1 = System.currentTimeMillis()
          val len = workerStore.get[Int]("trainPartitionSize")
          val startIdx = Random.nextInt(len - syncInterval * trainBatchSize)
          val it = trainIt.drop(startIdx)
          val t2 = System.currentTimeMillis()
          print("stuff took " + ((t2 - t1) * 1F / 1000F).toString + " s\n")
          for (j <- 0 to syncInterval - 1) {
            workerStore.get[TensorFlowNet]("net").step(it)
          }
          val t3 = System.currentTimeMillis()
          print("iters took " + ((t3 - t2) * 1F / 1000F).toString + " s\n")
        }
      )

      logger.log("collecting weights", i)
      netWeights = workers.map(_ => { workerStore.get[TensorFlowNet]("net").getWeights() }).reduce((a, b) => TensorFlowWeightCollection.add(a, b))
      TensorFlowWeightCollection.scalarDivide(netWeights, 1F * numWorkers)
      logger.log("weight = " + netWeights("fc6W").toFlat()(0).toString, i)
      i += 1
    }

    logger.log("finished training")
  }
}
