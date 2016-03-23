package apps

import java.io._
import scala.util.Random

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import org.bytedeco.javacpp.caffe._

import libs._
import loaders._
import preprocessing._

object TestApp {
  val trainBatchSize = 5
  val testBatchSize = 5
  val channels = 3
  val height = 32
  val width = 32
  val imShape = Array(channels, height, width)
  val size = imShape.product

  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setAppName("Test")
      .set("spark.driver.maxResultSize", "5G")
      .set("spark.task.maxFailures", "1")
      .setExecutorEnv("LD_LIBRARY_PATH", sys.env("LD_LIBRARY_PATH"))
    // Fetch generic options: they must precede program specific options
    var startIx = 0
    for (arg <- args if arg.startsWith("--")) {
      if (arg.startsWith("--master=")) {
        conf.setMaster(args(0).substring("--master=".length))
        startIx += 1
      } else {
        System.err.println(s"Unknown generic option [$arg]")
      }
    }
    val numWorkers = args(startIx).toInt

    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val sparkNetHome = sys.env("SPARKNET_HOME")
    val logger = new Logger(sparkNetHome + "/training_log_" + System.currentTimeMillis().toString + ".txt")

    logger.log("loading train data")
    val nData = 20
    val nTest = 5
    val trainImages = new Array[Array[Float]](nData)
    val trainLabels = new Array[Int](nData)
    for (i <- 0 to nData - 1) {
      trainImages(i) = new Array[Float](size) // initialized to 0
      trainLabels(i) = Random.nextInt(2)
    }
    val testImages = new Array[Array[Float]](nTest)
    val testLabels = new Array[Int](nTest)
    for (i <- 0 to nTest - 1) {
      testImages(i) = new Array[Float](size) // initialized to 0
      testLabels(i) = Random.nextInt(2)
    }

    val trainData = trainImages.zip(trainLabels).map{ case (a, b) => Row(a, b) }
    val testData = testImages.zip(testLabels).map{ case (a, b) => Row(a, b) }

    // var trainRDD = sc.parallelize(trainImages.zip(trainLabels))
    // logger.log("loading test data")
    //var testRDD = sc.parallelize(testImages.zip(testLabels))

    // convert to dataframes
    val schema = StructType(StructField("data", ArrayType(FloatType), false) :: StructField("label", IntegerType, false) :: Nil)
    // var trainDF = sqlContext.createDataFrame(trainRDD.map{ case (a, b) => Row(a, b)}, schema)
    // var testDF = sqlContext.createDataFrame(testRDD.map{ case (a, b) => Row(a, b)}, schema)

    //logger.log("repartition data")
    // trainDF = trainDF.repartition(numWorkers).cache()
    // testDF = testDF.repartition(numWorkers).cache()

    //val numTrainData = trainDF.count()
    //logger.log("numTrainData = " + numTrainData.toString)

    //val numTestData = testDF.count()
    //logger.log("numTestData = " + numTestData.toString)

    val netParam = new NetParameter()
    ReadProtoFromTextFileOrDie(sparkNetHome + "/models/cifar10/cifar10_quick_train_test.prototxt", netParam)
    val solverParam = new SolverParameter()
    ReadSolverParamsFromTextFileOrDie(sparkNetHome + "/models/cifar10/cifar10_quick_solver.prototxt", solverParam)
    solverParam.clear_net()
    solverParam.set_allocated_net_param(netParam)
    // Caffe.set_mode(Caffe.GPU)
    val solver = new CaffeSolver(solverParam, schema, new DefaultPreprocessor(schema))

    var netWeights = solver.trainNet.getWeights()

    var i = 0
    while (true) {
      logger.log("broadcasting weights", i)
      logger.log("setting weights on workers", i)
      solver.trainNet.setWeights(netWeights)

      logger.log("testing", i)
      var testIt = testData.iterator
      val numTestBatches = nTest / testBatchSize
      var accuracy = 0F
      for (j <- 0 to numTestBatches - 1) {
        val out = solver.trainNet.forward(testIt)
        accuracy += out("accuracy").get(Array())
      }
      accuracy /= numTestBatches
      logger.log("%.2f".format(100F * accuracy) + "% accuracy", i)

      logger.log("training", i)
      var it = trainData.iterator
      val t1 = System.currentTimeMillis()
      val t2 = System.currentTimeMillis()
      print("stuff took " + ((t2 - t1) * 1F / 1000F).toString + " s\n")
      for (j <- 0 to nData / trainBatchSize - 1) {
        solver.step(it)
      }
      val t3 = System.currentTimeMillis()
      print("iters took " + ((t3 - t2) * 1F / 1000F).toString + " s\n")

      logger.log("collecting weights", i)
      netWeights = solver.trainNet.getWeights()
      logger.log("weight = " + netWeights.allWeights("conv1")(0).toFlat()(0).toString, i)
      i += 1
    }

    logger.log("finished training")
  }
}
