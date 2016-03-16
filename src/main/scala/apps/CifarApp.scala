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

// for this app to work, $SPARKNET_HOME should be the SparkNet root directory
// and you need to run $SPARKNET_HOME/data/cifar10/get_cifar10.sh
object CifarApp {
  val trainBatchSize = 100
  val testBatchSize = 100
  val channels = 3
  val height = 32
  val width = 32
  val imShape = Array(channels, height, width)
  val size = imShape.product

  val workerStore = new WorkerStore()

  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setAppName("Cifar")
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

    val loader = new CifarLoader(sparkNetHome + "/data/cifar10/")
    logger.log("loading train data")
    var trainRDD = sc.parallelize(loader.trainImages.zip(loader.trainLabels))
    logger.log("loading test data")
    var testRDD = sc.parallelize(loader.testImages.zip(loader.testLabels))

    // convert to dataframes
    val schema = StructType(StructField("data", ArrayType(FloatType), false) :: StructField("label", IntegerType, false) :: Nil)
    var trainDF = sqlContext.createDataFrame(trainRDD.map{ case (a, b) => Row(a, b)}, schema)
    var testDF = sqlContext.createDataFrame(testRDD.map{ case (a, b) => Row(a, b)}, schema)

    logger.log("repartition data")
    trainDF = trainDF.repartition(numWorkers).cache()
    testDF = testDF.repartition(numWorkers).cache()

    val numTrainData = trainDF.count()
    logger.log("numTrainData = " + numTrainData.toString)

    val numTestData = testDF.count()
    logger.log("numTestData = " + numTestData.toString)

    val workers = sc.parallelize(Array.range(0, numWorkers), numWorkers)

    trainDF.foreachPartition(iter => workerStore.put("trainPartitionSize", iter.size))
    testDF.foreachPartition(iter => workerStore.put("testPartitionSize", iter.size))
    logger.log("trainPartitionSizes = " + workers.map(_ => workerStore.get[Int]("trainPartitionSize")).collect().deep.toString)
    logger.log("testPartitionSizes = " + workers.map(_ => workerStore.get[Int]("testPartitionSize")).collect().deep.toString)

    // initialize nets on workers
    workers.foreach(_ => {
      val netParam = new NetParameter()
      ReadProtoFromTextFileOrDie(sparkNetHome + "/models/cifar10/cifar10_quick_train_test.prototxt", netParam)
      val solverParam = new SolverParameter()
      ReadSolverParamsFromTextFileOrDie(sparkNetHome + "/models/cifar10/cifar10_quick_solver.prototxt", solverParam)
      solverParam.clear_net()
      solverParam.set_allocated_net_param(netParam)

      // Caffe.set_mode(Caffe.GPU)
      val solver = new CaffeSolver(solverParam, schema, new DefaultPreprocessor(schema))
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

      if (i % 5 == 0) {
        logger.log("testing", i)
        val testAccuracies = testDF.mapPartitions(
          testIt => {
            val numTestBatches = workerStore.get[Int]("testPartitionSize") / testBatchSize
            var accuracy = 0F
            for (j <- 0 to numTestBatches - 1) {
              val out = workerStore.get[CaffeSolver]("solver").trainNet.forward(testIt)
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
            workerStore.get[CaffeSolver]("solver").step(it)
          }
          val t3 = System.currentTimeMillis()
          print("iters took " + ((t3 - t2) * 1F / 1000F).toString + " s\n")
        }
      )

      logger.log("collecting weights", i)
      netWeights = workers.map(_ => { workerStore.get[CaffeSolver]("solver").trainNet.getWeights() }).reduce((a, b) => WeightCollection.add(a, b))
      netWeights.scalarDivide(1F * numWorkers)
      logger.log("weight = " + netWeights.allWeights("conv1")(0).toFlat()(0).toString, i)
      i += 1
    }

    logger.log("finished training")
  }
}
