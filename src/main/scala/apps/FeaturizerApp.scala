package apps

import java.io._

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import org.bytedeco.javacpp.caffe._

import scala.collection.mutable.Map

import libs._
import loaders._
import preprocessing._

// For this app to work, $SPARKNET_HOME should be the SparkNet root directory
// and you need to run $SPARKNET_HOME/data/cifar10/get_cifar10.sh. This app
// shows how to use an already trained network to featurize some images.
object FeaturizerApp {
  val batchSize = 100

  val workerStore = new WorkerStore()

  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setAppName("Featurizer")
      .set("spark.driver.maxResultSize", "5G")
      .set("spark.task.maxFailures", "1")
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
    logger.log("loading data")
    var trainRDD = sc.parallelize(loader.trainImages.zip(loader.trainLabels))

    // convert to dataframes
    val schema = StructType(StructField("data", ArrayType(FloatType), false) :: StructField("label", IntegerType, false) :: Nil)
    var trainDF = sqlContext.createDataFrame(trainRDD.map{ case (a, b) => Row(a, b)}, schema)

    logger.log("repartition data")
    trainDF = trainDF.repartition(numWorkers).cache()

    val workers = sc.parallelize(Array.range(0, numWorkers), numWorkers)

    trainDF.foreachPartition(iter => workerStore.put("trainPartitionSize", iter.size))

    // initialize nets on workers
    workers.foreach(_ => {
      val netParam = new NetParameter()
      ReadProtoFromTextFileOrDie(sparkNetHome + "/models/cifar10/cifar10_quick_train_test.prototxt", netParam)
      val net = CaffeNet(netParam, schema, new DefaultPreprocessor(schema))

      Caffe.set_mode(Caffe.GPU)
      workerStore.put("netParam", netParam) // prevent netParam from being garbage collected
      workerStore.put("net", net) // prevent net from being garbage collected
    })

    // initialize weights on master
    var netWeights = workers.map(_ => workerStore.get[CaffeNet]("net").getWeights()).collect()(0) // alternatively, load weights from a .caffemodel file
    logger.log("broadcasting weights")
    val broadcastWeights = sc.broadcast(netWeights)
    logger.log("setting weights on workers")
    workers.foreach(_ => workerStore.get[CaffeNet]("net").setWeights(broadcastWeights.value))

    // featurize the images
    val featurizedDF = trainDF.mapPartitions( it => {
      val trainPartitionSize = workerStore.get[Int]("trainPartitionSize")
      val numTrainBatches = trainPartitionSize / batchSize
      val featurizedData = new Array[Array[Float]](trainPartitionSize)
      val input = new Array[Row](batchSize)
      var i = 0
      var out = None: Option[Map[String, NDArray]]
      while (i < trainPartitionSize) {
        if (i % batchSize == 0) {
          it.copyToArray(input, 0, batchSize)
          out = Some(workerStore.get[CaffeNet]("net").forward(input.iterator, List("ip1")))
        }
        featurizedData(i) = out.get("ip1").slice(0, i % batchSize).toFlat()
        i += 1
      }
      featurizedData.iterator
    })

    logger.log("featurized " + featurizedDF.count().toString + " images")
  }
}
