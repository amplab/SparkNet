package apps

import java.io._
import java.net.InetAddress

import libs._
import com.redis._
import loaders.ImageNetLoader
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.apache.spark.sql.{ DataFrame, Row, SQLContext, SaveMode }
import org.apache.spark.{ SparkConf, SparkContext }
import org.bytedeco.javacpp.caffe._

import scala.collection.mutable
import scala.util.Random

object ImageNetApp {
  val channels = 3
  val fullHeight = 256
  val fullWidth = 256
  val croppedHeight = 227
  val croppedWidth = 227
  val fullImShape = Array(channels, fullHeight, fullWidth)
  val fullImSize = fullImShape.product

  val workerStore = new WorkerStore()
  val conv1Name = "conv1/7x7_s2"

  def main(args: Array[String]) {
    //total workers to be run
    val numWorkers = args(0).toInt
    // where the imagenet data located, can be s3 and hadoop
    val s3Bucket = args(1)
    // synInterval used in controlInput using dataframe
    val syncInterval = args(2).toInt
    //00 means all, 000 means 1/10, 0000 mean 1/100; this is for processing imagenet into parque format. The published version using random skipping in full dataframe
    val zeros = args(3)
    // from where to restart, 0 means from scratch. the snapshot should be locate under SPARKNET_HOME/data directory
    val restartSnapShot = args(4).toInt
    val trainBatchSize = args(5).toInt
    //using dataframe or lmdb.
    //ToDo: make sure dataframe is fast enough to be usable in training, switching to tensorflow may gaurantee the speed
    val controlInput = args(6).toBoolean
    //2 machine the requirement for each machine is around 198G for LMDB, that is bug in caffe of LMDB
    val memeoryLeakRestartInterval = args(7).toInt
    // train seperately to develop intelligence
    val firstOpinion = args(8).toBoolean
    //caffe iteration loop before sync up neuralnet weights
    val loop = args(9).toInt
    //gpuCount for each machine to be used
    val gpuCount = args(10).toInt
    val conf = new SparkConf()
    setUpSparkConf(conf)
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val sparkNetHome = sys.env("SPARKNET_HOME")
    val rediHost = sys.env("Redis")
    val gpuHosts = sys.env("GPU_HOSTS")
    implicit val logger = new Logger(sparkNetHome + "/training_log_" + System.currentTimeMillis().toString + ".txt")

    logger.log("numofWorkers=" + numWorkers)
    logger.log("loading train data")
    var trainDF = sqlContext.read.parquet("/imagenet/parquet/train")

    logger.log("loading test data")
    val testDF: DataFrame = sqlContext.read.parquet("/imagenet/parquet/val").select("data", "label")

    logger.log("computing mean image")
    /*
    val meanImage = trainDF.map(row => row(0).asInstanceOf[Array[Byte]].map(e => (e & 0xFF).toLong))
                           .reduce((a, b) => (a, b).zipped.map(_ + _))
                           .map(e => (e.toDouble / numTrainData).toFloat)
    val out: ObjectOutputStream  = new ObjectOutputStream(new FileOutputStream(fileName))
    out.writeObject(meanImage)
    out.close();
     */
    val fileName = sparkNetHome + "/imagenet.mean"
    val in: ObjectInputStream = new ObjectInputStream(new FileInputStream(fileName))
    val meanImage: Array[Float] = in.readObject().asInstanceOf[Array[Float]]
    logger.log("reading mean ")

    //testDF = testDF.cache()
    var gpuWorkers: RDD[Int] = sc.parallelize(Array.range(0, numWorkers),  numWorkers)

    val schema = StructType(StructField("data", BinaryType, false) :: StructField("label", IntegerType, false) :: Nil)

    // initialize nets on workers
    initResourceForGPUDistribution(rediHost, gpuHosts, gpuCount)
    gpuWorkers.foreach(t => initCaffe(sparkNetHome, restartSnapShot, schema, meanImage, controlInput))

    val count = if (controlInput) trainDF.select("id").count() else 1281167
    logger.log("train image count=" + count)

    if (controlInput) {
      //trainDF = trainDF.persist(StorageLevel.MEMORY_ONLY) //trainDF = trainDF.coalesce(numWorkers).persist(StorageLevel.MEMORY_ONLY)
    }

    var i = restartSnapShot
    while (true) {
      val testPeriod = 100
      //test(testDF, i, syncInterval, testPeriod)
      logger.log("training", i)
      if (i % memeoryLeakRestartInterval == 0 && i != 0 && i != restartSnapShot) {
        //save snapshot
        gpuWorkers.foreach(_ => workerStore.get[CaffeSolver]("solver").save(sparkNetHome + "/data/imnet" + i + "-b.caffemodel"))
      }

      if (controlInput) {
        //train(syncInterval, count, trainBatchSize, trainDF, sc, logger)

        val sfs: Seq[RDD[Row]] = (1 to syncInterval).map { wid =>
          val start = Random.nextInt(count.toInt - trainBatchSize)
          val end = start + trainBatchSize
          val sf = trainDF.filter(" id >= " + start + " and id < " + end + "").select("data", "label")

          logger.log("after sampling starting from ", start)
          sf.coalesce(1).rdd
        }

        val allfs = sc.union(sfs)
        logger.log("union partition length  =  " + allfs.partitions.length)
        allfs.foreachPartition(
          trainIt => {
            val t2 = System.currentTimeMillis()
            workerStore.get[CaffeSolver]("solver").step(trainIt)
            val t3 = System.currentTimeMillis()
            print("iters took " + ((t3 - t2) * 1F / 1000F).toString + " s\n")

          })

      } else {
        gpuWorkers.foreach(_ => workerStore.get[CaffeSolver]("solver").iter(loop))
      }

      var netWeights: mutable.Map[String, mutable.MutableList[Array[Float]]] = null
      if (!firstOpinion) {
        logger.log("collecting weights", i)
        val t3 = System.currentTimeMillis()
        netWeights = gpuWorkers.map(_ => workerStore.get[CaffeSolver]("solver").trainNet.getWeights()).treeReduce((a, b) => CaffeWeightCollection.add(a, b))
        val t4 = System.currentTimeMillis()
        logger.log("collect took " + ((t4 - t3) * 1F / 1000F).toString + " s\n")
        CaffeWeightCollection.scalarDivide(netWeights, numWorkers)
        logger.log("weight = " + netWeights(conv1Name)(0)(0).toString, i)
      }

      //no need to hack anymore
      if (i % 50 == 0) {
        //save frequently
        gpuWorkers.foreach(_ => workerStore.get[CaffeSolver]("solver").save(sparkNetHome + "/data/imnet" + i + ".caffemodel"))
      }

      val hack = true
      if (i % memeoryLeakRestartInterval == 0 && i != 0 && i != restartSnapShot) {
        //save snapshot
        gpuWorkers.foreach(_ => workerStore.get[CaffeSolver]("solver").save(sparkNetHome + "/data/imnet" + i + ".caffemodel"))

        if (hack && !controlInput) {
          //folowing code is hack due to LMDB memory issue with caffe
          gpuWorkers.foreach(t => initCaffeErrorHack(t, sparkNetHome, i, schema))
          //all screwed up , reinitializing, make sure sleep a bit
          //this has problem due to LMDB will start from scratch, you won't be able to get the full imagenet datas ---very bad verify this please
          Thread.sleep(20000)
          gpuWorkers = sc.parallelize(Array.range(0, numWorkers), numWorkers)
          initResourceForGPUDistribution(rediHost, gpuHosts,gpuCount)
          gpuWorkers.foreach(t => initCaffe(sparkNetHome, i, schema, meanImage, controlInput))
        }
      }

      //tricky sequence
      if (!firstOpinion) {
        logger.log("broadcasting weights", i)
        val t1 = System.currentTimeMillis()
        val broadcastWeights = sc.broadcast(netWeights)
        val t2 = System.currentTimeMillis()
        logger.log("broadcast took " + ((t2 - t1) * 1F / 1000F).toString + " s\n")

        logger.log("setting weights on workers", i)
        gpuWorkers.foreach(_ => workerStore.get[CaffeSolver]("solver").trainNet.setWeights(broadcastWeights.value))

        broadcastWeights.unpersist()
        //broadcastWeights.destroy()

        val t10 = System.currentTimeMillis()
        logger.log("setweight took " + ((t10 - t2) * 1F / 1000F).toString + " s\n")
      }

      i += 1
    }

    logger.log("finished training")
  }

  /**
   * Using Redis to init and distribute gpuid evenly to all hosts
   * @param gpuCount
   * @return
   */
  def initResourceForGPUDistribution(rediHost:String, gpuHosts: String, gpuCount: Int) = {
    val r = new RedisClient(rediHost, 6379)
    for (host <- gpuHosts.split(","); i <- Array.range(0, gpuCount)) {
      r.rpush(host, i)
    }
    r.disconnect
  }

  def train(syncInterval: Int, count: Long, sizeOfBatch: Int, trainDF: DataFrame, sc: SparkContext, logger: Logger) = {
    val sfs: Seq[RDD[Row]] = (1 to syncInterval).map { wid =>
      val start = Random.nextInt(count.toInt - sizeOfBatch)
      val end = start + sizeOfBatch
      val sf = trainDF.filter(" id >= " + start + " and id < " + end + "").select("data", "label")

      logger.log("after sampling starting from ", start)
      sf.coalesce(1).rdd
    }

    val allfs = sc.union(sfs)
    logger.log("union partition length  =  " + allfs.partitions.length)
    allfs.foreachPartition(
      trainIt => {
        val t2 = System.currentTimeMillis()
        workerStore.get[CaffeSolver]("solver").step(trainIt)
        val t3 = System.currentTimeMillis()
        print("iters took " + ((t3 - t2) * 1F / 1000F).toString + " s\n")

      })

  }

  //should do same thing as training, quick shot for now
  def test(testDF: DataFrame, epoc: Int, interval: Int, testPeriod: Int)(implicit logger: Logger) = {
    if (epoc % testPeriod == 0 && epoc != 0) {
      logger.log("testing", epoc)
      val repeat = 5 //256 * 10 * 5 still less than 50,000 testing, random sampling also will be okay
      val testAccuracies = testDF.coalesce(interval).mapPartitions(
        testIt => {
          var accuracy = 0F
          var loss = 0F
          for (j <- 1 to repeat) {

            val out = workerStore.get[CaffeSolver]("solver").trainNet.forward(testIt, List("loss3/top-1", "loss3/loss3"))
            accuracy += out("loss3/top-1").get(Array())
            loss = out("loss3/loss3").get(Array())
            println("testing " + j + " acc=" + accuracy + " loss=" + loss)
          }

          Array[(Float, Float)]((accuracy, loss)).iterator
        })

      val accuracies = testAccuracies.map { case (a, b) => a }.sum
      val numTestBatches = repeat * interval
      val accuracy = accuracies / numTestBatches
      logger.log("total testbatches = " + numTestBatches)

      logger.log("%.2f".format(100F * accuracy) + "% accuracy", epoc)
    }
  }

  val solverKey = "solver"

  def initCaffeErrorHack(wid: Int, sparkNetHome: String, snapshot: Int, schema: StructType) = {
    println("wid= " + wid + " initCaffeErrorHack for snapshot=" + snapshot)
    Thread.sleep(4000)
    for (solver <- workerStore.get_[CaffeSolver](solverKey)) {
      workerStore.store -= "solver"
      solver.close()
      throw new LinkageError("fake error due to LMDB issue to blowup executor" + wid) //LinkageError or ThreadDeath
    }
  }

  def getSolver(hostId: Int, gpuId: Int) = {
    val solver = workerStore.get[CaffeSolver]("solver")
    //very hard to recover since weight is not know here
    if (solver == null) {

    }
  }

  /**
   * Allocate GPU base on distributed GPU resources and initialize CaffeSolver
   * @param sparkNetHome
   * @param snapshot
   * @param schema
   * @param meanImage
   * @param controlInput
   */
  def initCaffe(sparkNetHome: String, snapshot: Int, schema: StructType, meanImage: Array[Float], controlInput: Boolean) = {
    println("initCaffe from snapshot=" + snapshot)
    Caffe.set_mode(Caffe.GPU)

    val hostname = InetAddress.getLocalHost.getHostName
    val r = new RedisClient("bdalab12", 6379)
    val gpuId = r.rpop(hostname).get.toInt
    r.disconnect
    println("gpuId=" + gpuId + " setup " + hostname)
    Caffe.SetDevice(gpuId)

    val solverParam = new SolverParameter()
    val solverId = if (gpuId == 0) "" else "1"
    ReadSolverParamsFromTextFileOrDie(sparkNetHome + "/models/bvlc_googlenet/solver" + solverId + ".prototxt", solverParam)

    if (controlInput) {
      val netParam = new NetParameter()
      ReadProtoFromTextFileOrDie(sparkNetHome + "/models/bvlc_googlenet/train_val.prototxt.input", netParam)
      solverParam.clear_net()
      solverParam.set_allocated_net_param(netParam)

      workerStore.put("netParam", netParam) // prevent netParam from being garbage collected
      workerStore.put("solverParam", solverParam) // prevent solverParam from being garbage collected
    }

    val solver = new CaffeSolver(gpuId, solverParam, schema, new ImageNetPreprocessor(schema, meanImage, fullHeight, fullWidth, croppedHeight, croppedWidth), controlInput)
    workerStore.put(solverKey, solver)
    println("put into map then setup gpu")

    if (snapshot > 0) {
      println("restore snapshot" + snapshot)
      solver.trainNet.copyTrainedLayersFrom(sparkNetHome + "/data/imnet" + snapshot + ".caffemodel")
      println("done")
    }
  }

  /**
   * experiment with different spark configuration
   * @param conf
   * @return
   */
  def setUpSparkConf(conf: SparkConf): SparkConf = {
    conf.setAppName("ImageNet")
      .set("spark.driver.maxResultSize", "18G")
      .set("spark.executor.heartbeatInterval", "60000") //default 10s

      .set("spark.task.maxFailures", "12800")
      //next line no effect
      .set("spark.yarn.max.executor.failures", "6400")

      .setExecutorEnv("LD_LIBRARY_PATH", sys.env("LD_LIBRARY_PATH"))
      .set("spark.kryoserializer.buffer", "2560k")
      .set("spark.kryoserializer.buffer.max", "256m")
      //.set("spark.io.compression.codec", "lz4")
      .set("spark.yarn.executor.memoryOverhead", "1024")

      // .set("spark.executor.extraJavaOptions", "-XX:+UseG1GC")
      .set("spark.memory.fraction", "0.75")
      .set("spark.memory.storageFraction", "0.8")
      .set("spark.shuffle.memoryFraction", "0.1")

      .set("spark.shuffle.consolidateFiles", "true")
      .set("spark.shuffle.file.buffer", "40960k")
      .set("spark.akka.frameSize", "100") //100MB default 10
    //.set("spark.shuffle.compress", "false") //default true

    //.set("spark.shuffle.manager", SORT)
  }

  /**
   * Save imagenet as parquet file to allow index predicate pushdown for fast data retrieving
   * @param s3Bucket
   * @param sc
   * @param sqlContext
   * @param zeros
   * @param logger
   */
  def saveAsParquet(s3Bucket: String, sc: SparkContext, sqlContext: SQLContext, zeros: String, logger: Logger) = {
    val loader = new ImageNetLoader(s3Bucket)
    var trainRDD = loader.apply(sc, "ILSVRC2012_img_train/train." + zeros + "*", "train.txt", fullHeight, fullWidth) //.repartition(200)

    //val testRDD = loader.apply(sc, "ILSVRC2012_img_val/val.0*", "val.txt", fullHeight, fullWidth)
    //testRDD.repartition(8)

    logger.log("after load train trainRDD.partition=" + trainRDD.getNumPartitions)

    // convert to dataframes
    val schema = StructType(StructField("data", BinaryType, false) :: StructField("label", IntegerType, false) :: Nil)
    var trainDF = sqlContext.createDataFrame(trainRDD.map { case (a, b) => Row(a, b) }, schema)

    //var trainDF = sqlContext.read.parquet("/imagenet/parquet/train_data")
    // var testDF = sqlContext.createDataFrame(testRDD.map { case (a, b) => Row(a, b) }, schema)

    logger.log("converting to parquet")
    val rows = trainDF.rdd.zipWithUniqueId.map {
      case (r: Row, id: Long) => Row.fromSeq((id % 100).toInt +: id +: r.toSeq)
    }
    val dfWithPK = sqlContext.createDataFrame(
      rows, StructType(StructField("key", IntegerType, false) +: StructField("id", LongType, false) +: schema.fields))

    dfWithPK.write.format("parquet").mode(SaveMode.Append).partitionBy("key").save("/imagenet/parquet/train")
    logger.log("done converting to parquet")

    Thread.sleep(200000)
  }
}
