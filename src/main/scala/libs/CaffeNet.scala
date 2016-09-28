package libs

import java.io._
import java.net.InetAddress
import java.nio.file.{Paths, Files}
import java.util.concurrent.{ArrayBlockingQueue, TimeUnit}

import org.apache.curator.framework.CuratorFrameworkFactory
import org.apache.curator.framework.recipes.locks.{InterProcessLock, InterProcessMutex}
import org.apache.curator.retry.ExponentialBackoffRetry
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import org.bytedeco.javacpp.caffe._

import scala.collection.mutable.Map
import scala.collection.mutable.MutableList
import java.util.Arrays


trait NetInterface {
  def forward(rowIt: Iterator[Row]): Array[Row]

  def forwardBackward(rowIt: Iterator[Row])

  def getWeights(): Map[String, MutableList[Array[Float]]]

  def setWeights(weights: Map[String, MutableList[Array[Float]]])

  def outputSchema(): StructType
}

object CaffeNet {
  def apply(gpuId: Int, netParam: NetParameter, schema: StructType, preprocessor: Preprocessor): CaffeNet = {
    return new CaffeNet(gpuId, netParam, schema, preprocessor, new FloatNet(netParam))
  }
}

class CaffeNet(gpuId: Int, netParam: NetParameter, schema: StructType, preprocessor: Preprocessor, caffeNet: FloatNet, controlInput: Boolean = false) {
  val inputSize = if (controlInput ) netParam.input_size else 2
  val batchSize =  if (controlInput ) netParam.input_shape(0).dim(0).toInt else 128


  private val transformations = new Array[(Any, Array[Float]) => Unit](inputSize)
  private val inputIndices = new Array[Int](inputSize)
  private val columnNames: Seq[String] = schema.map(entry => entry.name)
  private val inputRef = new Array[FloatBlob](inputSize)

  def getNet = caffeNet // TODO: For debugging

  val numOutputs = caffeNet.num_outputs
  val numLayers = caffeNet.layers().size.toInt
  val layerNames = List.range(0, numLayers).map(i => caffeNet.layers.get(i).layer_param.name.getString)
  val numLayerBlobs = List.range(0, numLayers).map(i => caffeNet.layers.get(i).blobs().size.toInt)

  val hostname = InetAddress.getLocalHost.getHostName
  //Todo:: ignore gpuId for now using 0
  val gpuLockPath = "/gpu/" + hostname + "/0" //+ gpuId
  val hosts = "bdalab12:2181"
  val baseSleepTimeMills = 1000
  val maxRetries = 3

  val retryPolicy = new ExponentialBackoffRetry(baseSleepTimeMills, maxRetries)
  val client = CuratorFrameworkFactory.newClient(hosts, retryPolicy)
  client.start()

  //setup transformations
  for (i <- 0 to inputSize - 1
      if controlInput
  ) {
    val name = netParam.input(i).getString
    transformations(i) = preprocessor.convert(name, JavaCPPUtils.getInputShape(netParam, i).drop(1)) // drop first index to ignore batchSize
    inputIndices(i) = columnNames.indexOf(name)
  }

  //speed up queues; due to bug C  [libcaffe.so.1.0.0-rc3+0x41551b]  caffe::SyncedMemory::mutable_cpu_data()+0xb set queuesize to 1
  val queueSize = 1
  //val queues = new ArrayBlockingQueue[FloatBlobVector](queueSize)

  //for (i <- 1 to queueSize) {
  // Preallocate a buffer for data input into the net
  val inputs = new FloatBlobVector(inputSize)
  for (i <- 0 to inputSize - 1
       if controlInput
  ) {
    val dims = new Array[Int](netParam.input_shape(i).dim_size)
    for (j <- dims.indices) {
      dims(j) = netParam.input_shape(i).dim(j).toInt
    }
    // prevent input blobs from being GCed
    // see https://github.com/bytedeco/javacpp-presets/issues/140
    inputRef(i) = new FloatBlob(dims)
    inputs.put(i, inputRef(i))
  }

  //queues.put(inputs)
  //}


  // in `inputBuffer`, the first index indexes the input argument, the second
  // index indexes into the batch, the third index indexes the values in the
  // data
  val inputBuffer = new Array[Array[Array[Float]]](inputSize)
  val inputBufferSize = new Array[Int](inputSize)
  for (i <- 0 to inputSize - 1
       if controlInput
  ) {
    println("inputSize = " + inputSize)
    inputBufferSize(i) = JavaCPPUtils.getInputShape(netParam, i).drop(1).product // drop 1 to ignore batchSize
    inputBuffer(i) = new Array[Array[Float]](batchSize)
    for (batchIndex <- 0 to batchSize - 1) {
      inputBuffer(i)(batchIndex) = new Array[Float](inputBufferSize(i))
    }
  }

  def transformInto(iterator: Iterator[Row]): Unit = { // Option[FloatBlobVector] = {
    //val input = queues.take().asInstanceOf[FloatBlobVector]
    try {
      var batchIndex = 0
      while (iterator.hasNext && batchIndex != batchSize) {

        val row = iterator.next
        println("batchIndex = " + batchIndex)
        for (i <- 0 to inputSize - 1) {
          println("input tr i=" + i)
          transformations(i)(row(inputIndices(i)), inputBuffer(i)(batchIndex))
        }

        batchIndex += 1
      }

      JavaCPPUtils.arraysToFloatBlobVector(inputBuffer, inputs, batchSize, inputBufferSize, inputSize)
     // Some(inputs)
    } catch {
      case e: Exception => {
        println("error in transform")
        println(e.getStackTraceString)
        None
      }
    } finally {
      // queues.put(input)
    }
  }

  def transformInto(iterator: Iterator[Row], inputs: FloatBlobVector) = {
    var batchIndex = 0
    while (iterator.hasNext && batchIndex != batchSize) {
      val row = iterator.next
      for (i <- 0 to inputSize - 1) {
        transformations(i)(row(inputIndices(i)), inputBuffer(i)(batchIndex))
      }
      batchIndex += 1
    }
    JavaCPPUtils.arraysToFloatBlobVector(inputBuffer, inputs, batchSize, inputBufferSize, inputSize)
  }

  val timeout = 1200
  //optimized later
  def forward(rowIt: Iterator[Row], dataBlobNames: List[String] = List[String]()): Map[String, NDArray] = {
    val outputs = Map[String, NDArray]()
    val lock = new InterProcessMutex(client, gpuLockPath)
    transformInto(rowIt)
    //for (input <- inputOpt) {
      if (lock.acquire(timeout, TimeUnit.SECONDS)) {
        try {
          println("forward got lock " + gpuLockPath)

          caffeNet.Forward(inputs)

          for (name <- dataBlobNames) {
            val floatBlob = caffeNet.blob_by_name(name)
            if (floatBlob == null) {
              throw new IllegalArgumentException("The net does not have a layer named " + name + ".\n")
            }
            outputs += (name -> JavaCPPUtils.floatBlobToNDArray(floatBlob))
          }
          return outputs
        } catch {
          case e: Exception => e.printStackTrace()
        } finally {
          // queues.put(input)
          lock.release()
        }
      } else {

        //queues.put(input)



        println("cannot get lock in 200 Seconds " + gpuLockPath)
      }
    //}




    return outputs
  }

  def forwardBackward(rowIt: Iterator[Row]) = {
    //for (input <- inputW) {
      val lock = new InterProcessMutex(client, gpuLockPath)

      if (lock.acquire(timeout, TimeUnit.SECONDS)) {
        try {
          println("got lock " + gpuLockPath)
          val t1 = System.currentTimeMillis()
          transformInto(rowIt, inputs)
          val t2 = System.currentTimeMillis()
          println("transformInto took " + ((t2 - t1) * 1F / 1000F).toString + " s")

          //tricky part
          // caffeNet.ForwardBackward(inputs)
          //caffeNet.ForwardBackward(new FloatBlobVector(inputs))

         // caffeNet.Forward(input)
         // caffeNet.Backward()
          //check if it is needed
          caffeNet.Update()
          //caffeNet.ForwardBackward(input)
          println("after backward")

          val t3 = System.currentTimeMillis()
          print("ForwardBackward took " + ((t3 - t2) * 1F / 1000F).toString + " s\n")
        } catch {
          case e: Exception => e.printStackTrace()
        } finally {
          //queues.put(input)
          lock.release()
        }
      } else {
        println("cannot get lock in 200 Seconds " + gpuLockPath)
        //queues.put(input)
      }

   // }
  }

  def getWeights(): Map[String, MutableList[Array[Float]]] = {
    val weights = Map[String, MutableList[Array[Float]]]()
    val t1 = System.currentTimeMillis()
    for (i <- 0 to numLayers - 1) {
      val weightList = MutableList[Array[Float]]()
      for (j <- 0 to numLayerBlobs(i) - 1) {
        val blob: FloatBlob = caffeNet.layers().get(i).blobs().get(j)

        val shape = JavaCPPUtils.getFloatBlobShape(blob)
        val data = new Array[Float](shape.product)
        blob.cpu_data.get(data, 0, data.length)

        weightList += data //NDArray(data, shape)
      }
      weights += (layerNames(i) -> weightList)
    }
    val t2 = System.currentTimeMillis()
    print("getWeights took " + ((t2 - t1) * 1F / 1000F).toString + " s\n")
    return weights
  }

  def setWeights(weights: Map[String, MutableList[Array[Float]]]) = {
    assert(weights.keys.size == numLayers)
    for (i <- 0 to numLayers - 1) {
      //assert( numLayerBlobs(i) == 2 )
      //only deal with gradient
      // println("blobs========================================"  + numLayerBlobs(i))

      for (j <- 0 to numLayerBlobs(i) - 1) {
        val blob: FloatBlob = caffeNet.layers().get(i).blobs().get(j)
        val shape = JavaCPPUtils.getFloatBlobShape(blob)
        assert(shape.product == weights(layerNames(i))(j).length) // check that weights are the correct shape
        blob.mutable_cpu_data.put(weights(layerNames(i))(j), 0, shape.product)
      }
    }
  }

  //handle memeory issue first try
  def clearWeights(weights: Map[String, MutableList[Array[Float]]]) = {
    assert(weights.keys.size == numLayers)
    /** cause layer not same size issue below; findout issue is LMDB keep on use memory
    for (i <- 0 to numLayers - 1) {
      //assert( numLayerBlobs(i) == 2 )
      //only deal with gradient
      // println("blobs========================================"  + numLayerBlobs(i))
      //weights(layerNames(i)).clear()
      //weights -= layerNames(i)
    }
    */
  }

  def copyTrainedLayersFrom(filepath: String) = {
    if (!Files.exists(Paths.get(filepath))) {
      throw new IllegalArgumentException("The file " + filepath + " does not exist.\n")
    }
    caffeNet.CopyTrainedLayersFrom(filepath)
  }

  def saveWeightsToFile(filepath: String) = {
    val f = new File(filepath)
    f.getParentFile.mkdirs
    val netParam = new NetParameter()
    caffeNet.ToProto(netParam)
    WriteProtoToBinaryFile(netParam, filepath)
  }

  def outputSchema(): StructType = {
    val fields = Array.range(0, numOutputs).map(i => {
      val output = caffeNet.blob_names().get(caffeNet.output_blob_indices().get(i)).getString
      new StructField(new String(output), DataTypes.createArrayType(DataTypes.FloatType), false)
    })
    StructType(fields)
  }
}
