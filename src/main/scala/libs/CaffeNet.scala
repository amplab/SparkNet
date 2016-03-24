package libs

import java.io._
import java.nio.file.{Paths, Files}

import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import org.bytedeco.javacpp.caffe._

import scala.collection.mutable.Map
import scala.collection.mutable.MutableList
import java.util.Arrays

trait NetInterface {
  def forward(rowIt: Iterator[Row]): Array[Row]
  def forwardBackward(rowIt: Iterator[Row])
  def getWeights(): Map[String, MutableList[NDArray]]
  def setWeights(weights: Map[String, MutableList[NDArray]])
  def outputSchema(): StructType
}

object CaffeNet {
  def apply(netParam: NetParameter, schema: StructType, preprocessor: Preprocessor): CaffeNet = {
    return new CaffeNet(netParam, schema, preprocessor, new FloatNet(netParam))
  }
}

class CaffeNet(netParam: NetParameter, schema: StructType, preprocessor: Preprocessor, caffeNet: FloatNet) {
  val inputSize = netParam.input_size
  val batchSize = netParam.input_shape(0).dim(0).toInt
  private val transformations = new Array[(Any, Array[Float]) => Unit](inputSize)
  private val inputIndices = new Array[Int](inputSize)
  private val columnNames = schema.map(entry => entry.name)
  // private val caffeNet = new FloatNet(netParam)
  private val inputRef = new Array[FloatBlob](inputSize)
  def getNet = caffeNet // TODO: For debugging

  val numOutputs = caffeNet.num_outputs
  val numLayers = caffeNet.layers().size.toInt
  val layerNames = List.range(0, numLayers).map(i => caffeNet.layers.get(i).layer_param.name.getString)
  val numLayerBlobs = List.range(0, numLayers).map(i => caffeNet.layers.get(i).blobs().size.toInt)

  for (i <- 0 to inputSize - 1) {
    val name = netParam.input(i).getString
    transformations(i) = preprocessor.convert(name, JavaCPPUtils.getInputShape(netParam, i).drop(1)) // drop first index to ignore batchSize
    inputIndices(i) = columnNames.indexOf(name)
  }

  // Preallocate a buffer for data input into the net
  val inputs = new FloatBlobVector(inputSize)
  for (i <- 0 to inputSize - 1) {
    val dims = new Array[Int](netParam.input_shape(i).dim_size)
    for (j <- dims.indices) {
      dims(j) = netParam.input_shape(i).dim(j).toInt
    }
    // prevent input blobs from being GCed
    // see https://github.com/bytedeco/javacpp-presets/issues/140
    inputRef(i) = new FloatBlob(dims)
    inputs.put(i, inputRef(i))
  }
  // in `inputBuffer`, the first index indexes the input argument, the second
  // index indexes into the batch, the third index indexes the values in the
  // data
  val inputBuffer = new Array[Array[Array[Float]]](inputSize)
  val inputBufferSize = new Array[Int](inputSize)
  for (i <- 0 to inputSize - 1) {
    inputBufferSize(i) = JavaCPPUtils.getInputShape(netParam, i).drop(1).product // drop 1 to ignore batchSize
    inputBuffer(i) = new Array[Array[Float]](batchSize)
    for (batchIndex <- 0 to batchSize - 1) {
      inputBuffer(i)(batchIndex) = new Array[Float](inputBufferSize(i))
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

  def forward(rowIt: Iterator[Row], dataBlobNames: List[String] = List[String]()): Map[String, NDArray] = {
    transformInto(rowIt, inputs)
    caffeNet.Forward(inputs)
    val outputs = Map[String, NDArray]()
    for (name <- dataBlobNames) {
      val floatBlob = caffeNet.blob_by_name(name)
      if (floatBlob == null) {
        throw new IllegalArgumentException("The net does not have a layer named " + name + ".\n")
      }
      outputs += (name -> JavaCPPUtils.floatBlobToNDArray(floatBlob))
    }
    return outputs
  }

  def forwardBackward(rowIt: Iterator[Row]) = {
    print("entering forwardBackward\n")
    val t1 = System.currentTimeMillis()
    transformInto(rowIt, inputs)
    val t2 = System.currentTimeMillis()
    print("transformInto took " + ((t2 - t1) * 1F / 1000F).toString + " s\n")
    caffeNet.ForwardBackward(inputs)
    val t3 = System.currentTimeMillis()
    print("ForwardBackward took " + ((t3 - t2) * 1F / 1000F).toString + " s\n")
  }

  def getWeights(): Map[String, MutableList[NDArray]] = {
    val weights = Map[String, MutableList[NDArray]]()
    for (i <- 0 to numLayers - 1) {
      val weightList = MutableList[NDArray]()
      for (j <- 0 to numLayerBlobs(i) - 1) {
        val blob = caffeNet.layers().get(i).blobs().get(j)
        val shape = JavaCPPUtils.getFloatBlobShape(blob)
        val data = new Array[Float](shape.product)
        blob.cpu_data.get(data, 0, data.length)
        weightList += NDArray(data, shape)
      }
      weights += (layerNames(i) -> weightList)
    }
    return weights
  }

  def setWeights(weights: Map[String, MutableList[NDArray]]) = {
    assert(weights.keys.size == numLayers)
    for (i <- 0 to numLayers - 1) {
      for (j <- 0 to numLayerBlobs(i) - 1) {
        val blob = caffeNet.layers().get(i).blobs().get(j)
        val shape = JavaCPPUtils.getFloatBlobShape(blob)
        assert(shape.deep == weights(layerNames(i))(j).shape.deep) // check that weights are the correct shape
        val flatWeights = weights(layerNames(i))(j).toFlat() // this allocation can be avoided
        blob.mutable_cpu_data.put(flatWeights, 0, flatWeights.length)
      }
    }
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
