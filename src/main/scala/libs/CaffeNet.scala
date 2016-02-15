package libs

import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import org.bytedeco.javacpp.caffe._

import scala.collection.mutable.Map
import scala.collection.mutable.MutableList
import java.util.Arrays

trait NetInterface {
  def forward(rowIt: Iterator[Row]): Array[Row]
  def forwardBackward(rowIt: Iterator[Row])
  def getWeights(): WeightCollection
  def setWeights(weights: WeightCollection)
  def outputSchema(): StructType
}

object JavaCPPCaffeNet {
  def apply(netParam: NetParameter, schema: StructType, preprocessor: Preprocessor): JavaCPPCaffeNet = {
    return new JavaCPPCaffeNet(netParam, schema, preprocessor, new FloatNet(netParam))
  }
}

class JavaCPPCaffeNet(netParam: NetParameter, schema: StructType, preprocessor: Preprocessor, caffeNet: FloatNet) {
  private val inputSize = netParam.input_size
  private val batchSize = netParam.input_shape(0).dim(0).toInt
  private val transformations = new Array[Any => NDArray](inputSize)
  private val inputIndices = new Array[Int](inputSize)
  private val columnNames = schema.map(entry => entry.name)
  // private val caffeNet = new FloatNet(netParam)
  private val inputRef = new Array[FloatBlob](inputSize)
  def getNet = caffeNet // TODO: For debugging

  private val numOutputs = caffeNet.num_outputs
  private val numLayers = caffeNet.layers().size.toInt
  private val layerNames = List.range(0, numLayers).map(i => caffeNet.layers.get(i).layer_param.name.getString)
  private val numLayerBlobs = List.range(0, numLayers).map(i => caffeNet.layers.get(i).blobs().size.toInt)

  // Caffe.set_mode(Caffe.GPU)

  for (i <- 0 to inputSize - 1) {
    val name = netParam.input(i).getString
    transformations(i) = preprocessor.convert(name, getInputShape(i))
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
  val inputBuffer = new Array[Array[Float]](inputSize)
  val inputBufferSize = new Array[Int](inputSize)
  for (i <- 0 to inputSize - 1) {
    inputBufferSize(i) = getInputShape(i).product
    inputBuffer(i) = new Array[Float](inputBufferSize(i) * batchSize)
  }

  def transformInto(iterator: Iterator[Row], data: FloatBlobVector): Unit = {
    var batchIndex = 0
    while (iterator.hasNext && batchIndex != batchSize) {
      val row = iterator.next
      for (i <- 0 to inputSize - 1) {
        val result = transformations(i)(row(inputIndices(i)))
        val flatArray = result.toFlat() // TODO: Make this efficient
        System.arraycopy(flatArray, 0, inputBuffer(i), batchIndex * inputBufferSize(i), inputBufferSize(i))
      }
      batchIndex += 1
    }
    for (i <- 0 to inputSize - 1) {
      val blob = data.get(i)
      val buffer = blob.cpu_data()
      buffer.put(inputBuffer(i), 0, batchSize * inputBufferSize(i))
    }
  }

  def forward(rowIt: Iterator[Row]): Map[String, NDArray] = {
    Caffe.set_mode(Caffe.GPU)
    transformInto(rowIt, inputs)
    val tops = caffeNet.Forward(inputs)
    val outputs = Map[String, NDArray]()
    for (j <- 0 to numOutputs - 1) {
      val outputName = caffeNet.blob_names().get(caffeNet.output_blob_indices().get(j)).getString
      val top = tops.get(j)
      val shape = Array.range(0, top.num_axes).map(i => top.shape.get(i))
      val output = new Array[Float](shape.product)
      top.cpu_data().get(output, 0, shape.product)
      outputs += (outputName -> NDArray(output, shape))
    }
    return outputs
  }

  def forwardBackward(rowIt: Iterator[Row]) = {
    Caffe.set_mode(Caffe.GPU)
    print("entering forwardBackward\n")
    val t1 = System.currentTimeMillis()
    transformInto(rowIt, inputs)
    val t2 = System.currentTimeMillis()
    print("transformInto took " + ((t2 - t1) * 1F / 1000F).toString + " s\n")
    caffeNet.ForwardBackward(inputs)
    val t3 = System.currentTimeMillis()
    print("ForwardBackward took " + ((t3 - t2) * 1F / 1000F).toString + " s\n")
  }

  def getWeights(): WeightCollection = {
    val weights = Map[String, MutableList[NDArray]]()
    for (i <- 0 to numLayers - 1) {
      val weightList = MutableList[NDArray]()
      for (j <- 0 to numLayerBlobs(i) - 1) {
        val blob = caffeNet.layers().get(i).blobs().get(j)
        val shape = getShape(blob)
        val data = new Array[Float](shape.product)
        blob.cpu_data.get(data, 0, data.length)
        weightList += NDArray(data, shape)
      }
      weights += (layerNames(i) -> weightList)
    }
    return new WeightCollection(weights, layerNames)
  }

  def setWeights(weights: WeightCollection) = {
    assert(weights.numLayers == numLayers)
    for (i <- 0 to numLayers - 1) {
      for (j <- 0 to numLayerBlobs(i) - 1) {
        val blob = caffeNet.layers().get(i).blobs().get(j)
        val shape = getShape(blob)
        assert(shape.deep == weights.allWeights(layerNames(i))(j).shape.deep) // check that weights are the correct shape
        val flatWeights = weights.allWeights(layerNames(i))(j).toFlat() // this allocation can be avoided
        blob.cpu_data.put(flatWeights, 0, flatWeights.length)
      }
    }
  }

  private def getShape(blob: FloatBlob): Array[Int] = {
    val numAxes = blob.num_axes()
    val shape = new Array[Int](numAxes)
    for (k <- 0 to numAxes - 1) {
      shape(k) = blob.shape(k)
    }
    return shape
  }

  private def getInputShape(i: Int): Array[Int] = {
    val numAxes = netParam.input_shape(i).dim_size - 1
    val shape = new Array[Int](numAxes)
    for (j <- 0 to numAxes - 1) {
      shape(j) = netParam.input_shape(i).dim(j + 1).toInt
    }
    return shape
  }

  def outputSchema(): StructType = {
    val fields = Array.range(0, numOutputs).map(i => {
      val output = caffeNet.blob_names().get(caffeNet.output_blob_indices().get(i)).getString
      new StructField(new String(output), DataTypes.createArrayType(DataTypes.FloatType), false)
    })
    StructType(fields)
  }
}
