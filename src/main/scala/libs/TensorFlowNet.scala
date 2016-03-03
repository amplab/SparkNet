package libs

import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import org.bytedeco.javacpp.tensorflow._
import scala.collection.mutable._
import java.nio.FloatBuffer

class TensorFlowNet(graph: GraphDef, schema: StructType, preprocessor: TensorFlowPreprocessor) {
  val options = new SessionOptions()
  val configProto = new ConfigProto()
  configProto.set_log_device_placement(true)
  configProto.set_allow_soft_placement(true)
  options.config(configProto)
  val session = new Session(options)
  val status1 = session.Create(graph)
  TensorFlowUtils.checkStatus(status1)
  val status2 = session.Run(new StringTensorPairVector(), new StringVector(), new StringVector("init//all_vars"), new TensorVector())
  TensorFlowUtils.checkStatus(status2)

  val nodeNames = Array.range(0, graph.node_size).map(i => graph.node(i).name.getString)

  // get input indices, names, and shapes
  val feedIndices = Array.range(0, graph.node_size).filter(i => graph.node(i).op.getString == "Placeholder" && !nodeNames(i).contains("//update_placeholder"))
  val inputSize = feedIndices.length
  val inputNames = feedIndices.map(i => nodeNames(i))
  val columnNames = schema.map(entry => entry.name)
  if (columnNames.toSet != inputNames.toSet) {
  // if (!(columnNames.toSet subsetOf inputNames.toSet)) {
    throw new Exception("The names in `schema` are not the same as the names in `graph`. `graph` has names " + inputNames.deep.toString + ", and `schema` has names " + columnNames.toString + "\n")
  }
  val inputShapes = feedIndices.map(i => TensorFlowUtils.getNodeShape(graph.node(i)))
  val inputTypes = feedIndices.map(i => TensorFlowUtils.getNodeType(graph.node(i)))
  val inputs = (inputShapes, inputTypes).zipped.map{ case (shape, dtype) => new Tensor(dtype, new TensorShape(shape.map(e => e.toLong):_*)) }
  val inputSizes = inputShapes.map(shape => shape.drop(1).product) // drop first index to ignore batchSize
  val batchSize = inputShapes(0)(0)

  val weightIndices = Array.range(0, graph.node_size).filter(i => graph.node(i).op.getString == "Variable")
  val weightNames = weightIndices.map(i => nodeNames(i))
  val weightShapes = weightIndices.map(i => TensorFlowUtils.getNodeShape(graph.node(i)))
  val weightTypes = weightIndices.map(i => TensorFlowUtils.getNodeType(graph.node(i)))

  val updateIndices = Array.range(0, graph.node_size).filter(i => graph.node(i).op.getString == "Placeholder" && nodeNames(i).contains("//update_placeholder"))
  val updateSize = updateIndices.length
  val updateNames = updateIndices.map(i => nodeNames(i))
  val updateShapes = updateIndices.map(i => TensorFlowUtils.getNodeShape(graph.node(i)))
  val updateInputs = updateShapes.map(shape => new Tensor(DT_FLOAT, new TensorShape(shape.map(e => e.toLong):_*)))

  val stepIndex = Array.range(0, graph.node_size).filter(i => nodeNames(i) == ("train//step"))(0)

  val transformations = new Array[(Any, Any) => Unit](inputSize)
  val inputIndices = new Array[Int](inputSize)
  for (i <- 0 to inputSize - 1) {
    val name = inputNames(i)
    transformations(i) = preprocessor.convert(name, inputShapes(i).drop(1)) // drop first index to ignore batchSize
    inputIndices(i) = columnNames.indexOf(name)
  }

  val inputBuffers = Array.range(0, inputSize).map(i => Array.range(0, batchSize).map(_ => TensorFlowUtils.newBuffer(inputTypes(i), inputSizes(i))))

  def loadFrom(iterator: Iterator[Row]) = {
    var batchIndex = 0
    while (iterator.hasNext && batchIndex != batchSize) {
      val row = iterator.next
      for (i <- 0 to inputSize - 1) {
        transformations(i)(row(inputIndices(i)), inputBuffers(i)(batchIndex))
        TensorFlowUtils.tensorFromFlatArray(inputs(i), inputBuffers(i)(batchIndex), batchIndex * inputSizes(i))
      }
      batchIndex += 1
    }
  }

  def forward(rowIt: Iterator[Row], dataTensorNames: List[String] = List[String]()): Map[String, NDArray] = {
    val outputs = new TensorVector()
    val outputNames = dataTensorNames.map(name => name + ":0")
    loadFrom(rowIt)
    val s = session.Run(new StringTensorPairVector(inputNames, inputs), new StringVector(outputNames:_*), new StringVector(), outputs)
    TensorFlowUtils.checkStatus(s)
    val result = Map[String, NDArray]()
    for (i <- 0 to dataTensorNames.length - 1) {
      result += (dataTensorNames(i) -> TensorFlowUtils.tensorToNDArray(outputs.get(i)))
    }
    result
  }

  def step(rowIt: Iterator[Row]) = {
    loadFrom(rowIt)
    val s = session.Run(new StringTensorPairVector(inputNames, inputs), new StringVector(), new StringVector("train//step"), new TensorVector())
    TensorFlowUtils.checkStatus(s)
  }

  // def forwardBackward(rowIt: Iterator[Row]) = {
  // }

  def getWeights(): Map[String, NDArray] = {
    val outputs = new TensorVector()
    val s = session.Run(new StringTensorPairVector(inputNames, inputs), new StringVector(weightNames.map(name => name + ":0"):_*), new StringVector(), outputs)
    TensorFlowUtils.checkStatus(s)
    val weights = Map[String, NDArray]()
    for (i <- 0 to weightNames.length - 1) {
      if (weightTypes(i) == DT_FLOAT) {
        weights += (weightNames(i) -> TensorFlowUtils.tensorToNDArray(outputs.get(i)))
      } else {
        print("Not returning weight for variable " + weightNames(i) + " because it does not have type float.\n")
      }
    }
    weights
  }

  def setWeights(weights: Map[String, NDArray]) = {
    // TODO(rkn): check that weights.keys are all valid
    for (name <- weights.keys) {
      val i = updateNames.indexOf(name + "//update_placeholder")
      TensorFlowUtils.tensorFromNDArray(updateInputs(i), weights(name))
    }
    val updatePlaceholderNames = weights.map{ case (name, array) => name + "//update_placeholder" }.toArray
    val updateAssignNames = weights.map{ case (name, array) => name + "//assign" }.toArray
    val updatePlaceholderVals = weights.map{ case (name, array) => updateInputs(updateNames.indexOf(name + "//update_placeholder")) }.toArray
    val s = session.Run(new StringTensorPairVector(updatePlaceholderNames, updatePlaceholderVals), new StringVector(), new StringVector(updateAssignNames:_*), new TensorVector())
    TensorFlowUtils.checkStatus(s)
  }
}
