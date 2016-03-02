import org.scalatest._

import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import org.bytedeco.javacpp.tensorflow._
import scala.collection.mutable._

import libs._
import loaders._

class TensorFlowNetSpec extends FlatSpec {
  val sparkNetHome = sys.env("SPARKNET_HOME")

  "GraphDef" should "be loaded" in {
    val graph = new GraphDef()
    val status = ReadBinaryProto(Env.Default(), sparkNetHome + "/models/tensorflow/mnist/mnist_graph.pb", graph)
    assert(status.ok)
  }

  "TensorFlowNet" should "be created" in {
    val graph = new GraphDef()
    val status = ReadBinaryProto(Env.Default(), sparkNetHome + "/models/tensorflow/mnist/mnist_graph.pb", graph)
    assert(status.ok)
    val schema = StructType(StructField("data", ArrayType(FloatType), false) :: StructField("label", LongType, false) :: Nil)
    new TensorFlowNet(graph, schema, new DefaultTensorFlowPreprocessor(schema))
  }

  "Converting NDArray to Tensor and back" should "preserve its value" in {
    val arraysAndTensors = List((NDArray(Array.range(0, 3 * 4 * 5).map(e => e.toFloat), Array[Int](3, 4, 5)), new Tensor(DT_FLOAT, new TensorShape(3L, 4L, 5L))),
                                (NDArray(Array.range(0, 1).map(e => e.toFloat), Array[Int]()), new Tensor(DT_FLOAT, new TensorShape())),
                                (NDArray(Array.range(0, 1000000).map(e => e.toFloat), Array[Int](1000000)), new Tensor(DT_FLOAT, new TensorShape(1000000L))))
                                // TODO(rkn): Note that if you pass an int into TensorShape(), it may not work. You need to pass in a long (For example, (new TensorShape(10)).dims == 0, but (new TensorShape(10L)).dims == 1).
    arraysAndTensors.foreach {
      case (arrayBefore, t) => {
        TensorFlowUtils.tensorFromNDArray(t, arrayBefore)
        val arrayAfter = TensorFlowUtils.tensorToNDArray(t)
        assert(NDArray.checkEqual(arrayBefore, arrayAfter, 1e-10F))
      }
    }
  }

  "Writing a batch of arrays to Tensor and back" should "preserve their values" in {
    val batchSize = 10
    val dataSize = 37
    val array = Array.range(0, batchSize * dataSize).map(e => e.toFloat)
    val t = new Tensor(DT_FLOAT, new TensorShape(batchSize.toLong, dataSize.toLong))
    for (i <- 0 to batchSize - 1) {
      TensorFlowUtils.tensorFromFlatArray(t, array, i * dataSize, i * dataSize, dataSize)
    }
    val arrayBefore = NDArray(array, Array[Int](batchSize, dataSize))
    val arrayAfter = TensorFlowUtils.tensorToNDArray(t)
    assert(NDArray.checkEqual(arrayBefore, arrayAfter, 1e-10F))
  }

  "TensorFlowNet" should "call forward" in {
    val batchSize = 64
    val graph = new GraphDef()
    val status = ReadBinaryProto(Env.Default(), sparkNetHome + "/models/tensorflow/mnist/mnist_graph.pb", graph)
    assert(status.ok)
    val schema = StructType(StructField("data", ArrayType(FloatType), false) :: StructField("label", LongType, false) :: Nil)
    val net = new TensorFlowNet(graph, schema, new DefaultTensorFlowPreprocessor(schema))
    val inputs = Array.range(0, batchSize).map(_ => Row(Array.range(0, 784).map(e => e.toFloat), 1L))
    val outputs = net.forward(inputs.iterator, List("conv1", "loss", "accuracy"))
    assert(outputs.keys == Set("conv1", "loss", "accuracy"))
    assert(outputs("conv1").shape.deep == Array[Int](5, 5, 1, 32).deep)
    assert(outputs("loss").shape.deep == Array[Int]().deep)
    assert(outputs("accuracy").shape.deep == Array[Int]().deep)
  }

  "Accuracies" should "sum to 1" in {
    // Note that the accuracies in this test will not sum to 1 if the net is stochastic (e.g., if it uses dropout)
    val batchSize = 64
    val graph = new GraphDef()
    val status = ReadBinaryProto(Env.Default(), sparkNetHome + "/models/tensorflow/mnist/mnist_graph.pb", graph)
    assert(status.ok)
    val schema = StructType(StructField("data", ArrayType(FloatType), false) :: StructField("label", LongType, false) :: Nil)
    val net = new TensorFlowNet(graph, schema, new DefaultTensorFlowPreprocessor(schema))
    var accuracies = Array.range(0, 10).map(e => e.toLong).map(i => {
      val inputs = Array.range(0, batchSize).map(_ => Row(Array.range(0, 784).map(e => e.toFloat / 784 - 0.5F), i))
      val outputs = net.forward(inputs.iterator, List("accuracy"))
      outputs("accuracy").toFlat()(0)
    })
    assert((accuracies.sum - 1F).abs <= 1e-6)
  }

  "Setting and getting weights" should "preserve their values" in {
    val batchSize = 64
    val graph = new GraphDef()
    val status = ReadBinaryProto(Env.Default(), sparkNetHome + "/models/tensorflow/mnist/mnist_graph.pb", graph)
    assert(status.ok)
    val schema = StructType(StructField("data", ArrayType(FloatType), false) :: StructField("label", LongType, false) :: Nil)
    val net = new TensorFlowNet(graph, schema, new DefaultTensorFlowPreprocessor(schema))
    val inputs = Array.range(0, batchSize).map(_ => Row(Array.range(0, 784).map(e => e.toFloat), 1L))

    val bVal = NDArray(Array.range(0, 10).map(e => e.toFloat), Array[Int](10))
    val wVal = NDArray(Array.range(0, 784 * 10).map(e => e.toFloat), Array[Int](784, 10))
    val conv1Val = NDArray(Array.range(0, 5 * 5 * 1 * 32).map(e => e.toFloat), Array[Int](5, 5, 1, 32))

    net.setWeights(Map(("conv1", conv1Val)))
    val weightsAfter = net.getWeights()
    assert(NDArray.checkEqual(conv1Val, weightsAfter("conv1"), 1e-10F))
  }

  "Calling forward" should "not change weight values" in {
    val batchSize = 64
    val graph = new GraphDef()
    val status = ReadBinaryProto(Env.Default(), sparkNetHome + "/models/tensorflow/mnist/mnist_graph.pb", graph)
    assert(status.ok)
    val schema = StructType(StructField("data", ArrayType(FloatType), false) :: StructField("label", LongType, false) :: Nil)
    val net = new TensorFlowNet(graph, schema, new DefaultTensorFlowPreprocessor(schema))
    val inputs = Array.range(0, batchSize).map(_ => Row(Array.range(0, 784).map(e => e.toFloat), 1L))
    val weightsBefore = net.getWeights()
    for (i <- 0 to 5 - 1) {
      net.forward(inputs.iterator, List("loss"))
    }
    val weightsAfter = net.getWeights()
    assert(TensorFlowWeightCollection.checkEqual(weightsBefore, weightsAfter, 1e-10F))
  }

  "TensorFlowNet" should "call step" in {
    val batchSize = 64
    val graph = new GraphDef()
    val status = ReadBinaryProto(Env.Default(), sparkNetHome + "/models/tensorflow/mnist/mnist_graph.pb", graph)
    assert(status.ok)
    val schema = StructType(StructField("data", ArrayType(FloatType), false) :: StructField("label", LongType, false) :: Nil)
    val net = new TensorFlowNet(graph, schema, new DefaultTensorFlowPreprocessor(schema))
    val inputs = Array.range(0, batchSize).map(_ => Row(Array.range(0, 784).map(e => e.toFloat), 1L))
    net.step(inputs.iterator)
  }

}
