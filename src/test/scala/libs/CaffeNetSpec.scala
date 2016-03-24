import org.scalatest._

import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import org.bytedeco.javacpp.caffe._

import scala.util.Random

import libs._

@Ignore
class CaffeNetSpec extends FlatSpec {
  val sparkNetHome = sys.env("SPARKNET_HOME")

  "NetParam" should "be loaded" in {
    val netParam = new NetParameter()
    ReadProtoFromTextFileOrDie(sparkNetHome + "/models/adult/adult.prototxt", netParam)
  }


  "CaffeNet" should "be created" in {
    val netParam = new NetParameter()
    ReadProtoFromTextFileOrDie(sparkNetHome + "/models/adult/adult.prototxt", netParam)
    val schema = StructType(StructField("C0", FloatType, false) :: Nil)
    Caffe.set_mode(Caffe.GPU)
    val net = CaffeNet(netParam, schema, new DefaultPreprocessor(schema))
  }

  "CaffeNet" should "call forward" in {
    val netParam = new NetParameter()
    ReadProtoFromTextFileOrDie(sparkNetHome + "/models/adult/adult.prototxt", netParam)
    val schema = StructType(StructField("C0", FloatType, false) :: Nil)
    Caffe.set_mode(Caffe.GPU)
    val net = CaffeNet(netParam, schema, new DefaultPreprocessor(schema))
    val inputs = List[Row](Row(0F), Row(1F))
    val outputs = net.forward(inputs.iterator, List("prob"))
    val keys = outputs.keys.toArray
    assert(keys.length == 1)
    assert(keys(0) == "prob")
    assert(outputs("prob").shape.deep == Array[Int](64, 10).deep) // these numbers are taken from adult.prototxt
  }

  "CaffeNet" should "call forwardBackward" in {
    val netParam = new NetParameter()
    ReadProtoFromTextFileOrDie(sparkNetHome + "/models/adult/adult.prototxt", netParam)
    val schema = StructType(StructField("C0", FloatType, false) :: Nil)
    Caffe.set_mode(Caffe.GPU)
    val net = CaffeNet(netParam, schema, new DefaultPreprocessor(schema))
    val inputs = List.range(0, 100).map(x => Row(x.toFloat))
    net.forwardBackward(inputs.iterator)
  }

  "Calling forward" should "leave weights unchanged" in {
    val netParam = new NetParameter()
    ReadProtoFromTextFileOrDie(sparkNetHome + "/models/adult/adult.prototxt", netParam)
    val schema = StructType(StructField("C0", FloatType, false) :: Nil)
    Caffe.set_mode(Caffe.GPU)
    val net = CaffeNet(netParam, schema, new DefaultPreprocessor(schema))
    val inputs = List[Row](Row(0F), Row(1F))
    val weightsBefore = net.getWeights()
    val outputs = net.forward(inputs.iterator)
    val weightsAfter = net.getWeights()
    assert(CaffeWeightCollection.checkEqual(weightsBefore, weightsAfter, 1e-10F)) // weights should be equal
  }

  "Calling forwardBackward" should "leave weights unchanged" in {
    val netParam = new NetParameter()
    ReadProtoFromTextFileOrDie(sparkNetHome + "/models/adult/adult.prototxt", netParam)
    val schema = StructType(StructField("C0", FloatType, false) :: Nil)
    Caffe.set_mode(Caffe.GPU)
    val net = CaffeNet(netParam, schema, new DefaultPreprocessor(schema))
    val inputs = List.range(0, 100).map(x => Row(x.toFloat))
    val weightsBefore = net.getWeights()
    net.forwardBackward(inputs.iterator)
    val weightsAfter = net.getWeights()
    assert(CaffeWeightCollection.checkEqual(weightsBefore, weightsAfter, 1e-10F)) // weights should be equal
  }

  "Saving and loading the weights" should "leave the weights unchanged" in {
    val netParam = new NetParameter()
    ReadProtoFromTextFileOrDie(sparkNetHome + "/models/cifar10/cifar10_quick_train_test.prototxt", netParam)
    val schema = StructType(StructField("data", ArrayType(FloatType), false) :: StructField("label", IntegerType) :: Nil)
    Caffe.set_mode(Caffe.GPU)
    val net1 = CaffeNet(netParam, schema, new DefaultPreprocessor(schema))
    net1.saveWeightsToFile(sparkNetHome + "/temp/cifar10.caffemodel")
    val net2 = CaffeNet(netParam, schema, new DefaultPreprocessor(schema))
    assert(!CaffeWeightCollection.checkEqual(net1.getWeights(), net2.getWeights(), 1e-10F)) // weights should not be equal
    net2.copyTrainedLayersFrom(sparkNetHome + "/temp/cifar10.caffemodel")
    assert(CaffeWeightCollection.checkEqual(net1.getWeights(), net2.getWeights(), 1e-10F)) // weights should be equal
  }

  "Putting input into net and taking it out" should "not change the input" in {
    val netParam = new NetParameter()
    ReadProtoFromTextFileOrDie(sparkNetHome + "/models/test/test.prototxt", netParam)
    val schema = StructType(StructField("data", ArrayType(FloatType), false) :: StructField("label", IntegerType, false) :: Nil)
    Caffe.set_mode(Caffe.GPU)
    val net = CaffeNet(netParam, schema, new DefaultPreprocessor(schema))

    val inputBuffer = new Array[Array[Array[Float]]](net.inputSize)
    assert(net.inputSize == 2)
    for (i <- 0 to net.inputSize - 1) {
      inputBuffer(i) = new Array[Array[Float]](net.batchSize)
      for (batchIndex <- 0 to net.batchSize - 1) {
        inputBuffer(i)(batchIndex) = Array.range(0, net.inputBufferSize(i)).map(e => e.toFloat)
      }
    }

    JavaCPPUtils.arraysToFloatBlobVector(inputBuffer, net.inputs, net.batchSize, net.inputBufferSize, net.inputSize) // put inputBuffer into net.inputs
    val inputBufferOut = JavaCPPUtils.arraysFromFloatBlobVector(net.inputs, net.batchSize, net.inputBufferSize, net.inputSize) // read inputs out of net.inputs

    // check if inputBuffer and inputBufferOut are the same
    for (i <- 0 to net.inputSize - 1) {
      var batchIndex = 0
      while (batchIndex < net.batchSize) {
        var j = 0
        while (j < inputBuffer(i)(batchIndex).length) {
          assert((inputBuffer(i)(batchIndex)(j) - inputBufferOut(i)(batchIndex)(j)).abs <= 1e-10)
          j += 1
        }
        batchIndex += 1
      }
    }

    // do it again
    val inputBuffer2 = new Array[Array[Array[Float]]](net.inputSize)
    assert(net.inputSize == 2)
    for (i <- 0 to net.inputSize - 1) {
      inputBuffer2(i) = new Array[Array[Float]](net.batchSize)
      for (batchIndex <- 0 to net.batchSize - 1) {
        inputBuffer2(i)(batchIndex) = Array.range(0, net.inputBufferSize(i)).map(e => Random.nextFloat)
      }
    }

    JavaCPPUtils.arraysToFloatBlobVector(inputBuffer2, net.inputs, net.batchSize, net.inputBufferSize, net.inputSize) // put inputBuffer into net.inputs
    val inputBufferOut2 = JavaCPPUtils.arraysFromFloatBlobVector(net.inputs, net.batchSize, net.inputBufferSize, net.inputSize) // read inputs out of net.inputs

    // check if inputBuffer and inputBufferOut are the same
    for (i <- 0 to net.inputSize - 1) {
      var batchIndex = 0
      while (batchIndex < net.batchSize) {
        var j = 0
        while (j < inputBuffer2(i)(batchIndex).length) {
          assert((inputBuffer2(i)(batchIndex)(j) - inputBufferOut2(i)(batchIndex)(j)).abs <= 1e-10)
          j += 1
        }
        batchIndex += 1
      }
    }

  }
}
