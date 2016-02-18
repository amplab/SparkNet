import org.scalatest._

import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import org.bytedeco.javacpp.caffe._

import libs._

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
    val net = CaffeNet(netParam, schema, new DefaultPreprocessor(schema))
  }

  "CaffeNet" should "call forward" in {
    val netParam = new NetParameter()
    ReadProtoFromTextFileOrDie(sparkNetHome + "/models/adult/adult.prototxt", netParam)
    val schema = StructType(StructField("C0", FloatType, false) :: Nil)
    val net = CaffeNet(netParam, schema, new DefaultPreprocessor(schema))
    val inputs = List[Row](Row(0F), Row(1F))
    val outputs = net.forward(inputs.iterator)
    val keys = outputs.keys.toArray
    assert(keys.length == 1)
    assert(keys(0) == "prob")
    assert(outputs("prob").shape.deep == Array[Int](64, 10).deep) // these numbers are taken from adult.prototxt
  }

  "CaffeNet" should "call forwardBackward" in {
    val netParam = new NetParameter()
    ReadProtoFromTextFileOrDie(sparkNetHome + "/models/adult/adult.prototxt", netParam)
    val schema = StructType(StructField("C0", FloatType, false) :: Nil)
    val net = CaffeNet(netParam, schema, new DefaultPreprocessor(schema))
    val inputs = List.range(0, 100).map(x => Row(x.toFloat))
    net.forwardBackward(inputs.iterator)
  }

  "Calling forward" should "leave weights unchanged" in {
    val netParam = new NetParameter()
    ReadProtoFromTextFileOrDie(sparkNetHome + "/models/adult/adult.prototxt", netParam)
    val schema = StructType(StructField("C0", FloatType, false) :: Nil)
    val net = CaffeNet(netParam, schema, new DefaultPreprocessor(schema))
    val inputs = List[Row](Row(0F), Row(1F))
    val weightsBefore = net.getWeights()
    val outputs = net.forward(inputs.iterator)
    val weightsAfter = net.getWeights()

    // check that the weights are unchanged
    assert(weightsBefore.layerNames == weightsAfter.layerNames)
    val layerNames = weightsBefore.layerNames
    for (i <- 0 to weightsBefore.numLayers - 1) {
      assert(weightsBefore.allWeights(layerNames(i)).length == weightsAfter.allWeights(layerNames(i)).length)
      for (j <- 0 to weightsBefore.allWeights(layerNames(i)).length - 1) {
        val weightBefore = weightsBefore.allWeights(layerNames(i))(j).toFlat
        val weightAfter = weightsAfter.allWeights(layerNames(i))(j).toFlat
        for (k <- 0 to weightBefore.length - 1) {
          assert((weightBefore(k) - weightAfter(k)).abs <= 1e-6)
        }
      }
    }
  }

  "Calling forwardBackward" should "leave weights unchanged" in {
    val netParam = new NetParameter()
    ReadProtoFromTextFileOrDie(sparkNetHome + "/models/adult/adult.prototxt", netParam)
    val schema = StructType(StructField("C0", FloatType, false) :: Nil)
    val net = CaffeNet(netParam, schema, new DefaultPreprocessor(schema))
    val inputs = List.range(0, 100).map(x => Row(x.toFloat))
    val weightsBefore = net.getWeights()
    net.forwardBackward(inputs.iterator)
    val weightsAfter = net.getWeights()

    // check that the weights are unchanged
    assert(weightsBefore.layerNames == weightsAfter.layerNames)
    val layerNames = weightsBefore.layerNames
    for (i <- 0 to weightsBefore.numLayers - 1) {
      assert(weightsBefore.allWeights(layerNames(i)).length == weightsAfter.allWeights(layerNames(i)).length)
      for (j <- 0 to weightsBefore.allWeights(layerNames(i)).length - 1) {
        val weightBefore = weightsBefore.allWeights(layerNames(i))(j).toFlat
        val weightAfter = weightsAfter.allWeights(layerNames(i))(j).toFlat
        for (k <- 0 to weightBefore.length - 1) {
          assert((weightBefore(k) - weightAfter(k)).abs <= 1e-6)
        }
      }
    }
  }
}
