import org.scalatest._

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}

import libs._

class PreprocessorSpec extends FlatSpec with BeforeAndAfterAll {
  val conf = new SparkConf().setAppName("TestSpec").setMaster("local")
  private var sc: SparkContext = null
  private var sqlContext: org.apache.spark.sql.SQLContext = null

  override protected def beforeAll(): Unit = {
    sc = new SparkContext(conf)
    sqlContext = new org.apache.spark.sql.SQLContext(sc)
  }

  override protected def afterAll(): Unit = {
    sc.stop()
  }

  "DefaultPreprocessor" should "preserve scalar values" in {
    val typesAndValues = List((IntegerType, 1), (FloatType, 1F), (DoubleType, 1D), (LongType, 1L))
    typesAndValues.foreach {
      case (t, v) => {
        val schema = StructType(StructField("x", t, false) :: Nil)
        val preprocessor = new DefaultPreprocessor(schema)
        val convert = preprocessor.convert("x", Array[Int](1))
        var x = Row(v)
        val df = sqlContext.createDataFrame(sc.parallelize(Array(x)), schema)
        val buffer = new Array[Float](1)
        convert(df.take(1)(0)(0), buffer)
        assert(buffer.deep == Array[Float](1).deep)
      }
    }
  }

  "DefaultPreprocessor" should "preserve array values" in {
    // val typesAndValues = List((ArrayType(IntegerType), Array[Int](0, 1, 2)), (ArrayType(FloatType), Array[Float](0, 1, 2)), (ArrayType(DoubleType), Array[Double](0, 1, 2)), (ArrayType(BinaryType), Array[Byte](0, 1, 2)))
    val typesAndValues = List((ArrayType(FloatType), Array[Float](0, 1, 2)),
                              (BinaryType, Array[Byte](0, 1, 2)))
    typesAndValues.foreach {
      case (t, v) => {
        val schema = StructType(StructField("x", t, false) :: Nil)
        val preprocessor = new DefaultPreprocessor(schema)
        val convert = preprocessor.convert("x", Array[Int](1, 3))
        var x = Row(v)
        val df = sqlContext.createDataFrame(sc.parallelize(Array(x)), schema)
        val buffer = new Array[Float](3)
        convert(df.take(1)(0)(0), buffer)
        assert(buffer.deep == Array[Float](0, 1, 2).deep)
      }
    }
  }

  "DefaultPreprocessor" should "be fast" in {
    // val typesAndValues = List((ArrayType(IntegerType), Array[Int](0, 1, 2)), (ArrayType(FloatType), Array[Float](0, 1, 2)), (ArrayType(DoubleType), Array[Double](0, 1, 2)), (ArrayType(BinaryType), Array[Byte](0, 1, 2)))
    val typesAndValues = List((ArrayType(FloatType), new Array[Float](256 * 256)),
                              (BinaryType, new Array[Byte](256 * 256)))
    val array = new Array[Float](256 * 256)
    typesAndValues.foreach {
      case (t, v) => {
        val schema = StructType(StructField("x", t, false) :: Nil)
        val preprocessor = new DefaultPreprocessor(schema)
        val convert = preprocessor.convert("x", Array[Int](256, 256))
        var x = Row(v)
        val df = sqlContext.createDataFrame(sc.parallelize(Array(x)), schema)
        val startTime = System.currentTimeMillis()
        val xExtracted = df.take(1)(0)
        val buffer = new Array[Float](256 * 256)
        for (i <- 0 to 256 - 1) {
          convert(xExtracted(0), buffer)
        }
        val endTime = System.currentTimeMillis()
        val totalTime = (endTime - startTime) * 1F / 1000
        print("DefaultPreprocessor converted 256 images in " + totalTime.toString + "s\n")
        assert(totalTime <= 1.0)
      }
    }
  }

  "ImageNetPreprocessor" should "subtract mean" in {
    val fullHeight = 4
    val fullWidth = 5
    val croppedHeight = 4
    val croppedWidth = 5
    val schema = StructType(StructField("x", BinaryType, false) :: Nil)
    val meanImage = Array.range(0, 3 * fullHeight * fullWidth).map(e => e.toFloat)
    val preprocessor = new ImageNetPreprocessor(schema, meanImage, fullHeight, fullWidth, croppedHeight, croppedWidth)
    val convert = preprocessor.convert("x", Array[Int](3, croppedHeight, croppedWidth))
    val image = Array.range(0, 3 * fullHeight * fullWidth).map(e => e.toByte)
    var x = Row(image)
    val df = sqlContext.createDataFrame(sc.parallelize(Array(x)), schema)
    val buffer = new Array[Float](3 * croppedHeight * croppedWidth)
    convert(df.take(1)(0)(0), buffer)
    assert(buffer.deep == (image.map(e => e.toFloat), meanImage).zipped.map(_ - _).deep)
  }

  "ImageNetPreprocessor" should "subtract mean and crop image" in {
    val fullHeight = 4
    val fullWidth = 5
    val croppedHeight = 2
    val croppedWidth = 4
    val schema = StructType(StructField("x", BinaryType, false) :: Nil)
    val meanImage = new Array[Float](3 * fullHeight * fullWidth)
    val preprocessor = new ImageNetPreprocessor(schema, meanImage, fullHeight, fullWidth, croppedHeight, croppedWidth)
    val convert = preprocessor.convert("x", Array[Int](3, croppedHeight, croppedWidth))
    val image = Array.range(0, 3 * fullHeight * fullWidth).map(e => e.toByte)
    var x = Row(image)
    val df = sqlContext.createDataFrame(sc.parallelize(Array(x)), schema)
    val buffer = new Array[Float](3 * croppedHeight * croppedWidth)
    convert(df.take(1)(0)(0), buffer)
    val convertedImage = NDArray(buffer, Array[Int](3, croppedHeight, croppedWidth))
    assert(convertedImage.shape.deep == Array[Int](3, croppedHeight, croppedWidth).deep)
    val cornerVal = convertedImage.get(Array[Int](0, 0, 0))
    assert(Set[Float](0, 1, 5, 6, 10, 11).contains(cornerVal))
    assert(convertedImage.toFlat().map(e => e - cornerVal).deep == Array[Float](0,  1,  2,  3,   5,  6,  7,  8,
                                                                                20, 21, 22, 23,  25, 26, 27, 28,
                                                                                40, 41, 42, 43,  45, 46, 47, 48).deep)
  }

  "ImageNetPreprocessor" should "be fast" in {
    val fullHeight = 256
    val fullWidth = 256
    val croppedHeight = 227
    val croppedWidth = 227
    val schema = StructType(StructField("x", BinaryType, false) :: Nil)
    val meanImage = new Array[Float](3 * fullHeight * fullWidth)
    val preprocessor = new ImageNetPreprocessor(schema, meanImage, fullHeight, fullWidth, croppedHeight, croppedWidth)
    val convert = preprocessor.convert("x", Array[Int](3, croppedHeight, croppedWidth))
    val image = Array.range(0, 3 * fullHeight * fullWidth).map(e => e.toByte)
    var x = Row(image)
    val df = sqlContext.createDataFrame(sc.parallelize(Array(x)), schema)
    val xExtracted = df.take(1)(0)
    val startTime = System.currentTimeMillis()
    val buffer = new Array[Float](3 * croppedHeight * croppedWidth)
    for (i <- 0 to 256 - 1) {
      convert(xExtracted(0), buffer)
    }
    val endTime = System.currentTimeMillis()
    val totalTime = (endTime - startTime) * 1F / 1000
    print("ImageNetPreprocessor converted 256 images in " + totalTime.toString + "s\n")
    assert(totalTime <= 0.2)
  }

}
