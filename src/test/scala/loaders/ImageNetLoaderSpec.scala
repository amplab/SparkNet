import org.scalatest._

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

import preprocessing.ScaleAndConvert
import loaders.ImageNetLoader

class ImageNetLoaderSpec extends FlatSpec {
  ignore should "be loadable from S3" in {
    val conf = new SparkConf().setAppName("ImageLoaderTest").setMaster("local")
    val sc = new SparkContext(conf)
    val loader = new ImageNetLoader("sparknet")
    val rdd = loader.apply(sc, "ILSVRC2012_training/", "train.txt")
    assert(rdd.count == 1281)
  }
  // TODO: Update this test
  ignore should "do preprocessing" in {
    val conf = new SparkConf().setAppName("ImageLoaderTest").setMaster("local")
    val sc = new SparkContext(conf)
    val loader = new ImageNetLoader("sparknet")
    val rdd = loader.apply(sc, "ILSVRC2012_train/", "train.txt")
    val converter = new ScaleAndConvert(1, 256, 256)
    val result = converter.makeMinibatchRDDWithCompression(rdd).collect()
    assert(result.length == 1281)
  }
}
