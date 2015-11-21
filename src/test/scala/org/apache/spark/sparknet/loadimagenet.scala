import org.scalatest._

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

import org.apache.spark.sparknet.preprocessing.ScaleAndConvert
import org.apache.spark.sparknet.loaders.ImageNetLoader

class ImageNetLoaderSpec extends FlatSpec {
	ignore should "be loadable from S3" in {
		val conf = new SparkConf().setAppName("ImageLoaderTest").setMaster("local")
		val sc = new SparkContext(conf)
		val loader = new ImageNetLoader("sparknet")
		val rdd = loader.apply(sc, "shuffled_trainset/files-shuf-000", "train_correct.txt")
		assert(rdd.count == 1281)
	}
	// TODO: Update this test
	ignore should "do preprocessing" in {
		val conf = new SparkConf().setAppName("ImageLoaderTest").setMaster("local")
		val sc = new SparkContext(conf)
		val loader = new ImageNetLoader("sparknet")
		val rdd = loader.apply(sc, "shuffled_trainset/files-shuf-000", "train_correct.txt")
		val converter = new ScaleAndConvert(256, 256)
		val result = converter.apply(rdd).collect()
		assert(result.length == 1281)
	}
}
