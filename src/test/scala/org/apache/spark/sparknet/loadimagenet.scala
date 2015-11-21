import org.scalatest._

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

class ImageNetLoaderSpec extends FlatSpec {
	"Imagenet" should "be loadable from S3" in {
		val conf = new SparkConf().setAppName("ImageLoaderTest").setMaster("local")
		val sc = new SparkContext(conf)
		val loader = new ImageNetLoader("sparknet")
		val rdd = loader.apply(sc, "shuffled_trainset/files-shuf-000", "train_correct.txt")
		assert(rdd.count == 1281)
	}
}
