import libs._

import org.scalatest._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

import java.nio.file.Paths

class LoadSqlDatasetSpec extends FlatSpec {
  "LoadSqlDataset" should "be able to load the adult dataset" in {
    val conf = new SparkConf().setAppName("DataFrameLoaderTest").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val sparkNetHome = sys.env("SPARKNET_HOME")

    val dataset = Paths.get(sparkNetHome, "data/adult.data").toString()
    val df = sqlContext.read.format("com.databricks.spark.csv").option("inferSchema", "true").load(dataset)
    val preprocessor = new DefaultPreprocessor(df.schema)

    val function0 = preprocessor.convert("C0", Array[Int](1))
    val function2 = preprocessor.convert("C2", Array[Int](1))
    val result0 = function0(df.take(1)(0)(0))
    val result2 = function2(df.take(1)(0)(2))

    assert((result0.get(Array[Int](0)) - 39.0).abs <= 1e-4)
    assert((result2.get(Array[Int](0)) - 77516.0).abs <= 1e-4)
  }
}
