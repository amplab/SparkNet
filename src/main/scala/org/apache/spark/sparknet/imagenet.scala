package org.apache.spark.sparknet

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

import org.apache.spark.sparknet.preprocessing.ScaleAndConvert
import org.apache.spark.sparknet.loaders.ImageNetLoader

object ImageNetApp {
	def main(args: Array[String]) {
		val syncIter = args(0).toInt
		val numWorkers = args(1).toInt
		val conf = new SparkConf()
			.setAppName("ImagenetGrad")
			.set("spark.driver.maxResultSize", "30G")
			.set("spark.task.maxFailures", "1")
			.set("spark.eventLog.enabled", "true")
		val sc = new SparkContext(conf)
		val loader = new ImageNetLoader("sparknet")
		val rdd = loader.apply(sc, "shuffled_trainset/files-shuf-", "train_correct.txt")
		val repartitionedRDD = rdd.repartition(numWorkers)
		val converter = new ScaleAndConvert(256, 256)
		val result = converter.apply(repartitionedRDD).persist()
		print("number of training images is " + result.count().toString)
	}
}
