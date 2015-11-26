package apps

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

import libs.ProtoLoader
import libs.CaffeLibrary
import libs.CaffeNet
import libs.WeightCollection
import libs.MinibatchSampler
import loaders.CifarLoader
import preprocessing.ScaleAndConvert

// for this app to work, $SPARKNET_HOME should be the SparkNet root directory
// and you need to run $SPARKNET_HOME/caffe/data/cifar10/get_cifar10.sh
object CifarApp {
	val sparkNetHome = sys.env("SPARKNET_HOME")

	System.load(sparkNetHome + "/build/libccaffe.so")
  val caffeLib = CaffeLibrary.INSTANCE

	val batchsize = 100
	val channels = 3
	val imgSize = 32

	var netParameter = ProtoLoader.loadNetPrototxt(sparkNetHome + "/caffe/examples/cifar10/cifar10_full_train_test.prototxt")
	netParameter = ProtoLoader.replaceDataLayers(netParameter, batchsize, channels, imgSize, imgSize)
	val solverParameter = ProtoLoader.loadSolverPrototxtWithNet(sparkNetHome + "/caffe/examples/cifar10/cifar10_full_solver.prototxt", netParameter, None)

	val nnet = CaffeNet(solverParameter)
	var netWeights = nnet.getWeights()

  def main(args: Array[String]) {
		val numWorkers = 2

		val conf = new SparkConf().setAppName("CifarApp").setMaster("local")
		val sc = new SparkContext(conf)

		val loader = new CifarLoader(sparkNetHome + "/caffe/data/cifar10/")

		val trainRDD = sc.parallelize(loader.trainImages.zip(loader.trainLabels.map(x => x.toInt)))
		val converter = new ScaleAndConvert(batchsize, imgSize, imgSize)

		var trainMinibatchRDD = converter.makeMinibatchRDDWithoutCompression(trainRDD).persist()
		trainMinibatchRDD = trainMinibatchRDD.coalesce(numWorkers)
		val trainPartitionSizes = trainMinibatchRDD.mapPartitions(iter => Array(iter.size).iterator).persist()

		print("size of dataset is " + trainMinibatchRDD.count().toString + "\n")

		val workers = sc.parallelize(Array.range(0, numWorkers), numWorkers)

		var i = 0
		val syncInterval = 10
		while (true) {
			val broadcastWeights = sc.broadcast(netWeights)
			trainPartitionSizes.zipPartitions(trainMinibatchRDD) (
				(lenIt, trainMinibatchIt) => {
					val len = lenIt.next
					val minibatchSampler = new MinibatchSampler(trainMinibatchIt, len, syncInterval)
					nnet.setTrainData(minibatchSampler, None)
					nnet.train(syncInterval)
					Array(0).iterator
				}
			).foreachPartition(_ => ())
			netWeights = workers.map(_ => { nnet.getWeights() }).reduce((a, b) => WeightCollection.add(a, b))
			netWeights.scalarDivide(1F * numWorkers)
			i += 1
		}
	}
}
