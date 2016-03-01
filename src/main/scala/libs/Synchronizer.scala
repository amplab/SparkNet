package libs

import scala.collection.mutable._

trait Synchronizer {
  def synchronize(weightList: Array[Map[String, NDArray]]): Array[Map[String, NDArray]]
}

class AveragingSynchronizer() extends Synchronizer {
  def synchronize(weightList: Array[Map[String, NDArray]]): Array[Map[String, NDArray]] = {
    val average = weightList.reduce((a, b) => TensorFlowWeightCollection.add(a, b))
    TensorFlowWeightCollection.scale(average, 1F / weightList.length)
    Array.range(0, weightList.length).map(_ => average)
  }
}

class ElasticAveragingSynchronizer(beta: Float) extends Synchronizer {
  def synchronize(weightList: Array[Map[String, NDArray]]): Array[Map[String, NDArray]] = {
    val average = weightList.reduce((a, b) => TensorFlowWeightCollection.add(a, b))
    TensorFlowWeightCollection.scale(average, beta / weightList.length)
		weightList.foreach(a => TensorFlowWeightCollection.scale(a, (1 - beta)))
		Array.range(0, weightList.length).map(i => TensorFlowWeightCollection.add(weightList(i), average))
  }
}
