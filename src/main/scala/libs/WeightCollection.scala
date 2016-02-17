package libs

import scala.collection.mutable.Map
import scala.collection.mutable.MutableList

class WeightCollection(val allWeights: Map[String, MutableList[NDArray]], val layerNames: List[String]) extends java.io.Serializable {
  val numLayers = layerNames.length

  def scalarDivide(v: Float) = {
    for (i <- 0 to numLayers - 1) {
      for (j <- 0 to allWeights(layerNames(i)).length - 1) {
        allWeights(layerNames(i))(j).scalarDivide(v)
      }
    }
  }
}

object WeightCollection extends java.io.Serializable {
  def add(wc1: WeightCollection, wc2: WeightCollection): WeightCollection = {
    assert(wc1.layerNames == wc2.layerNames)
    val layerNames = wc1.layerNames
    //check that the WeightCollection objects have the same shape
    for (i <- 0 to wc1.numLayers - 1) {
      assert(wc1.allWeights(layerNames(i)).length == wc2.allWeights(layerNames(i)).length)
      for (j <- 0 to wc1.allWeights(layerNames(i)).length - 1) {
        assert(wc1.allWeights(layerNames(i))(j).shape.deep == wc2.allWeights(layerNames(i))(j).shape.deep)
      }
    }
    // add the WeightCollection objects together
    var newWeights = Map[String, MutableList[NDArray]]()
    for (i <- 0 to wc1.numLayers - 1) {
      newWeights += (layerNames(i) -> MutableList())
      for (j <- 0 to wc1.allWeights(wc1.layerNames(i)).length - 1) {
        newWeights(layerNames(i)) += NDArray.plus(wc1.allWeights(layerNames(i))(j), wc2.allWeights(layerNames(i))(j))
      }
    }
    return new WeightCollection(newWeights, layerNames)
  }
}
