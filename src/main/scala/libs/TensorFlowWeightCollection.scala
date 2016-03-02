package libs

import scala.collection.mutable.Map
import scala.collection.mutable.MutableList

object TensorFlowWeightCollection {
	def scalarDivide(weights: Map[String, NDArray], v: Float) = {
		for (name <- weights.keys) {
      weights(name).scalarDivide(v)
    }
	}

  def add(wc1: Map[String, NDArray], wc2: Map[String, NDArray]): Map[String, NDArray] = {
    assert(wc1.keys == wc2.keys)
    // add the WeightCollection objects together
    var newWeights = Map[String, NDArray]()
    for (name <- wc1.keys) {
      newWeights += (name -> NDArray.plus(wc1(name), wc2(name)))
    }
    newWeights
  }

  def checkEqual(wc1: Map[String, NDArray], wc2: Map[String, NDArray], tol: Float): Boolean = {
    if (wc1.keys != wc2.keys) {
      return false
    }
    for (name <- wc1.keys) {
      if (!NDArray.checkEqual(wc1(name), wc2(name), tol)) {
        return false
      }
    }
    return true
  }
}
