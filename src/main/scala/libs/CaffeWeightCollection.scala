package libs

import scala.collection.mutable.Map
import scala.collection.mutable.MutableList

object CaffeWeightCollection {

  def scalarDivide(weights: Map[String, MutableList[Array[Float]]], v: Float) = {
    for (name <- weights.keys) {
      for (j <- 0 to weights(name).length - 1) {
        weights(name)(j) = weights(name)(j).map( (x: Float) => x / v )
      }
    }
  }


  def add(weights1: Map[String, MutableList[Array[Float]]], weights2: Map[String, MutableList[Array[Float]]]): Map[String, MutableList[Array[Float]]] = {
    val t1 = System.currentTimeMillis()
    if (weights1.keys != weights2.keys) {
      throw new Exception("weights1.keys != weights2.keys, weights1.keys = " + weights1.keys.toString + ", and weights2.keys = " + weights2.keys.toString + "\n")
    }
    val newWeights = Map[String, MutableList[Array[Float]]]()
    for (name <- weights1.keys) {
      newWeights += (name -> MutableList())
      if (weights1(name).length != weights2(name).length) {
        throw new Exception("weights1(name).length != weights2(name).length, name = " + name + ", weights1(name).length = " + weights1(name).length.toString + ", weights2(name).length = " + weights2(name).length.toString)
      }
      for (j <- 0 to weights1(name).length - 1) {
        if (weights1(name)(j).length != weights2(name)(j).length) {
          throw new Exception("weights1(name)(j).shape != weights2(name)(j).shape, name = " + name + ", j = " + j.toString + ", weights1(name)(j).shape = " + weights1(name)(j).length + "weigths2 = " + weights2(name)(j).length)
        }
        newWeights(name) += weights1(name)(j).zip(weights2(name)(j)).map( a => a._1 + a._2 )
      }
    }
    val t2 = System.currentTimeMillis()
    print("addWeights took " + ((t2 - t1) * 1F / 1000F).toString + " s\n")

    newWeights
  }


  def checkEqual(weights1: Map[String, MutableList[NDArray]], weights2: Map[String, MutableList[NDArray]], tol: Float): Boolean = {
    if (weights1.keys != weights2.keys) {
      return false
    }
    for (name <- weights1.keys) {
      if (weights1(name).length != weights2(name).length) {
        return false
      }
      for (j <- 0 to weights1(name).length - 1) {
        if (!NDArray.checkEqual(weights1(name)(j), weights2(name)(j), tol)) {
          return false
        }
      }
    }
    return true
  }

}
