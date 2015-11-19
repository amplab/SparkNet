/*
package org.apache.spark.sparknet

import scala.reflect.ClassTag

class NDArray private(val data: Array[Float], val dim: Int, val shape: Array[Int], val offset: Int, val strides: Array[Int]) extends java.io.Serializable {
  assert(dim >= 1)
  assert(shape.length == dim)
  assert(strides.length == dim)

  def subarray(lowerOffsets: Array[Int], upperOffsets: Array[Int]): NDArray = {
    assert(lowerOffsets.length == dim)
    assert(upperOffsets.length == dim)
    for (i <- 0 to dim - 1) {
      assert(0 <= lowerOffsets(i))
      assert(lowerOffsets(i) < upperOffsets(i))
      assert(upperOffsets(i) <= shape(i))
    }
    var newOffset = offset
    val newShape = new Array[Int](dim)
    for (i <- 0 to dim - 1) {
      newOffset += lowerOffsets(i) * strides(i)
      newShape(i) = upperOffsets(i) - lowerOffsets(i)
    }
    return new NDArray(data, dim, newShape, newOffset, strides)
  }

  def slice(axis: Int, index: Int): NDArray = {
    assert(dim >= 2) // we do not allow slicing of one-dimensional arrays right now because we do not support zero-dimensional arrays yet
    assert(0 <= index && index < shape(axis))
    val newShape = Array.concat(shape.slice(0, axis), shape.slice(axis + 1, dim))
    val newStrides = Array.concat(strides.slice(0, axis), strides.slice(axis + 1, dim))
    val newOffset = offset + index * strides(axis)
    return new NDArray(data, dim - 1, newShape, newOffset, newStrides)
  }

  def get(indices: Array[Int]): Float = {
    return data(computeInd(indices))
  }

  def set(indices: Array[Int], value: Float) = {
    data(computeInd(indices)) = value
  }

  def flatCopy(result: Array[Float]) = {
    assert(result.length == shape.product)
    val currIndices = new Array[Int](dim)
    //for (i <- 0 to dim - 1) {
    //  assert(currIndices(i) == 0) // this isn't necessary, just checking if Scala zero-initializes arrays
    //}
    val size = shape.product
    for (index <- 0 to shape.product - 1) {
      result(index) = get(currIndices)
      if (index != size - 1) {
        next(currIndices)
      }
    }
    //for (i <- 0 to dim - 1) {
    //  assert(currIndices(i) + 1 == shape(i)) // just another check
    //}
  }

  def toFlat(): Array[Float] = {
    //implicit val man = ClassTag[V](data.getClass.getComponentType.asInstanceOf[Class[V]]) // todo(this is so ugly, but we need it for now because apparently otherwise you can't declare Array[V] in scala...)
    val result = new Array[Float](shape.product)
    flatCopy(result)
    return result
  }

  //def add(that: NDArray) = {
  //  // add in place
  //  assert(shape.deep == that.shape.deep)
  //  val thisIndices = new Array[Int](dim)
  //  val thatIndices = new Array[Int](dim)
  //  for (i <- 0 to shape.product - 1) {
  //    data(computeInd(thisIndices)) += that.get(thatIndices)
  //    if (i != shape.product - 1) {
  //      next(thisIndices)
  //      that.next(thatIndices)
  //    }
  //  }
  //}

  def subtract(that: NDArray) = {
    assert(shape.deep == that.shape.deep)
    // specialized code for the case when the strides are the default strides
    if (strides.deep == NDArray.calcDefaultStrides(shape).deep && offset == 0 && that.strides.deep == NDArray.calcDefaultStrides(that.shape).deep && that.offset == 0) {
      for (i <- 0 to shape.product - 1) {
        data(i) -= that.data(i)
      }
    }
    // otherwise use general purpose code
    else {
      val indices = new Array[Int](dim)
      //val thatIndices = new Array[Int](dim)
      for (i <- 0 to shape.product - 1) {
        data(computeInd(indices)) -= that.get(indices)
        if (i != shape.product - 1) {
          next(indices)
        }
      }
    }
  }

  def scalarDivide(v: Float) = { // come up with consistent way to describe these methods (do they take NDArrays or just values?)
    // specialized code for the case when the strides are the default strides
    if (strides.deep == NDArray.calcDefaultStrides(shape).deep && offset == 0) {
      for (i <- 0 to shape.product - 1) {
        data(i) /= v
      }
    }
    // otherwise, use general purpose code
    else {
      var indices = new Array[Int](dim)
      for (i <- 0 to shape.product - 1) {
        data(computeInd(indices)) /= v
      }
    }
  }

  private def computeInd(indices: Array[Int]): Int = {
    //assert(indices.length == dim)
    //for (i <- 0 to dim - 1) {
    //  assert(0 <= indices(i) && indices(i) <= shape(i) - 1)
    //}
    var ind = offset
    for (i <- 0 to dim - 1) {
      ind += indices(i) * strides(i)
    }
    return ind
  }

  private def next(indices: Array[Int]) = {
    //assert(indices.length == dim)
    //for (i <- 0 to dim - 1) {
    //  assert(0 <= indices(i) && indices(i) < shape(i))
    //}
    var axis = dim - 1
    while (indices(axis) == shape(axis) - 1) {
      indices(axis) = 0
      axis -= 1
      //assert(axis >= 0)
    }
    indices(axis) += 1
  }
}

// todo: implement addition, subtraction, division, multiplication, and overload appropriate operators

//object NDArray extends java.io.Serializable { // todo: could just make "plus" a value instead of method, so it can be serialized
object NDArray {
  def apply(data: Array[Float], shape: Array[Int]) = new NDArray(data, shape.length, shape, 0, calcDefaultStrides(shape))

  def zeros(shape: Array[Int]) = new NDArray(new Array[Float](shape.product), shape.length, shape, 0, calcDefaultStrides(shape))

  def plus(v1: NDArray, v2: NDArray): NDArray = {
    assert(v1.shape.deep == v2.shape.deep)

    val indices = new Array[Int](v1.dim)
    val size = v1.shape.product
    val data = new Array[Float](size)

    // faster code when both v1 and v2 have the default strides
    if (v1.strides.deep == calcDefaultStrides(v1.shape).deep && v1.offset == 0 && v2.strides.deep == calcDefaultStrides(v2.shape).deep && v2.offset == 0) {
      for (i <- 0 to size - 1) {
        data(i) = v1.data(i) + v2.data(i)
      }
      return new NDArray(data, v1.dim, v1.shape, 0, calcDefaultStrides(v1.shape)) // do we want to copy shape?
    }

    for (i <- 0 to size - 1) {
      data(i) = v1.get(indices) + v2.get(indices)
      if (i != size - 1) {
        v1.next(indices)
      }
    }
    return new NDArray(data, v1.dim, v1.shape, 0, calcDefaultStrides(v1.shape))
  }

  private def calcDefaultStrides(shape: Array[Int]): Array[Int] = {
    // this code exists in both the class and object... probably a bad idea?
    val dim = shape.length
    val strides = new Array[Int](dim)
    var st = 1
    for (j <- dim - 1 to 0 by -1) {
      strides(j) = st
      st *= shape(j)
    }
    return strides
  }
}
*/

package org.apache.spark.sparknet

class NDArray private(val javaArray: JavaNDArray) extends java.io.Serializable {
  val dim = javaArray.dim
  val shape = javaArray.shape

  def subarray(lowerOffsets: Array[Int], upperOffsets: Array[Int]): NDArray = {
    return new NDArray(javaArray.subArray(lowerOffsets, upperOffsets))
  }

  def slice(axis: Int, index: Int): NDArray = {
    return new NDArray(javaArray.slice(axis, index))
  }

  def get(indices: Array[Int]): Float = {
    return javaArray.get(indices:_*)
  }

  def set(indices: Array[Int], value: Float) = {
    javaArray.set(indices, value)
  }

  def flatCopy(result: Array[Float]) = {
    javaArray.flatCopy(result)
  }

  def toFlat(): Array[Float] = {
    return javaArray.toFlat()
  }

  def add(that: NDArray) = {
    javaArray.add(that.javaArray)
  }

  def subtract(that: NDArray) = {
    javaArray.subtract(that.javaArray)
  }

  def scalarDivide(v: Float) = {
    javaArray.scalarDivide(v)
  }
}

object NDArray {
  def apply(data: Array[Float], shape: Array[Int]) = {
    new NDArray(new JavaNDArray(data, shape:_*))
  }

  def zeros(shape: Array[Int]) = new NDArray(new JavaNDArray(new Array[Float](shape.product), shape:_*))

  def plus(v1: NDArray, v2: NDArray): NDArray = {
    val v = new NDArray(new JavaNDArray(v1.toFlat(), v1.shape:_*))
    v.add(v2)
    return v
  }
}
