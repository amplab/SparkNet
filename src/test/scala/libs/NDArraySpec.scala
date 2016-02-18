import org.scalatest._
import libs.NDArray

class NDArraySpec extends FlatSpec {
  val raw = (0 to 3 * 4 * 5 - 1).toArray.map(x => x.toFloat)
  val tensor = NDArray(raw, Array(3, 4, 5))
  assert(tensor.shape.deep == Array(3, 4, 5).deep)

  // test get() and set()
  assert(tensor.get(Array(0, 0, 0)) == 0F)
  assert(tensor.get(Array(1, 2, 3)) == 33F)
  assert(tensor.get(Array(2, 3, 4)) == 59F)
  tensor.set(Array(1, 2, 3), 5F)
  assert(tensor.get(Array(1, 2, 3)) == 5F)
  tensor.set(Array(1, 2, 3), 33F)

  // test toFlat()
  assert(tensor.toFlat().deep == raw.deep)

  // test subarray()
  val subtensor = tensor.subarray(Array(0, 1, 2), Array(1, 3, 5))
  assert(subtensor.shape.deep == Array(1, 2, 3).deep)
  assert(subtensor.toFlat().deep == Array(7F, 8F, 9F, 12F, 13F, 14F).deep)

  // test slice()
  assert(tensor.slice(0, 0).shape.deep == Array(4, 5).deep)
  assert(tensor.slice(1, 0).shape.deep == Array(3, 5).deep)
  assert(tensor.slice(2, 0).shape.deep == Array(3, 4).deep)
  assert(tensor.slice(0, 0).slice(0, 0).shape.deep == Array(5).deep)
  assert(tensor.slice(0, 1).slice(1, 2).toFlat().deep == Array(22F, 27F, 32F, 37F).deep)
  assert(tensor.slice(2, 3).get(Array(1, 2)) == 33F)
  assert(tensor.slice(2, 4).slice(0, 2).toFlat().deep == Array(44F, 49F, 54F, 59F).deep)

  // test plus()
  val a1 = NDArray(Array(1F, 2F, 3F, 4F), Array(2, 2))
  val a2 = NDArray(Array(1F, 3F, 5F, 7F), Array(2, 2))
  val a3 = NDArray(Array(-2F, -5F, -8F, -11F), Array(2, 2))
  assert(NDArray.plus(NDArray.plus(a1, a2), a3).toFlat().deep == NDArray.zeros(Array(2, 2)).toFlat().deep)
  assert(NDArray.plus(tensor.subarray(Array(0, 0, 0), Array(1, 2, 3)), tensor.subarray(Array(1, 1, 1), Array(2, 3, 4))).toFlat().deep == Array(26F, 28F, 30F, 36F, 38F, 40F).deep)

  // test subtract()
  val a4 = NDArray(Array(1F, 2F, 3F, 3F, 2F, 1F), Array(2, 3))
  val a5 = NDArray(Array(1F, 3F, 5F, 5F, 3F, 1F), Array(2, 3))
  val a6 = NDArray(Array(0F, -1F, -2F, -2F, -1F, 0F), Array(2, 3))
  a4.subtract(a5)
  assert(a4.toFlat().deep == a6.toFlat().deep)
  a4.add(a5)
  assert(a4.toFlat().deep == Array(1F, 2F, 3F, 3F, 2F, 1F).deep)

  // test scalarDivide()
  val a7 = NDArray(Array(1F, 2F, 3F, 4F, 5F, 6F), Array(3, 2))
  val a8 = NDArray(Array(0.5F, 1F, 1.5F, 2F, 2.5F, 3F), Array(3, 2))
  a7.scalarDivide(2F)
  assert(a7.toFlat().deep == a8.toFlat().deep)
  a7.scalarDivide(0.5F)
  assert(a7.toFlat().deep == Array(1F, 2F, 3F, 4F, 5F, 6F).deep)

  // test flatCopyFast
  val rand = new java.util.Random();
  val a9 = NDArray(Array.fill(3 * 4 * 4)(rand.nextFloat), Array(3, 4, 4))
  val flatBuffer1 = new Array[Float](3 * 2 * 2)
  val flatBuffer2 = new Array[Float](3 * 2 * 2)
  val a10 = a9.subarray(Array(0, 1, 1), Array(3, 3, 3))
  a10.flatCopySlow(flatBuffer1)
  a10.flatCopy(flatBuffer2)
  val epsilon = 1e-7f
  for (i <- 0 to flatBuffer1.length - 1) {
    assert(math.abs(flatBuffer1(i) - flatBuffer2(i)) <= epsilon)
  }

  // performance test flatCopyFast
  val a11 = NDArray(Array.fill(3 * 256 * 256)(rand.nextFloat), Array(3, 256, 256))
  val a12 = a11.subarray(Array(0, 10, 10), Array(3, 237, 237))
  val flatBuffer3 = new Array[Float](3 * 227 * 227)
  val flatBuffer4 = new Array[Float](3 * 227 * 227)
  var startTime = System.currentTimeMillis()
  a12.flatCopySlow(flatBuffer3)
  var endTime = System.currentTimeMillis()
  print("flatCopy() took " + (1F * (endTime - startTime) / 1000).toString + "s\n")
  startTime = System.currentTimeMillis()
  a12.flatCopy(flatBuffer4)
  endTime = System.currentTimeMillis()
  print("flatCopyFast() took " + (1F * (endTime - startTime) / 1000).toString + "s\n")
  for (i <- 0 to flatBuffer3.length - 1) {
    assert(math.abs(flatBuffer3(i) - flatBuffer4(i)) <= epsilon)
  }
}
