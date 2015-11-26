import org.scalatest._
import libs.MinibatchSampler
import libs.ByteNDArray

class MinibatchSamplerSpec extends FlatSpec {
  val minibatches = Array.range(0, 100).map(
    i => {
      (Array(ByteNDArray(Array(i.toByte), Array(1))), Array(i))
    }
  )

  var sampler = new MinibatchSampler(minibatches.iterator, minibatches.length, 5)
  for (i <- 0 to 4) {
    var imageMinibatch = sampler.nextImageMinibatch()
    var labelMinibatch = sampler.nextLabelMinibatch()
    assert(imageMinibatch(0).get(Array(0)) == 1F * sampler.indices(i))
    assert(labelMinibatch(0) == sampler.indices(i))
  }

  sampler = new MinibatchSampler(minibatches.iterator, minibatches.length, 10)
  for (i <- 0 to 9) {
    var labelMinibatch = sampler.nextLabelMinibatch()
    var imageMinibatch = sampler.nextImageMinibatch()
    assert(imageMinibatch(0).get(Array(0)) == 1F * sampler.indices(i))
    assert(labelMinibatch(0) == sampler.indices(i))
  }

  val r = scala.util.Random
  sampler = new MinibatchSampler(minibatches.iterator, minibatches.length, 100)
  for (i <- 0 to 99) {
    var (imageMinibatch, labelMinibatch) = {
      if (r.nextBoolean) {
        val tempLabelMinibatch = sampler.nextLabelMinibatch()
        val tempImageMinibatch = sampler.nextImageMinibatch()
        (tempImageMinibatch, tempLabelMinibatch)
      } else {
        val tempImageMinibatch = sampler.nextImageMinibatch()
        val tempLabelMinibatch = sampler.nextLabelMinibatch()
        (tempImageMinibatch, tempLabelMinibatch)
      }
    }
    assert(imageMinibatch(0).get(Array(0)) == 1F * sampler.indices(i))
    assert(labelMinibatch(0) == sampler.indices(i))
  }
}
