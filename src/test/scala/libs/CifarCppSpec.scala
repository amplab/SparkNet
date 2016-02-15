import org.scalatest._
import org.bytedeco.javacpp.caffe._

class CifarCppSpec extends FlatSpec {
  "CifarCppNet" should "get chance digits right on randomly initialized net" in {
    val iterations = 10
    val sparkNetHome = sys.env("SPARKNET_HOME")
    val model = sparkNetHome + "/caffe/examples/cifar10/cifar10_full_train_test.prototxt"

    val caffeNet = new FloatNet(model, TEST)

    val bottomVec = new FloatBlobVector();
    var totalScore = 0.0
    for (i <- 0 to iterations - 1) {
      val iterLoss = new Array[Float](1);
      val result = caffeNet.Forward(bottomVec, iterLoss);
      for (j <- 0 to result.size().asInstanceOf[Int] - 1) {
        val resultVec = result.get(j).cpu_data();
        for (k <- 0 to result.get(j).count() - 1) {
          val score = resultVec.get(k);
          val outputName = caffeNet.blob_names().get(
            caffeNet.output_blob_indices().get(j)).getString();
          println(outputName, score)
          if (outputName == "accuracy") {
            totalScore += score
          }
        }
      }
    }

    totalScore /= iterations
    assert(0.08 <= totalScore && totalScore <= 0.12)
  }
}
