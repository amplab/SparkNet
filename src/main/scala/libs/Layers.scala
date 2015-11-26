// This file should be extended and tested more

package libs

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer

import caffe._
import caffe.Caffe._

object RDDLayer {
	def apply(name: String, shape: List[java.lang.Long]) : LayerParameter = {
		val blobShape = BlobShape.newBuilder()
		blobShape.addAllDim(shape.asJava)
		val layerParam = JavaDataParameter.newBuilder()
		layerParam.setShape(blobShape)
		val result = LayerParameter.newBuilder()
		result.setType("JavaData")
		result.setName(name)
		result.addAllTop(List(name).asJava)
		result.setJavaDataParam(layerParam)
		return result.build()
	}
}

object ConvolutionLayer {
	def apply(name: String, bottom: List[String], kernel: Tuple2[Int, Int], numOutput: Int): LayerParameter = {
		val layerParam = ConvolutionParameter.newBuilder()
		layerParam.setKernelH(kernel._1)
		layerParam.setKernelW(kernel._2)
		layerParam.setNumOutput(numOutput)
		val result = LayerParameter.newBuilder()
		result.setType("Convolution")
		result.setName(name)
		result.addAllTop(List(name).asJava)
		result.addAllBottom(bottom.asJava)
		result.setConvolutionParam(layerParam)
		return result.build()
	}
}

object Pooling extends Enumeration {
	type Pooling = Value
	val Max, Ave = Value
}

import Pooling._

object PoolingLayer {
	def apply(name: String, bottom: List[String], pooling: Pooling, kernel: Tuple2[Int, Int], stride: Tuple2[Int,Int]): LayerParameter = {
		val pool = if(pooling == Pooling.Ave) {
			PoolingParameter.PoolMethod.AVE
		} else {
			PoolingParameter.PoolMethod.MAX
		}
		val layerParam = PoolingParameter.newBuilder()
		layerParam.setKernelH(kernel._1)
		layerParam.setKernelW(kernel._2)
		layerParam.setStrideH(stride._1)
		layerParam.setStrideW(stride._2)
		layerParam.setPool(pool)
		val result = LayerParameter.newBuilder()
		result.setType("Pooling")
		result.setName(name)
		result.addAllTop(List(name).asJava)
		result.addAllBottom(bottom.asJava)
		result.setPoolingParam(layerParam)
		return result.build()
	}
}

object InnerProductLayer {
	def apply(name: String, bottom: List[String], numOutput: Int): LayerParameter = {
		val layerParam = InnerProductParameter.newBuilder()
		layerParam.setNumOutput(numOutput)
		val result = LayerParameter.newBuilder()
		result.setType("InnerProduct")
		result.setName(name)
		result.addAllTop(List(name).asJava)
		result.addAllBottom(bottom.asJava)
		result.setInnerProductParam(layerParam)
		return result.build()
	}
}

object ReLULayer {
	def apply(name: String, bottom: List[String]): LayerParameter = {
		val layerParam = ReLUParameter.newBuilder()
		val result = LayerParameter.newBuilder()
		result.setType("ReLU")
		result.setName(name)
		result.addAllTop(List(name).asJava)
		result.addAllBottom(bottom.asJava)
		result.setReluParam(layerParam)
		return result.build()
	}
}

object SoftmaxWithLoss {
	def apply(name: String, bottom: List[String]): LayerParameter = {
		val lossParam = LossParameter.newBuilder()
		val softmaxParam = SoftmaxParameter.newBuilder()
		val result = LayerParameter.newBuilder()
		result.setType("SoftmaxWithLoss")
		result.setName(name)
		result.addAllTop(List(name).asJava)
		result.addAllBottom(bottom.asJava)
		result.setLossParam(lossParam)
		result.setSoftmaxParam(softmaxParam)
		return result.build()
	}
}

object NetParam {
	def apply(name: String, layers: LayerParameter*): NetParameter = {
		val result = NetParameter.newBuilder()
		result.setName(name)
		result.addAllLayer(layers.toList.asJava)
		return result.build()
	}
}
