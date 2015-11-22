#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void JavaDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
  const BlobShape& shape = this->layer_param_.java_data_param().shape();
  shape_ = shape;
  batch_size_ = shape.dim(0);
  java_shape_ = new int[shape.dim_size() - 1]; // exclude batch size
  size_ = 1;
  for(int i = 1; i < shape.dim_size(); ++i) { // batch_size is not part of size
    java_shape_[i - 1] = shape.dim(i);
    size_ *= shape.dim(i);
  }
  CHECK_GT(batch_size_ * size_, 0) <<
      "batch_size, channels, height, and width must be specified and"
      " positive in memory_data_param";
  top[0]->Reshape(shape_);
  buffer_ = new Dtype[batch_size_ * size_];
  for(int i = 0; i < batch_size_ * size_; ++i) {
    buffer_[i] = static_cast<Dtype>(0.0);
  }
}

template <typename Dtype>
void JavaDataLayer<Dtype>::SetCallback(java_callback_t callback) {
  java_callback_ = callback;
}

template <typename Dtype>
void JavaDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(java_callback_) << "JavaDataLayer needs to be initalized by calling SetCallback";
  // Get data from scala
  java_callback_(static_cast<void*>(buffer_), batch_size_, shape_.dim_size() - 1, java_shape_);
  top[0]->Reshape(shape_);
  top[0]->set_cpu_data(buffer_);
}

INSTANTIATE_CLASS(JavaDataLayer);
REGISTER_LAYER_CLASS(JavaData);

}  // namespace caffe
