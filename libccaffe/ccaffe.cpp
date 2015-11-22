#include "caffe/net.hpp"
#include "caffe/solver.hpp"
#include "caffe/sgd_solvers.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/common.hpp"
#include <string>
#include <boost/shared_ptr.hpp>
#include <unistd.h>

#include <glog/logging.h>

#include "ccaffe.h"

using boost::shared_ptr;

struct caffenet_state {
  caffe::Net<DTYPE> *net;
  caffe::Net<DTYPE> *testnet;
  caffe::Solver<DTYPE> *solver;
  std::vector<DTYPE> *test_score;
};

void init_logging(const char* log_filename, int log_verbosity) {
  google::SetLogDestination(google::INFO, log_filename);
  google::InitGoogleLogging("scalacaffe");
  google::SetStderrLogging(log_verbosity);
}

void set_basepath(const char* path) {
  chdir(path);
}

int get_int_size() {
  return sizeof(int);
}

int get_dtype_size() {
  return sizeof(DTYPE);
}

caffenet_state* make_net_from_protobuf(const char* net_param, int net_param_len) {
  caffenet_state *state = new caffenet_state();
  caffe::NetParameter param;
  param.ParseFromString(std::string(net_param, net_param_len));
  state->net = new caffe::Net<DTYPE>(param);
  state->test_score = new std::vector<DTYPE>();
  return state;
}

caffenet_state* make_solver_from_prototxt(const char* solver_file_name) {
  caffenet_state *state = new caffenet_state();
  state->solver = new caffe::SGDSolver<DTYPE>(std::string(solver_file_name));
  state->net = state->solver->net().get();
  state->testnet = state->solver->test_nets()[0].get(); // assumes there is only one test net
  state->test_score = new std::vector<DTYPE>();
  return state;
}

void destroy_net(caffenet_state* state) {
  delete state->net;
  delete state->test_score;
  delete state;
}

void destroy_solver(caffenet_state* state) {
  delete state->solver;
  delete state->test_score;
  delete state;
}

int num_layers(caffenet_state* state) {
  return state->net->layers().size();
}

const char* layer_name(caffenet_state* state, int layer_idx) {
  return state->net->layers()[layer_idx]->layer_param().name().c_str();
}

int num_layer_weights(caffenet_state* state, int layer_idx) {
  return state->net->layers()[layer_idx]->blobs().size();
}

void* get_data_blob(caffenet_state* state, int blob_idx) {
  return state->net->blobs()[blob_idx].get();
}

const char* get_data_blob_name(caffenet_state* state, int blob_idx) {
  return state->net->blob_names()[blob_idx].c_str();
}

void* get_weight_blob(caffenet_state* state, int layer_idx, int blob_idx) {
  return state->net->layers()[layer_idx]->blobs()[blob_idx].get();
}

int num_data_blobs(caffenet_state* state) {
  return state->net->blobs().size();
}

int num_output_blobs(caffenet_state* state) {
  return state->net->num_outputs();
}

int num_test_scores(caffenet_state* state) {
  //assert(state->test_score->size() > 0);
  return state->test_score->size();
}

#define TO_BLOB(blob) ((caffe::Blob<DTYPE>*)blob)

DTYPE* get_data(void* blob) {
  return TO_BLOB(blob)->mutable_cpu_data();
}

DTYPE* get_diff(void* blob) {
  return TO_BLOB(blob)->mutable_cpu_diff();
}

int get_num_axes(void* blob) {
  return TO_BLOB(blob)->num_axes();
}

int get_axis_shape(void* blob, int axis) {
  return TO_BLOB(blob)->shape(axis);
}

int set_data_callback(caffenet_state* state, int layer_idx, java_callback_t callback, caffe::Net<DTYPE> *net) {
  if (layer_idx >= net->layers().size()) {
    return -1;
  }
  boost::shared_ptr<caffe::JavaDataLayer<DTYPE> > md_layer =
    boost::dynamic_pointer_cast<caffe::JavaDataLayer<DTYPE> >(net->layers()[layer_idx]);
  if (!md_layer) {
    return -2;
  }
  md_layer->SetCallback(callback);
  return 0;
}

int set_train_data_callback(caffenet_state* state, int layer_idx, java_callback_t callback) {
  return set_data_callback(state, layer_idx, callback, state->net);
}

int set_test_data_callback(caffenet_state* state, int layer_idx, java_callback_t callback) {
  return set_data_callback(state, layer_idx, callback, state->testnet);
}

void forward(caffenet_state* state) {
  int start_ind = 0;
  int end_ind = state->net->layers().size() - 1;
  state->net->ForwardFromTo(start_ind, end_ind);
}

void backward(caffenet_state* state) {
  int start_ind = 0;
  int end_ind = state->net->layers().size() - 1;
  state->net->BackwardFromTo(end_ind, start_ind);
}

int solver_step(caffenet_state* state, int step) {
  state->solver->Step(step);
  return 0; // success
}

void solver_test(caffenet_state* state, int num_steps) {
  state->test_score->clear();
  state->solver->TestAndStoreResult(0, num_steps, state->test_score);
}

DTYPE get_test_score(caffenet_state* state, int accuracy_idx) {
  assert(0 <= accuracy_idx && accuracy_idx < state->test_score->size());
  return (*(state->test_score))[accuracy_idx];
}

void set_global_error_callback(error_callback_t c) {
  global_caffe_error_callback = c;
}

void set_mode_gpu() {
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
}

void set_mode_cpu() {
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
}

void set_device(int gpu_id) {
  caffe::Caffe::SetDevice(gpu_id);
}

void load_weights_from_file(caffenet_state* state, const char* filename) {
  state->net->CopyTrainedLayersFrom(filename);
}

void restore_solver_from_file(caffenet_state* state, const char* filename) {
  state->solver->Restore(filename);
}
