#include "caffe/net.hpp"
#include "caffe/solver.hpp"
#include "caffe/sgd_solvers.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/common.hpp"
#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <unistd.h>

#include <glog/logging.h>

#include "ccaffe.h"

using boost::shared_ptr;
using google::protobuf::Message;

struct caffenet_state {
  caffe::Net<DTYPE> *net; // holds a net, can be used independently of solver and testnet
  caffe::Solver<DTYPE> *solver; // holds a solver
  caffe::Net<DTYPE> *testnet; // reference to the the first (and only) testnet of the solver
  std::vector<DTYPE> *test_score; // scratch space to store test scores
  Message* proto; // scratch space to store protobuf message
  std::vector<char>* buffer; // scratch space to pass serialized protobuf to client
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

caffenet_state* create_state() {
  caffenet_state *state = new caffenet_state();
  state->net = NULL;
  state->testnet = NULL;
  state->solver = NULL;
  state->test_score = new std::vector<DTYPE>();
  state->proto = NULL;
  state->buffer = new std::vector<char>();
  return state;
}

void destroy_state(caffenet_state* state) {
  if (state->solver != NULL) {
    delete state->solver;
  } else if (state->net != NULL) {
    delete state->net;
  }
  if (state->proto != NULL) {
    delete state->proto;
  }
  delete state->test_score;
  delete state->buffer;
  delete state;
}

void load_solver_from_protobuf(caffenet_state* state, const char* solver_param, int solver_param_len) {
  caffe::SolverParameter param;
  param.ParseFromString(std::string(solver_param, solver_param_len));
  state->solver = new caffe::SGDSolver<DTYPE>(param);
  state->net = state->solver->net().get();
  state->testnet = state->solver->test_nets()[0].get();
}

void load_net_from_protobuf(caffenet_state* state, const char* net_param, int net_param_len) {
  caffe::NetParameter param;
  param.ParseFromString(std::string(net_param, net_param_len));
  state->net = new caffe::Net<DTYPE>(param);
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

void save_weights_to_file(caffenet_state* state, const char* filename) {
  caffe::NetParameter net_param;
  state->net->ToProto(&net_param, state->solver->param().snapshot_diff());
  WriteProtoToBinaryFile(net_param, filename);
}

void restore_solver_from_file(caffenet_state* state, const char* filename) {
  state->solver->Restore(filename);
}

template<typename M>
bool parse_prototxt(caffenet_state* state, const char* filename) {
  if(!state->proto) {
    state->proto = new M();
  } else {
    delete state->proto;
    state->proto = new M();
  }
  bool success = caffe::ReadProtoFromTextFile(filename, state->proto);
  int size = state->proto->ByteSize();
  state->buffer->resize(size);
  state->proto->SerializeToArray(&(*state->buffer)[0], size);
  return success;
}

bool parse_net_prototxt(caffenet_state* state, const char* filename) {
  return parse_prototxt<caffe::NetParameter>(state, filename);
}

bool parse_solver_prototxt(caffenet_state* state, const char* filename) {
  return parse_prototxt<caffe::SolverParameter>(state, filename);
}

int get_prototxt_len(caffenet_state* state) {
  return state->buffer->size();
}

char* get_prototxt_data(caffenet_state* state) {
  return &(*state->buffer)[0];
}
