// A low level C API for Caffe

#define DTYPE float

extern "C" {
  typedef void (*error_callback_t)(const char *msg);

  error_callback_t global_caffe_error_callback;

  struct caffenet_state;

  // initialize glog
  void init_logging(const char* log_filename, int log_verbosity);

  void set_basepath(const char* path);

  int get_int_size(); // get number of bytes for native int
  int get_dtype_size(); // get number of bytes for DTYPE

  caffenet_state* make_net_from_protobuf(const char* net_param, int net_param_len);
  caffenet_state* make_solver_from_prototxt(const char* solver_file_name);
  void destroy_net(caffenet_state* state);
  void destroy_solver(caffenet_state* state);

  // TODO: Write documentation for these methods (in particular the layer index)
  // Both return an error code: 0 for success, -1 if layer_idx is invalid, -2 if layer is no data layer
  int set_train_data_callback(caffenet_state* state, int layer_idx, java_callback_t callback);
  int set_test_data_callback(caffenet_state* state, int layer_idx, java_callback_t callback);

  void forward(caffenet_state* state);
  void backward(caffenet_state* state);

  int solver_step(caffenet_state* state, int step);
  void solver_test(caffenet_state* state, int num_steps);

  int num_test_scores(caffenet_state* state);
  DTYPE get_test_score(caffenet_state* state, int test_score_layer_idx);

  int num_layers(caffenet_state* state);
  const char* layer_name(caffenet_state* state, int layer_idx);
  int num_layer_weights(caffenet_state* state, int layer_idx);

  int num_data_blobs(caffenet_state* state);
  void* get_data_blob(caffenet_state* state, int blob_idx);
  const char* get_data_blob_name(caffenet_state* state, int blob_idx);
  void* get_weight_blob(caffenet_state* state, int layer_idx, int weight_idx);

  DTYPE* get_data(void* blob); // get the data from a blob
  DTYPE* get_diff(void* blob); // get the gradients from a blob

  int get_num_axes(void* blob);
  int get_axis_shape(void* blob, int axis);

  void set_global_error_callback(error_callback_t c);

  void set_mode_cpu();
  void set_mode_gpu();
  void set_device(int gpu_id);

  void load_weights_from_file(caffenet_state* state, const char* filename);
  void restore_solver_from_file(caffenet_state* state, const char* filename);
}
