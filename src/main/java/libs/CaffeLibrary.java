package libs;

import com.sun.jna.Callback;
import com.sun.jna.Pointer;
import com.sun.jna.Library;
import com.sun.jna.Native;

public interface CaffeLibrary extends Library {
  CaffeLibrary INSTANCE = (CaffeLibrary)Native.loadLibrary("ccaffe", CaffeLibrary.class);

  // extend this to create a callback that will fill a data layer
  interface java_callback_t extends Callback {
    void invoke(Pointer data, int batch_size, int num_dims, Pointer shape);
  }

  void init_logging(String log_filename, int log_verbosity);
  void set_basepath(String path);

  int get_int_size();
  int get_dtype_size();

  void create_db(Pointer state, String db_name, int name_len);
  void write_to_db(Pointer state, byte[] image, int label, int height, int width, String key_str);
  void commit_db_txn(Pointer state);
  void close_db(Pointer state);
  byte[] image_to_datum(byte[] image, int label, int height, int width);

  void save_mean_image(Pointer state, float[] mean_image, int height, int width, String filename, int filename_len);

  Pointer create_state();
  void destroy_state(Pointer state);

  void load_net_from_protobuf(Pointer state, Pointer net_param, int net_param_len);
  void load_solver_from_protobuf(Pointer state, Pointer solver_param, int solver_param_len);

  int set_train_data_callback(Pointer state, int layer_idx, java_callback_t callback);
  int set_test_data_callback(Pointer state, int layer_idx, java_callback_t callback);

  void forward(Pointer state);
  void backward(Pointer state);

  int solver_step(Pointer state, int step);
  void solver_test(Pointer state, int num_steps);

  int num_test_scores(Pointer state);
  float get_test_score(Pointer state, int test_score_layer_idx);

  int num_layers(Pointer state);
  String layer_name(Pointer state, int layer_idx);
  int num_layer_weights(Pointer state, int layer_idx);

  int num_data_blobs(Pointer state);
  Pointer get_data_blob(Pointer state, int blob_idx);
  String get_data_blob_name(Pointer state, int blob_idx);
  Pointer get_weight_blob(Pointer state, int layer_idx, int weight_idx);

  Pointer get_data(Pointer blob);
  Pointer get_diff(Pointer blob);

  int get_num_axes(Pointer blob);
  int get_axis_shape(Pointer blob, int axis);

  void set_mode_cpu();
  void set_mode_gpu();
  void set_device(int gpu_id);

  void load_weights_from_file(Pointer state, String filename);
  void save_weights_to_file(Pointer state, String filename);
  void restore_solver_from_file(Pointer state, String filename);

  boolean parse_net_prototxt(Pointer state, String filename);
  boolean parse_solver_prototxt(Pointer state, String filename);

  int get_prototxt_len(Pointer state);
  Pointer get_prototxt_data(Pointer state);
}
