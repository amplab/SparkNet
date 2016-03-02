# This file is adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/alexnet/alexnet_benchmark.py

from datetime import datetime
import math
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

BATCH_SIZE = 128
IMAGE_SIZE = 227
NUM_CHANNELS = 3
SEED = 66478

def print_activations(t):
  print(t.op.name, ' ', t.get_shape().as_list())


def inference(images):
  # conv1
  with tf.name_scope('conv1') as scope:
    kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
                                             stddev=1e-1, seed=SEED), name='weights')
    conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope)
    print_activations(conv1)

  # lrn1
  # TODO(shlens, jiayq): Add a GPU version of local response normalization.

  # pool1
  pool1 = tf.nn.max_pool(conv1,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool1')
  print_activations(pool1)

  # conv2
  with tf.name_scope('conv2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192],
                                             dtype=tf.float32,
                                             stddev=1e-1,
                                             seed=SEED), name='weights')
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope)
  print_activations(conv2)

  # pool2
  pool2 = tf.nn.max_pool(conv2,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool2')
  print_activations(pool2)

  # conv3
  with tf.name_scope('conv3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                             dtype=tf.float32,
                                             stddev=1e-1,
                                             seed=SEED), name='weights')
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope)
    print_activations(conv3)

  # conv4
  with tf.name_scope('conv4') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1,
                                             seed=SEED), name='weights')
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope)
    print_activations(conv4)

  # conv5
  with tf.name_scope('conv5') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1,
                                             seed=SEED), name='weights')
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope)
    print_activations(conv5)

  # pool5
  pool5 = tf.nn.max_pool(conv5,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool5')
  print_activations(pool5)

  fc6W = tf.Variable(
      tf.truncated_normal([9216, 4096],
                          stddev=0.1,
                          seed=SEED),
                          name="fc6W")
  fc6b = tf.Variable(tf.zeros([4096]), name="fc6b")
  fc6 = tf.nn.relu_layer(tf.reshape(pool5, [BATCH_SIZE, 9216]), fc6W, fc6b, name="fc6")

  fc7W = tf.Variable(
      tf.truncated_normal([4096, 4096],
                          stddev=0.1,
                          seed=SEED),
                          name="fc7W")
  fc7b = tf.Variable(tf.zeros([4096]), name="fc7b")
  fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b, name="fc7")

  fc8W = tf.Variable(
      tf.truncated_normal([4096, 1000],
                          stddev=0.1,
                          seed=SEED),
                          name="fc8W")
  fc8b = tf.Variable(tf.zeros([1000]), name="fc8b")
  fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b, name="fc8")

  return fc8


sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

with tf.device('/gpu:0'):
  # Generate some dummy images.
  # Note that our padding definition is slightly different the cuda-convnet.
  # In order to force the model to start with the same activations sizes,
  # we add 3 to the image_size and employ VALID padding above.
  images = tf.placeholder(
      tf.float32,
      shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
      name="data")
  labels = tf.placeholder(tf.int32, shape=(BATCH_SIZE,), name="label")
  labels = tf.to_int64(labels)

  # Build a Graph that computes the logits predictions from the
  # inference model.
  logits = inference(images)

  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels), name="loss")

  # Use simple momentum for the optimization.
  optimizer = tf.train.MomentumOptimizer(0.01,
                                         0.9).minimize(loss,
                                                       name="train//step")

  # Predictions for the current training minibatch.
  probs = tf.nn.softmax(logits, name="probs")
  prediction = tf.arg_max(probs, 1, name="prediction")
  correct_prediction = tf.equal(prediction, labels, name="correct_prediction")
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")

  # Build an initialization operation.
  tf.initialize_variables(tf.all_variables(), name="init//all_vars")

  # this code traverses the graph and adds Assign nodes for each variable
  variables = [node for node in sess.graph_def.node if node.op == "Variable"]
  for v in variables:
    n = sess.graph.as_graph_element(v.name + ":0")
    dtype = tf.as_dtype(sess.graph.get_operation_by_name(v.name).get_attr("dtype"))
    update_placeholder = tf.placeholder(dtype, n.get_shape().as_list(), name=(v.name + "//update_placeholder"))
    tf.assign(n, update_placeholder, name=(v.name + "//assign"))

  from google.protobuf.text_format import MessageToString
  print MessageToString(sess.graph_def)
  filename = "alexnet_graph.pb"
  s = sess.graph_def.SerializeToString()
  f = open(filename, "wb")
  f.write(s)
