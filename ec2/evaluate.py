# Evaluate a trained model

import caffe

caffe_root = '/root/caffe/'

model = '/imagenet2/multigpu/multigpu-0.caffemodel'

caffe.set_mode_gpu()
net = caffe.Net(caffe_root + 'models/bvlc_googlenet/train_val.prototxt',
                model,
                caffe.TEST)

import IPython
IPython.embed()
