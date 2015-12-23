# Evaluate a trained model

import caffe

caffe_root = '/root/caffe/'

model = '/imagenet2/multigpu/multigpu-0.caffemodel'

caffe.set_mode_gpu()
caffe.set_device(1)

net = caffe.Net(caffe_root + 'models/bvlc_googlenet/train_val.prototxt',
                model,
                caffe.TEST)

test_iters = 1000
accuracy = 0

for it in range(test_iters):
    net.forward()
    accuracy += net.blobs['loss1/top-5'].data

accuracy /= test_iters

print 'Accuracy:', accuracy

import IPython
IPython.embed()
