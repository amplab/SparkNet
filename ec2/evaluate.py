# Evaluate a trained model

import caffe

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("index", type=int)
args = parser.parse_args()

caffe_root = '/root/caffe/'

model = '/imagenet2/multigpu/multigpu-" + args.index + ".caffemodel'

caffe.set_mode_gpu()
caffe.set_device(1)

net = caffe.Net(caffe_root + 'models/bvlc_googlenet/train_val.prototxt',
                model,
                caffe.TEST)

test_iters = 1000
accuracy = 0

for it in range(test_iters):
    print "iteration ", it
    net.forward()
    accuracy += net.blobs['loss1/top-5'].data

accuracy /= test_iters

print 'Accuracy:', accuracy

with open("/imagnet2/multigpu-acc.txt", "a") as myfile:
    myfile.write(str() + ',' + str(accuracy) + '\n')
