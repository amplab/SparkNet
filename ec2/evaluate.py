# Evaluate a trained model

import caffe

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("index", type=int)
parser.add_argument("net", type=str, help="path to the train_val.prototxt file")
parser.add_argument("prefix", type=str)
parser.add_argument("dir", type=str, help="directory where the model files are stored")
parser.add_argument("logfile", type=str, help="path where the resulting logfile will be stored")
args = parser.parse_args()

caffe_root = '/root/caffe/'

model = args.prefix + str(args.index) + '.caffemodel'

caffe.set_mode_gpu()
caffe.set_device(1)

net = caffe.Net(args.net, model, caffe.TEST)

test_iters = 1000
accuracy_top1 = 0
accuracy_top5 = 0

for it in range(test_iters):
    print "iteration ", it
    net.forward()
    accuracy_top1 += net.blobs['loss1/top-1'].data
    accuracy_top5 += net.blobs['loss1/top-5'].data

accuracy_top1 /= test_iters
accuracy_top5 /= test_iters

print 'Accuracy top-1:', accuracy_top1
print 'Accuracy top-5:', accuracy_top5

with open(args.logfile, "a") as myfile:
    myfile.write(str(args.index) + ',' + str(accuracy_top1) + ',' + str(accuracy_top5) + '\n')
