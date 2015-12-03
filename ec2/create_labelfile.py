import argparse
import os
import IPython

parser = argparse.ArgumentParser()
parser.add_argument("directory", help="directory where JPEGs are stored", type=str)
parser.add_argument("trainfile", help="train.txt from caffe (can contain more entries than files in directory)", type=str)
parser.add_argument("outfile", help="target location of new train.txt", type=str)
args = parser.parse_args()

labelmap = dict()

trainfile = open(args.trainfile, 'r')
for line in trainfile.readlines():
    (fname, label) = line.split()
    labelmap[fname.upper()] = label # poor man's file name normalization
trainfile.close()

outfile = open(args.outfile, 'w')
for root, dirs, files in os.walk(args.directory):
    for f in files:
        outfile.write(f + " " + labelmap[f.upper()] + "\n")
outfile.close()
