# Script to upload the imagenet dataset to Amazon S3 or another remote file
# system (have to change the function upload_file to support more storage
# systems).

import boto3
import urllib
import tarfile, io
import argparse
import random
import PIL.Image

import collections

parser = argparse.ArgumentParser()
parser.add_argument("s3_bucket", help="Bucket to which imagenet data is uploaded", type=str)
parser.add_argument("--train_tar_file", help="Path to the ILSVRC2012_img_train.tar file", type=str)
parser.add_argument("--val_tar_file", help="Path to the ILSVRC2012_img_val.tar file", type=str)
parser.add_argument("--num_train_chunks", help="Number of train .tar files generated", type=int, default=1000)
parser.add_argument("--num_val_chunks", help="Number of val .tar files generated", type=int, default=50)
parser.add_argument("--new_width", help="Width to resize images to", type=int, default=-1)
parser.add_argument("--new_height", help="Height to resize images to", type=int, default=-1)
args = parser.parse_args()

url = "http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz"
urllib.urlretrieve(url, "caffe_ilsvrc12.tar.gz")
tar = tarfile.open("caffe_ilsvrc12.tar.gz")
train_label_file = tar.extractfile("train.txt")
val_label_file = tar.extractfile("val.txt")

new_image_size = None
if args.new_width != -1 and args.new_height != -1:
    new_image_size = (args.new_width, args.new_height)

s3 = boto3.client('s3')

"""Change this function if you want to upload to HDFS or local storage"""
def upload_file(targetname, stream):
  print "starting to upload", targetname, "to bucket", args.s3_bucket
  s3.put_object(Bucket=args.s3_bucket, Key=targetname, Body=stream)
  print "finished uploading", targetname, "to bucket", args.s3_bucket

def split_label_file(label_file, num_chunks):
    lines = label_file.readlines()
    split_lines = map(lambda s: s.split(), lines)
    random.shuffle(split_lines)
    num_images = len(split_lines)
    shuffled_lists = [[] for _ in range(num_chunks)]
    for i in range(num_images):
        shuffled_lists[i % num_chunks].append(split_lines[i])
    return shuffled_lists

def resize_and_add_image(next_file, file_name, imgfile, new_size=None):
    img = PIL.Image.open(imgfile)
    if new_size is not None:
        img = img.resize(new_size, PIL.Image.ANTIALIAS)
    output = io.BytesIO()
    img.save(output, format='JPEG')
    output.seek(0)
    tarinfo = tarfile.TarInfo(name=file_name)
    tarinfo.size = len(output.getvalue())
    next_file.addfile(tarinfo, fileobj=output)

def process_val_files(val_tar_file, val_label_file, num_chunks):
    val_file = tarfile.open(val_tar_file)
    chunks = split_label_file(val_label_file, num_chunks)
    for i, chunk in enumerate(chunks):
        output = io.BytesIO() # process validation files in memory
        next_file = tarfile.open(mode= "w", fileobj=output)
        for file_name, label in chunk:
            imgfile = val_file.extractfile(file_name)
            resize_and_add_image(next_file, file_name, imgfile, new_size=new_image_size)
        output.seek(0)
        upload_file("ILSVRC2012_img_val/val." + str(i).zfill(3) + ".tar", output)

def build_index(train_tar_file):
    index = dict()
    filehandles = []
    train_file = tarfile.open(train_tar_file)
    for member in train_file.getmembers():
        subtar = tarfile.open(fileobj=train_file.extractfile(member.name))
        filehandles.append(subtar)
        current_member = subtar.next()
        while current_member is not None:
            offset = current_member.offset
            filename = current_member.name
            current_member = subtar.next()
            index[filename] = (subtar, offset)
    return index, filehandles

def process_train_files(train_tar_file, train_label_file, num_chunks):
    chunks = split_label_file(train_label_file, num_chunks)
    index, filehandles = build_index(train_tar_file)
    for i, chunk in enumerate(chunks):
        output = io.BytesIO() # process training files in memory
        next_file = tarfile.open(mode="w", fileobj=output)
        for file_name, label in chunk:
            (folder, img_name) = file_name.split('/')
            (file_handle, offset) = index[img_name]
            file_handle.offset = offset
            imgfile = file_handle.extractfile(file_handle.next())
            resize_and_add_image(next_file, img_name, imgfile, new_size=new_image_size)
        output.seek(0)
        upload_file("ILSVRC2012_img_train/train." + str(i).zfill(5) + ".tar", output)
    for handle in filehandles:
        handle.close()

if __name__ == "__main__":
    upload_file("train.txt", train_label_file.read())
    train_label_file.seek(0) # make it possible to read from this file again
    upload_file("val.txt", val_label_file.read())
    val_label_file.seek(0) # make it possible to read from this file again

    if args.train_tar_file is not None:
        process_train_files(args.train_tar_file, train_label_file, 1000)
    if args.val_tar_file is not None:
        process_val_files(args.val_tar_file, val_label_file, 50)
