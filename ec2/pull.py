# To distribute the imagenet data among 6 machines say, call this script with
# python pull.py 0 167 /imgnet/train
# python pull.py 167 333 /imgnet/train/
# python pull.py 333 499 /imgnet/train/
# python pull.py 499 665 /imgnet/train/
# python pull.py 665 830 /imgnet/train/
# python pull.py 830 1000 /imgnet/train/

import boto3
import tarfile, io
import argparse
import os

s3 = boto3.client('s3')

parser = argparse.ArgumentParser()
parser.add_argument("start_idx", help="first index of tar file to be pulled", type=int)
parser.add_argument("stop_idx", help="stop index of tar file to be pulled (exclusive)", type=int)
parser.add_argument("directory", help="directory where JPEGs will be stored", type=str)
args = parser.parse_args()

def download_files(directory, tar_path):
    response = s3.get_object(Bucket='sparknet', Key=tar_path)

    output = io.BytesIO()

    chunk = response['Body'].read(1024 * 8)
    while chunk:
        output.write(chunk)
        chunk = response['Body'].read(1024 * 8)

    output.seek(0) # go to the beginning of the .tar file

    tar = tarfile.open(mode= "r", fileobj=output)

    for member in tar.getmembers():
        filename = member.path # in a format like 'n02099601_3085.JPEG'
        content = tar.extractfile(member)
        out = open(os.path.join(directory, filename), 'w')
        out.write(content.read())
        out.close()


directory = os.path.join(args.directory, '%03d-%03d' % (args.start_idx, args.stop_idx))
if not os.path.exists(directory):
    os.makedirs(directory)

for idx in range(args.start_idx, args.stop_idx):
    download_files(directory, 'ILSVRC2012_train/files-shuf-%03d.tar' % idx)
