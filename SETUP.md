# Setting up a SparkNet cluster

Here, we describe how you can create your own AMI for a SparkNet cluster. The
instructions were motivated by `https://github.com/amplab/spark-ec2/blob/branch-1.5/create_image.sh`.

1.) Launch a Ubuntu Server 14.04 LTS (HVM) AMI with a g2.8xlarge instance in Amazon EC2.

2.) Install software for development:

```
sudo apt-get update
sudo apt-get install emacs24-nox
sudo apt-get install gcc g++ git openjdk-7-jdk
sudo apt-get install pssh
```

3.) Install prerequisites for our version of caffe:

```
sudo apt-get install libprotobuf-dev protobuf-compiler libhdf5-serial-dev
sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install libgflags-dev libgoogle-glog-dev
```

4.) Install CUDA as explained in `https://github.com/BVLC/caffe/wiki/Install-Caffe-on-EC2-from-scratch-%28Ubuntu,-CUDA-7,-cuDNN%29`

5.) Run

```echo "export JAVA_HOME=/usr/lib/jvm/java-1.7.0-openjdk-amd64" >> ~/.bash_profile```

6.) Clone the SparkNet repo (if you don't want to change the files, put it under /root/) and enter

```
cd SparkNet
mkdir build; cd build
cmake ../libccaffe
make -j32
```

6.) Run the following command to launch the cluster:

```
./spark-ec2 --key-pair=KEYPAIR --identity-file=KEYFILE.pem --region=us-west-2 --zone=us-west-2b --instance-type=g2.8xlarge --ami=ami-9a5e4cfb -s 5 --copy-aws-credentials --spark-version 1.5.0 --spot-price 1.5 --no-ganglia --user-data $SPARKNET_HOME/ec2/cloud-config.txt launch sparknet
```
