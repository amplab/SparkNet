Creating JAR files for JavaCPP
==============================

This document describes how we create the JAR files for JavaCPP. The libraries
are build in Ubuntu 12.04 so that we are binary compatible across a wide variety
of distributions. We build binaries for Caffe and also TensorFlow, which
requires Bazel.

Start an EC2 AMI with Ubuntu 12.04 and run these commands:

```
wget http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/rpmdeb/cuda-repo-ubuntu1204-7-0-local_7.0-28_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1204-7-0-local_7.0-28_amd64.deb
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install cuda -y
```

Make sure CUDA is found by Caffe and TensorFlow:

```
sudo ln -s /usr/local/cuda-7.0 /usr/local/cuda
```

Install CuDNN: Download `cudnn-7.0-linux-x64-v4.0-rc.tgzcudnn-7.0-linux-x64-v4.0-rc.tgz` from the CuDNN website and run:

```
tar -zxf cudnn-7.0-linux-x64-v4.0-rc.tgz
cd cuda
sudo cp lib64/* /usr/local/cuda/lib64/
sudo cp include/cudnn.h /usr/local/cuda/include/
```

Install some development tools needed in subsequent steps:

```
sudo apt-get install python-pip python-dev build-essential git zip zlib1g-dev cmake gfortran maven
pip install numpy
```

Install and activate the JDK 8:

```
sudo add-apt-repository ppa:openjdk-r/ppa
# When prompted you'll need to press ENTER to continue
sudo apt-get update
sudo apt-get install -y openjdk-8-jdk

sudo update-alternatives --config java
sudo update-alternatives --config javac
```

Install and activate gcc-4.9:

```
sudo apt-get install python-software-properties
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install g++-4.9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 50
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 50
```

Install Bazel (needs JDK 8):
```
cd ~
git clone https://github.com/bazelbuild/bazel.git
cd bazel
git checkout tags/0.1.4
./compile.sh
sudo cp output/bazel /usr/bin
```

Install JavaCPP:
```
cd ~
git clone https://github.com/bytedeco/javacpp.git
cd javacpp
mvn install
```

Install the JavaCPP presets:
```
cd ~
git clone https://github.com/bytedeco/javacpp-presets.git
cd javacpp-presets
bash cppbuild.sh install opencv caffe tensorflow
mvn install --projects=.,opencv,caffe,tensorflow -Djavacpp.platform.dependency=false
```
