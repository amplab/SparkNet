Creating JAR files for JavaCPP
==============================

This document describes how we create the JAR files for JavaCPP. The libraries
are build in Ubuntu 14.04. We build binaries for Caffe and also TensorFlow,
which requires Bazel.

Start an EC2 AMI with Ubuntu 14.04 and run these commands:

```
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install cuda-7-0 -y
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

For the following step to work, we had to do
```
locate DisableStupidWarnings.h
```
which gives the following output:
```
/home/ubuntu/.cache/bazel/_bazel_ubuntu/d557fe27c3b1f8a6b8a21796588f212a/external/eigen_archive/eigen-eigen-73a4995594c6/Eigen/src/Core/util/DisableStupidWarnings.h
```
You should adapt the following paths according to this output:
```
export CPLUS_INCLUDE_PATH="/home/ubuntu/.cache/bazel/_bazel_ubuntu/d557fe27c3b1f8a6b8a21796588f212a/external/eigen_archive/eigen-eigen-73a4995594c6/:$CPLUS_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="/home/ubuntu/.cache/bazel/_bazel_ubuntu/d557fe27c3b1f8a6b8a21796588f212a/external/eigen_archive/:$CPLUS_INCLUDE_PATH"
```

Install the JavaCPP presets:
```
cd ~
git clone https://github.com/pcmoritz/javacpp-presets.git
cd javacpp-presets
bash cppbuild.sh install opencv caffe tensorflow
mvn install --projects=.,opencv,caffe,tensorflow -Djavacpp.platform.dependency=false
```

Creating JAR files for CentOS 6
===============================

These instructions are based on [the javacpp wiki](https://github.com/bytedeco/javacpp-presets/wiki/Build-Environments).

First, install Docker using

```
sudo apt-get install docker.io
```

and run the CentOS 6 container with

```
sudo docker run -it centos:6 /bin/bash
```

Inside the container, run the following commands:

```
yum install git wget cmake emacs
cd ~

wget https://www.softwarecollections.org/en/scls/rhscl/rh-java-common/epel-6-x86_64/download/rhscl-rh-java-common-epel-6-x86_64.noarch.rpm
wget https://www.softwarecollections.org/en/scls/rhscl/maven30/epel-6-x86_64/download/rhscl-maven30-epel-6-x86_64.noarch.rpm
yum install scl-utils *.rpm

cd /etc/yum.repos.d/
wget http://linuxsoft.cern.ch/cern/devtoolset/slc6-devtoolset.repo
rpm --import http://linuxsoft.cern.ch/cern/slc6X/x86_64/RPM-GPG-KEY-cern

yum install devtoolset-2 maven30
scl enable devtoolset-2 maven30 bash

cd ~
git clone https://github.com/bytedeco/javacpp.git
cp javacpp
mvn install
cd ..
```

```
git clone https://github.com/bytedeco/javacpp-presets.git
cd javacpp-presets
```
Change `CPU_ONLY=0` to `CPU_ONLY=1` in the `linux-x86_64` section of `caffe/cppbuild.sh`,
apply the following changes to `opencv/cppbuild.sh`:
```
-download https://github.com/Itseez/opencv/archive/$OPENCV_VERSION.tar.gz opencv-$OPENCV_VERSION.tar.gz
-download https://github.com/Itseez/opencv_contrib/archive/$OPENCV_VERSION.tar.gz opencv_contrib-$OPENCV_VERSION.tar.gz
+wget https://github.com/Itseez/opencv/archive/$OPENCV_VERSION.zip -O opencv-$OPENCV_VERSION.zip
+wget https://github.com/Itseez/opencv_contrib/archive/$OPENCV_VERSION.zip -O opencv_contrib-$OPENCV_VERSION.zip

-tar -xzvf ../opencv-$OPENCV_VERSION.tar.gz
-tar -xzvf ../opencv_contrib-$OPENCV_VERSION.tar.gz
+unzip ../opencv-$OPENCV_VERSION.zip
+unzip ../opencv_contrib-$OPENCV_VERSION.zip
```
and these changes to `caffe/src/main/java/org/bytedeco/javacpp/presets/caffe.java`:
```
-    @Platform(value = {"linux-x86_64", "macosx-x86_64"}, define = {"SHARED_PTR_NAMESPACE boost", "USE_LEVELDB", "USE_LMDB", "USE_OPENCV"}) })
+    @Platform(value = {"linux-x86_64", "macosx-x86_64"}, define = {"SHARED_PTR_NAMESPACE boost", "USE_LEVELDB", "USE_LMDB", "USE_OPENCV", "CPU_ONLY"}) })
```

Then build the presets using:
```
./cppbuild.sh install opencv caffe
mvn install -Djavacpp.platform.dependency=false --projects .,opencv,caffe
```
