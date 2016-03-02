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
