Creating JAR files for JavaCPP
==============================

This document describes how we create the JAR files for JavaCPP. The actual
libraries are linked in Cent OS 6 so that we are binary compatible across a wide
variety of distributions (as suggested in [the JavaCPP wiki](https://github.com/bytedeco/javacpp-presets/wiki/Build-Environments)). We
build binaries for Caffe and also TensorFlow, which requires Bazel.

Start an EC2 AMI with Ubuntu 14.04 and run these commands:

```
wget http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/rpmdeb/cuda-repo-ubuntu1404-7-0-local_7.0-28_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404-7-0-local_7.0-28_amd64.deb
sudo apt-get update
sudo apt-get upgrade -y
```

Apply the DRM module workaround from [the Caffe wiki](https://github.com/BVLC/caffe/wiki/Caffe-on-EC2-Ubuntu-14.04-Cuda-7) and install CUDA:
```
sudo apt-get install -y linux-image-extra-`uname -r` linux-headers-`uname -r` linux-image-`uname -r`
sudo apt-get install cuda -y
```

Install docker:

```
sudo apt-get install docker.io
```

Launch Cent OS 6 via Docker:

```
sudo docker run -it -v /usr/local/cuda:/usr/local/cuda centos:6 /bin/bash
```

Inside of docker, run the following commands:

Install dev tools:
```
cd /root/
wget https://www.softwarecollections.org/en/scls/rhscl/rh-java-common/epel-6-x86_64/download/rhscl-rh-java-common-epel-6-x86_64.noarch.rpm
wget https://www.softwarecollections.org/en/scls/rhscl/maven30/epel-6-x86_64/download/rhscl-maven30-epel-6-x86_64.noarch.rpm
wget https://www.softwarecollections.org/en/scls/rhscl/python27/epel-6-x86_64/download/rhscl-python27-epel-6-x86_64.noarch.rpm
yum install scl-utils *.rpm
wget https://linuxsoft.cern.ch/cern/devtoolset/slc6-devtoolset.repo --no-check-certificate
rpm --import http://linuxsoft.cern.ch/cern/slc6X/x86_64/RPM-GPG-KEY-cern
yum install devtoolset-2 maven30 python27
scl enable devtoolset-2 maven30 python27 bash
```

Install JDK 8 which is required for building Bazel:
```
yum install git wget unzip cmake
wget --no-cookies --no-check-certificate --header "Cookie: gpw_e24=http%3A%2F%2Fwww.oracle.com%2F; oraclelicense=accept-securebackup-cookie" "http://download.oracle.com/otn-pub/java/jdk/8u72-b15/jdk-8u72-linux-x64.tar.gz"
tar xzf jdk-8u72-linux-x64.tar.gz
cd /opt/jdk1.8.0_72/
alternatives --install /usr/bin/java java /opt/jdk1.8.0_72/bin/java 2
alternatives --install /usr/bin/javac javac /opt/jdk1.8.0_72/bin/javac 2
```

Install Bazel:
```
git clone https://github.com/bazelbuild/bazel.git
cd bazel
git checkout tags/0.1.4
./compile.sh
sudo cp output/bazel /usr/bin
```

Install JavaCPP:
```
git clone https://github.com/bytedeco/javacpp.git
cd javacpp
mvn install
```

Install the JavaCPP presets:
```
git clone https://github.com/bytedeco/javacpp-presets.git
cd javacpp-presets
bash cppbuild.sh install opencv caffe tensorflow
```
