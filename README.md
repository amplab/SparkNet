# SparkNet
Distributed Neural Networks for Spark

## Using SparkNet
To run SparkNet, you will need a [Spark](http://spark.apache.org) cluster.
SparkNet apps can be run using [spark-submit](http://spark.apache.org/docs/latest/submitting-applications.html).

Example SparkNet apps are in `src/main/scala/apps`. To run them, you will need to do a couple things.

### Dependencies
For now, you have to install the following.
However, we intend to release an AMI with these dependencies already installed.
Dependencies:

1. sbt 0.13 - [installation instructions](http://www.scala-sbt.org/0.13/tutorial/Installing-sbt-on-Linux.html)
2. cuda
3. protobuf

### Setup

**On EC2:**

1. For each worker node, create one volume (e.g., 100GB) and attach it to the worker (e.g., for instance, at `/dev/sdf`)

**On the master:**

1. Clone the SparkNet repository.
2. Set the `SPARKNET_HOME` environment variable to the SparkNet directory.
3. Build Caffe by running the following:

        cd $SPARKNET_HOME
        mkdir build
        cd build
        cmake ../libccaffe
        make -j 30
4. Increase the Java heap space with `export _JAVA_OPTIONS="-Xmx8g"`.
5. Run `mkdir /tmp/spark-events` (Spark does some logging there).
6. Build SparkNet by doing:

        cd $SPARKNET_HOME
        sbt assembly
**On each worker:**

1. Clone the SparkNet repository.
2. Set the `SPARKNET_HOME` environment variable to the SparkNet directory.
3. Build Caffe as on the master.
4. Run `mount /dev/xvdf /mnt2/spark` to mount the volume you created earlier (assuming you attached the volume at `/dev/sdf`). Spark will spill data to disk here. If everything fits in memory, then this may not be necessary.


### Example Apps
#### Cifar

To run CifarApp, do the following:

1. First get the Cifar data with

        $SPARKNET_HOME/caffe/data/cifar10/get_cifar10.sh
2. Then submit the job with `spark-submit`

        $SPARK_HOME/bin/spark-submit --class apps.CifarApp SparkNetPreview/target/scala-2.10/sparknetpreview-assembly-0.1-SNAPSHOT.jar 5

#### ImageNet
To run ImageNet, do the following:

1.  Obtain the ImageNet data by following the instructions [here](http://www.image-net.org/download-images). This involves creating an account and submitting a request.
2. Put the training tar files on S3 at `s3://sparknet/ILSVRC2012_training`
3. Tar the validation files by running

        TODO
and put them on S3 at `s3://sparknet/ILSVRC2012_val`
4. Submit a job on the master with

        spark-submit --class apps.ImageNetApp $SPARKNET_HOME/target/scala-2.10/sparknet-assembly-0.1-SNAPSHOT.jar n
where `n` is the number of worker nodes in your Spark cluster.

## The SparkNet Architecture
SparkNet is a deep learning library for Spark.
Here we describe a bit of the design.
### Calling Caffe from Java and Scala
We use [Java Native Access](https://github.com/java-native-access/jna) to call C code from Java.
Since Caffe is written in C++, we first create a C wrapper for Caffe in `libccaffe/ccaffe.cpp` and `libccaffe/ccaffe.h`.
We then create a Java interface to the C wrapper in `src/main/java/libs/CaffeLibrary.java`.
This library could be called directly, but the easiest way to use it is through the `CaffeNet` class in `src/main/scala/libs/Net.scala`.

To enable Caffe to read data from Spark RDDs, we define a `JavaDataLayer` in `caffe/include/caffe/data_layers.hpp` and `caffe/src/caffe/layers/java_data_layer.cpp`.

### Defining Models
A model is specified in a `NetParameter` object, and a solver is specified in a `SolverParameter` object.
These can be specified directly in Scala, for example:
```
val netParam = NetParam ("LeNet",
  RDDLayer("data", shape=List(batchsize, 1, 28, 28), None),
  RDDLayer("label", shape=List(batchsize, 1), None),
  ConvolutionLayer("conv1", List("data"), kernel=(5,5), numOutput=20),
  PoolingLayer("pool1", List("conv1"), pooling=Pooling.Max, kernel=(2,2), stride=(2,2)),
  ConvolutionLayer("conv2", List("pool1"), kernel=(5,5), numOutput=50),
  PoolingLayer("pool2", List("conv2"), pooling=Pooling.Max, kernel=(2,2), stride=(2,2)),
  InnerProductLayer("ip1", List("pool2"), numOutput=500),
  ReLULayer("relu1", List("ip1")),
  InnerProductLayer("ip2", List("relu1"), numOutput=10),
  SoftmaxWithLoss("loss", List("ip2", "label"))
)
```
Conveniently, they can be loaded from Caffe prototxt files:

```
val sparkNetHome = sys.env("SPARKNET_HOME")
var netParameter = ProtoLoader.loadNetPrototxt(sparkNetHome + "/caffe/models/bvlc_reference_caffenet/train_val.prototxt")
netParameter = ProtoLoader.replaceDataLayers(netParameter, trainBatchSize, testBatchSize, channels, croppedHeight, croppedWidth)
val solverParameter = ProtoLoader.loadSolverPrototxtWithNet(sparkNetHome + "/caffe/models/bvlc_reference_caffenet/solver.prototxt", netParameter, None)
```
The third line modifies the `NetParameter` object to read data from a `JavaDataLayer`.
A `CaffeNet` object can then be created from a `SolverParameter` object:
```
val net = CaffeNet(solverParameter)
```
