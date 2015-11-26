import AssemblyKeys._

assemblySettings

// to regenerate the Java protobuf files, run:
// protoc -I=$SPARKNET_HOME/caffe/src/caffe/proto/ --java_out=$SPARKNET_HOME/src/main/scala/protobuf/' $SPARKNET_HOME/caffe/src/caffe/proto/caffe.proto

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.4.1" % "provided"

libraryDependencies += "net.java.dev.jna" % "jna" % "4.2.1"

libraryDependencies += "org.scalatest" % "scalatest_2.10" % "2.0" % "test"

libraryDependencies += "com.amazonaws" % "aws-java-sdk" % "1.10.21"

libraryDependencies += "net.coobird" % "thumbnailator" % "0.4.2"

libraryDependencies ++= Seq("com.twelvemonkeys.imageio" % "imageio" % "3.1.2",
                        "com.twelvemonkeys.imageio" % "imageio-jpeg" % "3.1.2")

libraryDependencies += "com.twelvemonkeys.imageio" % "imageio-metadata" % "3.1.2"
libraryDependencies += "com.twelvemonkeys.imageio" % "imageio-core" % "3.1.2"
libraryDependencies += "com.twelvemonkeys.common" % "common-lang" % "3.1.2"

// the following is needed to make spark more compatible with amazon's aws package
dependencyOverrides ++= Set(
 "com.fasterxml.jackson.core" % "jackson-databind" % "2.4.4"
)

test in assembly := {}
