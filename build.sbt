import AssemblyKeys._

assemblySettings

// to regenerate the Java protobuf files, run:
// protoc -I=$SPARKNET_HOME/caffe/src/caffe/proto/ --java_out=$SPARKNET_HOME/src/main/scala/protobuf/' $SPARKNET_HOME/caffe/src/caffe/proto/caffe.proto

classpathTypes += "maven-plugin"

resolvers += "javacpp" at "http://www.eecs.berkeley.edu/~rkn/temp/"

// resolvers += "Local Maven Repository" at "file://"+Path.userHome.absolutePath+"/.m2/repository"

libraryDependencies += "org.bytedeco" % "javacpp" % "1.2-SNAPSHOT"

libraryDependencies += "org.bytedeco.javacpp-presets" % "caffe" % "master-1.2-SNAPSHOT"

libraryDependencies += "org.bytedeco.javacpp-presets" % "caffe" % "master-1.2-SNAPSHOT" classifier "linux-x86_64"

libraryDependencies += "org.bytedeco.javacpp-presets" % "opencv" % "3.1.0-1.2-SNAPSHOT"

libraryDependencies += "org.bytedeco.javacpp-presets" % "opencv" % "3.1.0-1.2-SNAPSHOT" classifier "linux-x86_64"

// libraryDependencies += "org.bytedeco" % "javacpp" % "1.1"

// libraryDependencies += "org.bytedeco.javacpp-presets" % "caffe" % "master-1.1"
// libraryDependencies += "org.bytedeco.javacpp-presets" % "caffe" % "master-1.1" classifier "linux-x86"
// libraryDependencies += "org.bytedeco.javacpp-presets" % "caffe" % "master-1.1" classifier "linux-x86_64"
// libraryDependencies += "org.bytedeco.javacpp-presets" % "caffe" % "master-1.1" classifier "macosx-x86_64"
// libraryDependencies += "org.bytedeco.javacpp-presets" % "caffe" % "master-1.1" classifier "windows-x86"
// libraryDependencies += "org.bytedeco.javacpp-presets" % "caffe" % "master-1.1" classifier "windows-x86_64"

// libraryDependencies += "org.bytedeco.javacpp-presets" % "opencv" % "3.0.0-1.1"
// libraryDependencies += "org.bytedeco.javacpp-presets" % "opencv" % "3.0.0-1.1" classifier "linux-x86"
// libraryDependencies += "org.bytedeco.javacpp-presets" % "opencv" % "3.0.0-1.1" classifier "linux-x86_64"
// libraryDependencies += "org.bytedeco.javacpp-presets" % "opencv" % "3.0.0-1.1" classifier "macosx-x86_64"
// libraryDependencies += "org.bytedeco.javacpp-presets" % "opencv" % "3.0.0-1.1" classifier "windows-x86"
// libraryDependencies += "org.bytedeco.javacpp-presets" % "opencv" % "3.0.0-1.1" classifier "windows-x86_64"

libraryDependencies += "com.google.protobuf" % "protobuf-java" % "2.5.0"

libraryDependencies += "org.apache.spark" %% "spark-sql" % "1.4.1" % "provided"

libraryDependencies += "com.databricks" % "spark-csv_2.11" % "1.3.0"

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
