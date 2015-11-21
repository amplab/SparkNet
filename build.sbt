import AssemblyKeys._

assemblySettings

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.4.1" % "provided"

libraryDependencies += "net.java.dev.jna" % "jna" % "4.2.1"

libraryDependencies += "org.scalatest" % "scalatest_2.10" % "2.0" % "test"

libraryDependencies += "com.amazonaws" % "aws-java-sdk" % "1.10.21"

// the following is needed to make spark more compatible with amazon's aws package
dependencyOverrides ++= Set(
 "com.fasterxml.jackson.core" % "jackson-databind" % "2.4.4"
)
