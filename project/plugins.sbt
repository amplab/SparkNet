resolvers += Resolver.url("bintray-sbt-plugins", url("http://dl.bintray.com/sbt/sbt-plugin-releases"))(Resolver.ivyStylePatterns)

addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.11.2")

// Eclipse Plugin
//addSbtPlugin("com.typesafe.sbteclipse" % "sbteclipse-plugin" % "2.4.0")

// Idea Plugin
addSbtPlugin("com.github.mpeltonen" % "sbt-idea" % "1.6.0")

// Scala Code Coverage Tool
addSbtPlugin("org.scoverage" % "sbt-scoverage" % "1.1.0")

// create a dependency graph
addSbtPlugin("net.virtual-void" % "sbt-dependency-graph" % "0.7.4")

// plugin for creating distributable Scala packages
addSbtPlugin("org.xerial.sbt" % "sbt-sonatype" % "0.2.1")

