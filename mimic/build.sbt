import sbtassembly.AssemblyKeys.{assemblyJarName, assemblyMergeStrategy}
import sbtassembly.{MergeStrategy, PathList}

lazy val projectName = "big-data-project"
lazy val projectVersion = "1.1.0"
lazy val projectOrganization = "org.sunlab"
lazy val projectScalaVersion = "2.11.12"

lazy val hadoopVersion = "2.7.5"
lazy val sparkVersion = "2.3.0"

lazy val commonSettings = Seq(
  name := projectName,
  version := projectVersion,
  organization := projectOrganization,
  scalaVersion := projectScalaVersion,
  licenses := Seq("MIT" -> url("http://opensource.org/licenses/MIT")),
  homepage := Some(url("https://bitbucket.org/realsunlab/bigdata-bootcamp"))
)


lazy val hadoopDependencies = Seq(
  "org.apache.hadoop" % "hadoop-hdfs" % hadoopVersion,
  "org.apache.hadoop" % "hadoop-aws" % hadoopVersion,
  //  "org.apache.hadoop" % "hadoop-client" % hadoopVersion,
  //  "org.apache.hadoop" % "hadoop-mapreduce" % hadoopVersion,
  //  "org.apache.hadoop" % "hadoop-mapreduce-client-core" % hadoopVersion,
  //  "org.apache.hadoop" % "hadoop-mapreduce-client-common" % hadoopVersion,
  //  "org.apache.hadoop" % "hadoop-annotations" % hadoopVersion,
  //  "org.apache.hadoop" % "hadoop-yarn-api" % hadoopVersion,
  //  "org.apache.hadoop" % "hadoop-yarn-client" % hadoopVersion,
  //  "org.apache.hadoop" % "hadoop-streaming" % hadoopVersion,
  //  "org.apache.hadoop" % "hadoop-distcp" % hadoopVersion,
  "org.apache.hadoop" % "hadoop-common" % hadoopVersion
)

lazy val sparkDependencies = Seq(
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-streaming" % sparkVersion,
  "org.apache.spark" %% "spark-hive" % sparkVersion,
  "org.apache.spark" %% "spark-graphx" % sparkVersion,
  //  "org.apache.spark" %% "spark-network-common" % sparkVersion,
  //  "org.apache.spark" %% "spark-network-shuffle" % sparkVersion,
  //  "org.apache.spark" %% "spark-yarn" % sparkVersion,
  //  "org.apache.spark" %% "spark-launcher" % sparkVersion
  "org.apache.spark" %% "spark-core" % sparkVersion
)


lazy val dependenciesSettings = Seq(
  resolvers ++= Seq(
    "Atlassian Releases" at "https://maven.atlassian.com/public/",
    "scalaz-bintray" at "https://dl.bintray.com/scalaz/releases",
    Resolver.sonatypeRepo("snapshots"),
    Classpaths.typesafeReleases,
    Classpaths.sbtPluginReleases
  ),
  libraryDependencies ++= Seq(
    "ch.qos.logback" % "logback-classic" % "1.1.2", // logger, can be ignored in play framwork
    "com.databricks" %% "spark-csv" % "1.5.0",
    "com.github.fommil.netlib" % "all" % "1.1.2",
    "org.scalatest" %% "scalatest" % "2.2.5" % Test
  ) ++
    hadoopDependencies ++
    sparkDependencies,
  dependencyOverrides ++= Seq(
    "org.scala-lang" % "scala-reflect" % projectScalaVersion,
    "org.scala-lang" % "scala-compiler" % projectScalaVersion,
    "org.scala-lang" % "scala-library" % projectScalaVersion,
    "com.google.code.findbugs" % "jsr305" % "3.0.2",
    "com.univocity" % "univocity-parsers" % "2.5.9",
    "io.netty" % "netty" % "3.9.9.Final",
    "net.java.dev.jets3t" % "jets3t" % "0.9.4",
    "com.jamesmurty.utils" % "java-xmlbuilder" % "1.1",
    "io.netty" % "netty-all" % "4.1.17.Final",
    "commons-net" % "commons-net" % "3.1",
    "com.google.guava" % "guava" % "11.0.2",
    "commons-codec" % "commons-codec" % "1.10"
  ),
  excludeDependencies ++= Seq(
    ExclusionRule(organization = "org.slf4j", name = "slf4j-log4j12") // ignore default logger, use logback instead
  )
)

lazy val assemblySettings = Seq(
  assemblyJarName in assembly := s"$projectName-$projectVersion-$projectScalaVersion.jar",
  assemblyMergeStrategy in assembly := {
    case PathList("javax", "servlet", xs@_*) => MergeStrategy.last
    case PathList("javax", "activation", xs@_*) => MergeStrategy.last
    case PathList("org", "apache", xs@_*) => MergeStrategy.last
    case PathList("com", "google", xs@_*) => MergeStrategy.last
    case PathList("com", "esotericsoftware", xs@_*) => MergeStrategy.last
    case PathList("com", "codahale", xs@_*) => MergeStrategy.last
    case PathList("com", "yammer", xs@_*) => MergeStrategy.last
    case "about.html" => MergeStrategy.rename
    case "META-INF/ECLIPSEF.RSA" => MergeStrategy.discard
    case "META-INF/mailcap" => MergeStrategy.last
    case "META-INF/mimetypes.default" => MergeStrategy.last
    case "META-INF/MANIFEST.MF" => MergeStrategy.discard
    case "plugin.properties" => MergeStrategy.concat
    case "log4j.properties" => MergeStrategy.concat
    case x =>
      // p1
      //val oldStrategy = (assemblyMergeStrategy in assembly).value
      //oldStrategy(x)
      // p2
      // ignore
      if (x.startsWith("META-INF/") &&
        (x.endsWith(".DSA") || x.endsWith(".RSA") || x.endsWith(".SF"))) {
        MergeStrategy.discard
      } else {
        MergeStrategy.last
      }
  }
)



lazy val launchSettings = Seq(
  // set the main class for 'sbt run'
  mainClass in(Compile, run) := Some("edu.gatech.cse6250.main.Main")
)

lazy val root = Project(id = projectName, base = file("."))
  .settings(commonSettings: _*)
  .settings(dependenciesSettings: _*)
  .settings(assemblySettings: _*)
  .settings(launchSettings: _*)

fork := true

parallelExecution in Test := false
