
// Comment to get more information during initialization
logLevel := Level.Warn

resolvers ++= Seq(
  "Typesafe repository" at "http://repo.typesafe.com/typesafe/releases/", //  The Typesafe repository
  "Sonatype snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/" // The Sonatype snapshots repository
)

// Use the Scalariform plugin to reformat the code
addSbtPlugin("org.scalariform" % "sbt-scalariform" % "1.8.2")

// for publish to Sonatype
addSbtPlugin("org.xerial.sbt" % "sbt-sonatype" % "2.3")

//addSbtPlugin("com.jsuereth" % "sbt-pgp" % "1.1.0")

// for package, visit https://github.com/sbt/sbt-assembly for detail
addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.14.6")
