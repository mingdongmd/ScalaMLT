lazy val root = (project in file(".")).
settings(
	name := "scalaMLT",
	version := "0.0.1",
	scalaVersion := "2.12.4"
)

//publishMavenStyle := true
libraryDependencies  ++= Seq(
  "org.scalanlp" %% "breeze" % "0.13.2",
  "org.scalanlp" %% "breeze-natives" % "0.13.2",
  "org.scalanlp" %% "breeze-viz" % "0.13.2"
)

resolvers ++= Seq(
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)
resolvers += DefaultMavenRepository
//resolvers += JavaNet1Repository
resolvers += Resolver.sonatypeRepo("releases")
resolvers += Resolver.typesafeRepo("releases")
resolvers += Resolver.typesafeIvyRepo("releases")
resolvers += Resolver.sbtPluginRepo("releases")
resolvers += Resolver.bintrayRepo("owner", "repo")
resolvers += Resolver.jcenterRepo






