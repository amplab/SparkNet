Run the tests with (the jna.nosys part makes sure that the JNA version from the .jar is run)

sbt test -Djna.nosys=true

You can run tests selectively using

sbt "test-only ImageNetLoaderSpec"
