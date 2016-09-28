package loaders

import java.net.URI
import java.nio.file._

import org.apache.commons.compress.archivers.ArchiveStreamFactory
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import preprocessing._

import scala.collection.mutable.ArrayBuffer

class ImageNetLoader(bucket: String) extends java.io.Serializable {
  // Given a path to a directory containing a number of files, and an
  // optional number of parts, return an RDD with one URI per file on the
  // data path.
  val FS_PARAM_NAME = "fs.defaultFS"

  def getFilePathsRDD(sc: SparkContext, path: String, numParts: Option[Int] = None): RDD[URI] = {
    println("path=" + path)

    val dirPath = bucket + "/" + path
    val my = sc.wholeTextFiles(dirPath)
    val result = my.map { case (fn, content) => new URI(fn) }
    result
  }

  // Load the labels file from S3, which associates the filename of each image with its class.
  def getLabels(sc: SparkContext, labelsPath: String): Map[String, Int] = {
    println("getLabels=== =============" + labelsPath)

    val labelsRDD = sc.textFile(bucket + "/" + labelsPath)
    val my: RDD[(String, Int)] = labelsRDD.map(line => {
      val Array(path, label) = line.split(" ")
      val filename = Paths.get(path).getFileName().toString()
      filename -> label.toInt
    })

    val myMap = my.collect().toMap

    myMap
  }

  //@transient val conf: Configuration  = new Configuration()
  //@transient val fs:FileSystem  = FileSystem.get(conf)

  def loadImagesFromTar(filePathsRDD: RDD[URI], broadcastMap: Broadcast[Map[String, Int]], height: Int = 256, width: Int = 256): RDD[(Array[Byte], Int)] = {
    filePathsRDD.flatMap(
      fileUri => {
        val conf: Configuration = new Configuration()
        conf.setBoolean("fs.hdfs.impl.disable.cache", true)
        val fs: FileSystem = FileSystem.get(conf)
        val stream = fs.open(new Path(fileUri.getPath()))
        println("stream= " + fileUri.getPath())

        val tarStream = new ArchiveStreamFactory().createArchiveInputStream("tar", stream).asInstanceOf[TarArchiveInputStream]
        var entry = tarStream.getNextTarEntry()
        val images = new ArrayBuffer[(Array[Byte], Int)] // accumulate image and labels data here

        while (entry != null) {
          if (!entry.isDirectory) {
            var offset = 0
            var ret = 0
            val content = new Array[Byte](entry.getSize().toInt)
            while (ret >= 0 && offset != entry.getSize()) {
              ret = tarStream.read(content, offset, content.length - offset)
              if (ret >= 0) {
                offset += ret
              }
            }
            // load the image data
            val filename = Paths.get(entry.getName()).getFileName().toString
            val decompressedResizedImage = ScaleAndConvert.decompressImageAndResize(content, height, width)
            //println("loadtar= " + filename)

            if (!decompressedResizedImage.isEmpty) {
              images += ((decompressedResizedImage.get, broadcastMap.value(filename)))

              entry = tarStream.getNextTarEntry()
            } else {
              println("empty image " + filename + " mapto= " + broadcastMap.value(filename))
            }

          }
        }

        try {
         tarStream.close()
         stream.close()
         fs.close()
        } catch {
          case e: Exception => e.printStackTrace()
        }

        //to close stream??
        images.iterator
      }
    )
  }

  // Loads images from dataPath, and creates a new RDD of (imageData,
  // label) pairs; each image is associated with the labels provided in
  // labelPath
  def apply(sc: SparkContext, dataPath: String, labelsPath: String, height: Int = 256, width: Int = 256, numParts: Option[Int] = None): RDD[(Array[Byte], Int)] = {
    val filePathsRDD = getFilePathsRDD(sc, dataPath, numParts)
    //filePathsRDD.persist(StorageLevel.MEMORY_ONLY)

    val labelsMap = getLabels(sc, labelsPath)
    val broadcastMap = sc.broadcast(labelsMap)
    loadImagesFromTar(filePathsRDD, broadcastMap, height, width)
  }
}
