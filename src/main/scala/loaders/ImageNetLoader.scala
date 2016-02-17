package loaders

import java.net.URI
import java.io._
import java.nio.file._

import scala.collection.mutable._
import scala.collection.JavaConversions._

import com.amazonaws.services.s3._
import com.amazonaws.services.s3.model._
import com.amazonaws.auth.profile.ProfileCredentialsProvider

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast

import org.apache.commons.compress.archivers.ArchiveStreamFactory
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream

import libs._
import preprocessing._

class ImageNetLoader(bucket: String) extends java.io.Serializable {
  // Given a path to a directory containing a number of files, and an
  // optional number of parts, return an RDD with one URI per file on the
  // data path.
  def getFilePathsRDD(sc: SparkContext, path: String, numParts: Option[Int] = None): RDD[URI] = {
    val s3Client = new AmazonS3Client(new ProfileCredentialsProvider())
    val listObjectsRequest = new ListObjectsRequest().withBucketName(bucket).withPrefix(path)
    var filePaths = ArrayBuffer[URI]()
    var objectListing: ObjectListing = null
    do {
      objectListing = s3Client.listObjects(listObjectsRequest)
      for (elt <- objectListing.getObjectSummaries()) {
        filePaths += new URI(elt.getKey())
      }
      listObjectsRequest.setMarker(objectListing.getNextMarker())
    } while (objectListing.isTruncated())
    sc.parallelize(filePaths, numParts.getOrElse(filePaths.length))
  }

  // Load the labels file from S3, which associates the filename of each image with its class.
  def getLabels(labelsPath: String) : Map[String, Int] = {
    val s3Client = new AmazonS3Client(new ProfileCredentialsProvider())
    val labelsFile = s3Client.getObject(new GetObjectRequest(bucket, labelsPath))
    val labelsReader = new BufferedReader(new InputStreamReader(labelsFile.getObjectContent()))
    var labelsMap : Map[String, Int] = Map()
    var line = labelsReader.readLine()
    while (line != null) {
      val Array(path, label) = line.split(" ")
      val filename = Paths.get(path).getFileName().toString()
      labelsMap(filename) = label.toInt
      line = labelsReader.readLine()
    }
    labelsMap
  }

  def loadImagesFromTar(filePathsRDD: RDD[URI], broadcastMap: Broadcast[Map[String, Int]], height: Int = 256, width: Int = 256): RDD[(Array[Byte], Int)] = {
    filePathsRDD.flatMap(
      fileUri => {
        val s3Client = new AmazonS3Client(new ProfileCredentialsProvider())
        val stream = s3Client.getObject(new GetObjectRequest(bucket, fileUri.getPath())).getObjectContent()
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
            if (!decompressedResizedImage.isEmpty) {
              images += ((decompressedResizedImage.get, broadcastMap.value(filename)))
              entry = tarStream.getNextTarEntry()
            }
          }
        }
        images.iterator
      }
    )
  }

  // Loads images from dataPath, and creates a new RDD of (imageData,
  // label) pairs; each image is associated with the labels provided in
  // labelPath
  def apply(sc: SparkContext, dataPath: String, labelsPath: String, height: Int = 256, width: Int = 256, numParts: Option[Int] = None): RDD[(Array[Byte], Int)] = {
    val filePathsRDD = getFilePathsRDD(sc, dataPath, numParts)
    val labelsMap = getLabels(labelsPath)
    val broadcastMap = sc.broadcast(labelsMap)
    loadImagesFromTar(filePathsRDD, broadcastMap, height, width)
  }
}
