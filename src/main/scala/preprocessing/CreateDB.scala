/*
package preprocessing

import com.sun.jna.Pointer

import org.apache.spark.rdd.RDD

import libs._

// dbType should be either "lmdb" or "leveldb"
class CreateDB(caffeLib: CaffeLibrary, dbType: String) {
  val state = caffeLib.create_state()

  def makeDBFromPartition(dataIt: Iterator[(ByteImage, Int)], dbName: String, height: Int, width: Int) = {
    caffeLib.create_db(state, dbName, dbName.length, dbType, dbType.length)
    var counter = 0
    val imBuffer = new Array[Byte](3 * height * width)
    while (dataIt.hasNext) {
      val (image, label) = dataIt.next
      image.copyToBuffer(imBuffer)
      caffeLib.write_to_db(state, imBuffer, label, 3, height, width, counter.toString)
      counter += 1
      if (counter % 1000 == 0) {
        caffeLib.commit_db_txn(state)
        print(counter.toString + " images written to db\n")
      }
    }
    if (counter % 1000 != 0) {
      caffeLib.commit_db_txn(state) // commit any remaining transactions
      print(counter.toString + " images written to db\n")
    }
    caffeLib.close_db(state)
  }

  def makeDBFromMinibatchPartition(minibatchIt: Iterator[(Array[ByteImage], Array[Int])], dbName: String, height: Int, width: Int) = {
    caffeLib.create_db(state, dbName, dbName.length, dbType, dbType.length)
    var counter = 0
    val imBuffer = new Array[Byte](3 * height * width)
    while (minibatchIt.hasNext) {
      val (images, labels) = minibatchIt.next
      var i = 0
      while (i < images.length) {
        images(i).copyToBuffer(imBuffer)
        caffeLib.write_to_db(state, imBuffer, labels(i), 3, height, width, counter.toString)
        counter += 1
        i += 1
      }
      caffeLib.commit_db_txn(state)
      print(counter.toString + " images written to db\n")
    }
    caffeLib.close_db(state)
  }
}
*/
