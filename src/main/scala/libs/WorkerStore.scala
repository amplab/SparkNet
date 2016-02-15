package libs

import scala.collection.mutable.Map

class WorkerStore() {
  val store = Map[String, Any]()

  def get[T](key: String): T = {
    return store(key).asInstanceOf[T]
  }

  def put(key: String, value: Any) = {
    store += (key -> value)
  }

  var nets: Map[String, CaffeNet] = Map()
  var caffeLib: Option[CaffeLibrary] = None

  def setNet(name: String, net: CaffeNet) = {
    nets += (name -> net)
  }

  def getNet(name: String): CaffeNet = {
    nets(name)
  }

  def setLib(library: CaffeLibrary) = {
    caffeLib = Some(library)
  }

  def getLib(): CaffeLibrary = {
    assert(!caffeLib.isEmpty)
    caffeLib.get
  }
}
