package libs

import libs._

class WorkerStore() {
  var nets: Map[String, CaffeNet] = Map()
  var caffeLib: Option[CaffeLibrary] = None

  def setNet(name: String, net: CaffeNet) = {
    nets += (name -> net)
  }

  def getNet(name: String): CaffeNet = {
    return nets(name)
  }

  def setLib(library: CaffeLibrary) = {
    caffeLib = Some(library)
  }

  def getLib(): CaffeLibrary = {
    assert(!caffeLib.isEmpty)
    return caffeLib.get
  }
}
