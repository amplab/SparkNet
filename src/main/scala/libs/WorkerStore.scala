package libs

import libs._

class WorkerStore() {
  var nets: Map[String, CaffeNet] = Map()

  def setNet(name: String, net: CaffeNet) = {
    nets += (name -> net)
  }

  def getNet(name: String): CaffeNet = {
    return nets(name)
  }
}
