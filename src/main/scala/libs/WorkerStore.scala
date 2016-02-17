package libs

import scala.collection.mutable.Map

class WorkerStore() {
  val store = Map[String, Any]()

  def get[T](key: String): T = {
    store(key).asInstanceOf[T]
  }

  def put(key: String, value: Any) = {
    store += (key -> value)
  }
}
