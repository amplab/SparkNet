package libs

import scala.collection.mutable.Map

class WorkerStore() {
  val store = Map[String, Any]()

  def get[T](key: String): T = {
    store(key).asInstanceOf[T]
  }

  def get_[T](key: String): Option[T] = {
    if( store.contains(key))
      Some(store(key).asInstanceOf[T])
    else
      None
  }

  def put(key: String, value: Any) = {
    store += (key -> value)
  }
}
