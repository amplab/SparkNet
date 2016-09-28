package libs

import java.io._
import java.util.Date

class Logger(filepath: String) {
  val startTime = System.currentTimeMillis()
  val logfile = new PrintWriter(new File(filepath))

  def log(message: String, i: Int = -1) {
    val elapsedTime = 1F * (System.currentTimeMillis() - startTime) / 1000
    if (i == -1) {
      logfile.write(new Date().toString() + "  " + elapsedTime.toString + ": "  + message + "\n")
    } else {
      logfile.write(new Date().toString()  + "  " + elapsedTime.toString + ", i = " + i.toString + ": "+ message + "\n")

    }
    logfile.flush()
  }
}
