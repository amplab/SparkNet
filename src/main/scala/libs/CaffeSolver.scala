package libs

import java.util.concurrent._

import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.bytedeco.javacpp.caffe._

trait Solver {
  def step(rowIt: Iterator[Row])
  def iter(iters: Int)
}

class CaffeSolver(val id: Int, solverParam: SolverParameter, schema: StructType, preprocessor: Preprocessor, controlInput: Boolean) {
  val solver: FloatSolver = {
    println("creating solver")
    FloatSolverRegistry.CreateSolver(solverParam)
  }
  val trainNet = new CaffeNet(id, solverParam.net_param, schema, preprocessor, solver.net(), controlInput)
  //val service = Executors.newCachedThreadPool()
  val service = Executors.newSingleThreadExecutor() //figure out optimization later for inputs using queue

  val timeout = 1200
  def step(rowIt: Iterator[Row]) {
    trainNet.forwardBackward(rowIt)
   /* try {
      val future = service.submit(new StepCallable(this, rowIt, 1))
      future.get(timeout, TimeUnit.SECONDS)
    }
    catch {
      case e: TimeoutException => println("Calculation took to long")
      case e: Exception => e.printStackTrace()
    }*/

  }

  class StepCallable(solver: CaffeSolver, rowIt: Iterator[Row], iterations: Int) extends Callable[Boolean] {

    def call(): Boolean = {
      trainNet.forwardBackward(rowIt)

      //very important...... to check .................
      //solver.ApplyUpdate()

      //solver.solver.Step(iterations)
      true
    }
  }

  def iter(iters: Int): Unit = {
    println("id=" + id)
    /*try {
      val future = service.submit(new StepCallable(this, null, iters))
      future.get(timeout, TimeUnit.SECONDS)
    }
    catch {
      case e: TimeoutException => println("Calculation took to long")
      case e: Exception => e.printStackTrace()
    }*/
    solver.Step(iters)

  }

  def close() = {
    try {
      println("closing solver")
      //close lmdb layer need c++ support in caffe -- reopen from last read index
      solver.net().layers().get(0).close()

      solver.close()
    } catch {
      case e: Exception => println("close solver exception ignored")
    }
  }

  def save(filePath: String) = {
    //only do it for gpuId 0
    if (id == 0)
      {
        trainNet.saveWeightsToFile(filePath)
      }
  }




}
