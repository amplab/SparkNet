package libs;

import java.nio.*;
import static org.bytedeco.javacpp.tensorflow.*;

// This class exists because calling t.createBuffer() directly in Scala seems to
// cause a crash, but it works in Java.
public final class TensorFlowHelper {
  public static FloatBuffer createFloatBuffer(Tensor t) {
    FloatBuffer tFlat = t.createBuffer();
    return tFlat;
  }

  public static IntBuffer createIntBuffer(Tensor t) {
    IntBuffer tFlat = t.createBuffer();
    return tFlat;
  }

  public static ByteBuffer createByteBuffer(Tensor t) {
    ByteBuffer tFlat = t.createBuffer();
    return tFlat;
  }

  public static DoubleBuffer createDoubleBuffer(Tensor t) {
    DoubleBuffer tFlat = t.createBuffer();
    return tFlat;
  }

  public static LongBuffer createLongBuffer(Tensor t) {
    LongBuffer tFlat = t.createBuffer();
    return tFlat;
  }
}
