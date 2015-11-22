package libs;

import java.util.Arrays;

public class JavaNDUtils {
  // Returns the product of the entries in vs
  public static int arrayProduct(int[] vs) {
    int result = 1;
    for (int i = 0; i < vs.length; i++) {
      result *= vs[i];
    }
    return result;
  }

  // Computes the standard packed array strides for a given shape
  public static final int[] calcDefaultStrides(int[] shape) {
    int dim = shape.length;
    int[] strides = new int[dim];
    int st = 1;
    for (int i = dim - 1; i >= 0; i--) {
      strides[i] = st;
      st *= shape[i];
    }
    return strides;
  }

  // Computes the dot product between two int vectors
  public static int dot(int[] xs, int[] ys) {
    int result = 0;
    assert(xs.length == ys.length);
    for (int i = 0; i < xs.length; i++) {
      result += xs[i] * ys[i];
    }
    return result;
  }

  // Returns a "deep" copy of the argument
  public static final int[] copyOf(int[] data) {
    return Arrays.copyOf(data, data.length);
  }

  // Remove element from position index in data, return deep copy
  public static int[] removeIndex(int[] data, int index) {
    assert(0 <= index);
    assert(index < data.length);
    int len = data.length;
    int[] result = new int[len - 1];
    System.arraycopy(data, 0, result, 0, index);
    System.arraycopy(data, index + 1, result, index, len - index - 1);
    return result;
  }

  // Check if two shapes are the same
  public static boolean shapesEqual(int[] shape1, int[] shape2) {
    if (shape1.length != shape2.length) {
      return false;
    }
    for (int i = 0; i < shape1.length; i++) {
      if (shape1[i] != shape2[i]) {
        return false;
      }
    }
    return true;
  }
}
