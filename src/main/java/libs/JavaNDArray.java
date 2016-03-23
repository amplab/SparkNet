package libs;

import java.util.Formatter;

public class JavaNDArray implements java.io.Serializable {
  protected final float[] data;
  protected final int dim;
  protected final int[] shape;
  private final int offset;
  private final int[] strides;

  public JavaNDArray(float[] data, int dim, int[] shape, int offset, int[] strides) {
    // TODO(rkn): check that all of the arguments are consistent with each other
    assert(shape.length == strides.length);
    this.data = data;
    this.dim = dim;
    this.shape = shape;
    this.offset = offset;
    this.strides = strides;
  }

  public JavaNDArray(int... shape) {
    this(new float[JavaNDUtils.arrayProduct(shape)], shape.length, shape, 0, JavaNDUtils.calcDefaultStrides(shape));
  }

  public JavaNDArray(float[] data, int... shape) {
    this(data, shape.length, shape, 0, JavaNDUtils.calcDefaultStrides(shape));
  }

  public int shape(int axis) {
    return shape[axis];
  }

  public JavaNDArray slice(int axis, int index) {
    return new JavaNDArray(data, dim - 1, JavaNDUtils.removeIndex(shape, axis), offset + index * strides[axis], JavaNDUtils.removeIndex(strides, axis));
  }

  public JavaNDArray subArray(int[] lowerOffsets, int[] upperOffsets) {
    int[] newShape = new int[dim];
    for (int i = 0; i < dim; i++) {
      newShape[i] = upperOffsets[i] - lowerOffsets[i];
    }
    return new JavaNDArray(data, dim, JavaNDUtils.copyOf(newShape), offset + JavaNDUtils.dot(lowerOffsets, strides), strides); // todo: why copy shape?
  }

  public void set(int[] indices, float value) {
    int ix = offset;
    assert(indices.length == dim);
    for (int i = 0; i < dim; i++) {
      ix += indices[i] * strides[i];
    }
    data[ix] = value;
  }

  public float get(int... indices) {
    int ix = offset;
    for (int i = 0; i < dim; i++) {
      ix += indices[i] * strides[i];
    }
    return data[ix];
  }

  private int flatIndex = 0;

  private void baseFlatInto(int offset, float[] result) {
    if (strides[dim - 1] == 1) {
      System.arraycopy(data, offset, result, flatIndex, shape[dim - 1]);
      flatIndex += shape[dim - 1];
    } else {
      for (int i = 0; i < shape[dim - 1]; i += 1) {
        result[flatIndex] = data[offset + i * strides[dim - 1]];
        flatIndex += 1;
      }
    }
  }

  private void recursiveFlatInto(int currDim, int offset, float[] result) {
    if (currDim == dim - 1) {
      baseFlatInto(offset, result);
    } else {
      for (int i = 0; i < shape[currDim]; i += 1) {
        recursiveFlatInto(currDim + 1, offset + i * strides[currDim], result);
      }
    }
  }

  public void flatCopy(float[] result) {
    assert(result.length == JavaNDUtils.arrayProduct(shape));
    if (dim == 0) {
      result[0] = data[offset];
    } else {
      flatIndex = 0;
      recursiveFlatInto(0, offset, result);
    }
  }

  public void flatCopySlow(float[] result) {
    assert(result.length == JavaNDUtils.arrayProduct(shape));
    int[] indices = new int[dim];
    int index = 0;
    for (int i = 0; i <= result.length - 2; i++) {
      result[index] = get(indices);
      next(indices);
      index += 1;
    }
    result[index] = get(indices);  // we can only call next result.length - 1 times
  }

  public float[] toFlat() {
    float[] result = new float[JavaNDUtils.arrayProduct(shape)];
    flatCopy(result);
    return result;
  }

  public JavaNDArray flatten() {
    int[] flatShape = {JavaNDUtils.arrayProduct(shape)};
    return new JavaNDArray(data, flatShape.length, flatShape, 0, JavaNDUtils.calcDefaultStrides(flatShape));
  }

  // Note that this buffer may be larger than the apparent size of the
  // JavaByteNDArray. This could happen if the current object came from a
  // subarray or slice call.
  public float[] getBuffer() {
    return data;
  }

  public void add(JavaNDArray that) {
    assert(JavaNDUtils.shapesEqual(shape, that.shape));
    int[] indices = new int[dim];
    int index = 0;
    // the whole method can be optimized when we have the default strides
    for (int i = 0; i <= JavaNDUtils.arrayProduct(shape) - 2; i++) {
      set(indices, get(indices) + that.get(indices)); // this can be made faster
      next(indices);
    }
    set(indices, get(indices) + that.get(indices));
  }

  public void subtract(JavaNDArray that) {
    assert(JavaNDUtils.shapesEqual(shape, that.shape));
    int[] indices = new int[dim];
    int index = 0;
    // the whole method can be optimized when we have the default strides
    for (int i = 0; i <= JavaNDUtils.arrayProduct(shape) - 2; i++) {
      set(indices, get(indices) - that.get(indices)); // this can be made faster
      next(indices);
    }
    set(indices, get(indices) - that.get(indices));
  }

  public void scalarDivide(float v) {
    int[] indices = new int[dim];
    int index = 0;
    // the whole method can be optimized when we have the default strides
    for (int i = 0; i <= JavaNDUtils.arrayProduct(shape) - 2; i++) {
      set(indices, get(indices) / v); // this can be made faster
      next(indices);
    }
    set(indices, get(indices) / v);
  }

  private void next(int[] indices) {
    int axis = dim - 1;
    while (indices[axis] == shape[axis] - 1) {
      indices[axis] = 0;
      axis -= 1;
    }
    indices[axis] += 1;
  }

  public boolean equals(JavaNDArray that, float tol) {
    if (!JavaNDUtils.shapesEqual(shape, that.shape)) {
      return false;
    }
    int[] indices = new int[dim];
    int index = 0;
    // the whole method can be optimized when we have the default strides
    for (int i = 0; i <= JavaNDUtils.arrayProduct(shape) - 2; i++) {
      if (Math.abs(get(indices) - that.get(indices)) > tol) {
        return false;
      }
      next(indices);
    }
    if (Math.abs(get(indices) - that.get(indices)) > tol) {
      return false;
    }
    return true;
  }

  private static void print1DArray(JavaNDArray array, StringBuilder builder) {
    Formatter formatter = new Formatter(builder);
    for(int i = 0; i < array.shape(0); i++) {
      formatter.format("%1.3e ", array.get(i));
    }
  }

  private static void print2DArray(JavaNDArray array, StringBuilder builder) {
    Formatter formatter = new Formatter(builder);
    for (int i = 0; i < array.shape(0); i++) {
      for (int j = 0; j < array.shape(1); j++) {
        formatter.format("%1.3e ", array.get(i, j));
      }
      if (i != array.shape(0) - 1)
        builder.append("\n");
    }
  }

  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append("NDArray of shape ");
    builder.append(shape[0]);
    for(int d = 1; d < dim; d++) {
      builder.append("x");
      builder.append(shape[d]);
    }
    builder.append("\n");
    if (dim == 1) {
      print1DArray(this, builder);
    }
    if (dim == 2) {
      print2DArray(this, builder);
    }
    if (dim == 3) {
      builder.append("\n");
      for(int i = 0; i < shape(0); i++) {
        builder.append("[").append(i).append(", :, :] = \n");
        JavaNDArray s = slice(0, i);
        print2DArray(s, builder);
        if (i != shape(0) - 1)
          builder.append("\n\n");
      }
    }
    if (dim > 3) {
      builder.append("flattened array = \n");
      print1DArray(this.flatten(), builder);
    }
    return builder.toString();
  }
}
