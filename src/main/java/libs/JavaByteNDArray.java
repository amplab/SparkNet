package libs;

import java.util.Formatter;

public class JavaByteNDArray implements java.io.Serializable {
  protected final byte[] data;
  protected final int dim;
  protected final int[] shape;
  private final int offset;
  private final int[] strides;

  public JavaByteNDArray(byte[] data, int dim, int[] shape, int offset, int[] strides) {
    assert(data.length == JavaNDUtils.arrayProduct(shape));
    assert(shape.length == strides.length);
    this.data = data;
    this.dim = dim;
    this.shape = shape;
    this.offset = offset;
    this.strides = strides;
  }

  public JavaByteNDArray(int... shape) {
    this(new byte[JavaNDUtils.arrayProduct(shape)], shape.length, shape, 0, JavaNDUtils.calcDefaultStrides(shape));
  }

  public JavaByteNDArray(byte[] data, int... shape) {
    this(data, shape.length, shape, 0, JavaNDUtils.calcDefaultStrides(shape));
  }

  public JavaByteNDArray slice(int axis, int index) {
    return new JavaByteNDArray(data, dim - 1, JavaNDUtils.removeIndex(shape, axis), offset + index * strides[axis], JavaNDUtils.removeIndex(strides, axis));
  }

  public JavaByteNDArray subArray(int[] lowerOffsets, int[] upperOffsets) {
    int[] newShape = new int[dim];
    for (int i = 0; i < dim; i++) {
      newShape[i] = upperOffsets[i] - lowerOffsets[i];
    }
    return new JavaByteNDArray(data, dim, JavaNDUtils.copyOf(newShape), offset + JavaNDUtils.dot(lowerOffsets, strides), strides); // todo: why copy shape?
  }

  public void set(int[] indices, byte value) {
    int ix = offset;
    assert(indices.length == dim);
    for (int i = 0; i < dim; i++) {
      ix += indices[i] * strides[i];
    }
    data[ix] = value;
  }

  public byte get(int... indices) {
    int ix = offset;
    for (int i = 0; i < dim; i++) {
      ix += indices[i] * strides[i];
    }
    return data[ix];
  }

  public void flatCopy(byte[] result) {
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

  public void flatCopyFloat(float[] result) {
    assert(result.length == JavaNDUtils.arrayProduct(shape));
    int[] indices = new int[dim];
    int index = 0;
    for (int i = 0; i <= result.length - 2; i++) {
      result[index] = (get(indices) & 0xFF); // and with 0xFF to get unsigned value
      next(indices);
      index += 1;
    }
    result[index] = get(indices);  // we can only call next result.length - 1 times
  }

  public byte[] toFlat() {
    byte[] result = new byte[JavaNDUtils.arrayProduct(shape)];
    flatCopy(result);
    return result;
  }

  // Note that this buffer may be larger than the apparent size of the
  // JavaByteNDArray. This could happen if the current object came from a
  // subarray or slice call.
  public byte[] getBuffer() {
    return data;
  }

  public void copyBufferToFloatArray(float[] result) {
    assert(result.length == data.length);
    for (int i = 0; i < result.length; i++) {
      result[i] = (data[i] & 0xFF);
    }
  }

  private void next(int[] indices) {
    int axis = dim - 1;
    while (indices[axis] == shape[axis] - 1) {
      indices[axis] = 0;
      axis -= 1;
    }
    indices[axis] += 1;
  }
}
