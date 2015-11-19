package org.apache.spark.sparknet;

import java.util.Formatter;

public class JavaNDArray {
	protected final float[] data;
  protected final int dim;
	protected final int[] shape;
  private final int offset;
	private final int[] strides;

	public JavaNDArray(float[] data, int dim, int[] shape, int offset, int[] strides) {
    assert(data.length == JavaNDUtils.arrayProduct(shape));
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

	public JavaNDArray slice(int axis, int index) {
		return new JavaNDArray(data, dim - 1, JavaNDUtils.removeIndex(shape, axis), offset + index * strides[axis], JavaNDUtils.removeIndex(strides, axis));
	}

	public JavaNDArray subArray(int[] lowerOffsets, int[] upperOffsets) {
    int[] newShape = new int[dim];
    for (int i = 0; i < dim; i++) {
      newShape[i] = upperOffsets[i] - lowerOffsets[i];
    }
		return new JavaNDArray(data, dim, JavaNDUtils.copyOf(newShape), offset + JavaNDUtils.dot(lowerOffsets, strides), strides); // todo: why copy shape (shape used to be an argument, but was it necessary even then?)?
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

	/*
	public void flatCopy(float[] result) {
		assert(result.length >= JavaNDUtils.arrayProduct(shape));
		int[] curr = new int[dim];
		int idx = 0;
		while (true) {
			result[idx] = get(curr);
			int i = dim - 1;
			// next index
			while (curr[i] == shape[i] - 1) {
				if(i == 0)
					return;
				i -= 1;
			}
			curr[i] += 1;
			for(int j = dim-1; j > i; j--)
				curr[j] = 0;
			idx += 1;
		}
	}
	*/

	public void flatCopy(float[] result) {
		assert(result.length == JavaNDUtils.arrayProduct(shape));
		int[] indices = new int[dim];
		int index = 0;
		for (int i = 0; i <= result.length - 2; i++) {
			result[index] = get(indices);
			next(indices);
			index += 1;
		}
		result[index] = get(indices);	// we can only call next result.length - 1 times
	}


	public float[] toFlat() {
		float[] result = new float[JavaNDUtils.arrayProduct(shape)];
		flatCopy(result);
		return result;
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
}
