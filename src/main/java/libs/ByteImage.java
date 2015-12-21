// this only works for a 2D image, but a similar class can be written for ND tensors
package libs;

import java.awt.image.BufferedImage;

public class ByteImage implements java.io.Serializable {
  private byte[] red;
  private byte[] green;
  private byte[] blue;

  private final int height;
  private final int width;

  public byte getRed(int row, int col) {
    return red[row * width + col];
  }

  public byte getGreen(int row, int col) {
    return green[row * width + col];
  }

  public byte getBlue(int row, int col) {
    return blue[row * width + col];
  }

  public int getHeight() {
    return height;
  }

  public int getWidth() {
    return width;
  }

  // create a byte image from a BufferedImage
  public ByteImage(BufferedImage image) {
    width = image.getWidth();
    height = image.getHeight();
    int[] pixels = image.getRGB(0, 0, width, height, null, 0, width);
    red = new byte[width * height];
    green = new byte[width * height];
    blue = new byte[width * height];


    for (int row = 0; row < height; row++) {
      for (int col = 0; col < width; col++) {
        int rgb = pixels[row * width + col];
        red[row * width + col] = (byte)((rgb >> 16) & 0xFF);
        green[row * width + col] = (byte)((rgb >> 8) & 0xFF);
        blue[row * width + col] = (byte)(rgb & 0xFF);
      }
    }
  }

  public ByteImage(byte[] image, int height, int width) {
    assert(image.length == 3 * width * height);
    this.height = height;
    this.width = width;
    red = new byte[width * height];
    green = new byte[width * height];
    blue = new byte[width * height];
    for (int i = 0; i < width * height; i++) {
      red[i] = image[i];
      green[i] = image[i + width * height];
      blue[i] = image[i + 2 * width * height];
    }
  }

  // create an "empty" byte image
  public ByteImage(int height, int width) {
    this.width = width;
    this.height = height;
    red = new byte[width * height];
    green = new byte[width * height];
    blue = new byte[width * height];
  }

  public void copyToBuffer(byte[] buffer) {
    assert(3 * height * width == buffer.length);
    for (int i = 0; i < width * height; i++) {
      buffer[0 * height * width + i] = red[i];
      buffer[1 * height * width + i] = green[i];
      buffer[2 * height * width + i] = blue[i];
    }
  }

  public void cropInto(float[] buffer, int[] lowerOffsets, int[] upperOffsets) {
    assert(0 <= lowerOffsets[0] && lowerOffsets[0] < upperOffsets[0] && upperOffsets[0] <= height);
    assert(0 <= lowerOffsets[1] && lowerOffsets[1] < upperOffsets[1] && upperOffsets[1] <= width);

    final int h = upperOffsets[0] - lowerOffsets[0];
    final int w = upperOffsets[1] - lowerOffsets[1];
    final int lr = lowerOffsets[0];
    final int lc = lowerOffsets[1];

    assert(buffer.length == h * w);

    for(int row = 0; row < w; row++) {
      for(int col = 0; col < h; col++) {
        buffer[0 * w * h + row * w + col] = red[(row + lr) * width + (col + lc)];
        buffer[1 * w * h + row * w + col] = green[(row + lr) * width + (col + lc)];
        buffer[2 * w * h + row * w + col] = blue[(row + lr) * width + (col + lc)];
      }
    }
  }
}
