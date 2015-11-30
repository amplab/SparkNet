// this only works for a 2D image, but a similar class can be written for ND tensors

import java.awt.image.BufferedImage;

class ByteImage {
  byte[] red;
  byte[] green;
  byte[] blue;

  final int width;
  final int height;

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

  // create an "empty" byte image
  public ByteImage(int width, int height) {
    this.width = width;
    this.height = height;
    red = new byte[width * height];
    green = new byte[width * height];
    blue = new byte[width * height];
  }

  void cropInto(float[] buffer, int[] lowerOffsets, int[] upperOffsets) {
    final int w = upperOffsets[0] - lowerOffsets[0];
    final int h = upperOffsets[1] - lowerOffsets[1];
    final int lr = lowerOffsets[0];
    final int lc = lowerOffsets[1];
    for(int row = 0; row < w; row++) {
      for(int col = 0; col < h; col++) {
        buffer[0 * w * h + row * w + col] = red[(row + lr) * width + (col + lc)];
        buffer[1 * w * h + row * w + col] = green[(row + lr) * width + (col + lc)];
        buffer[2 * w * h + row * w + col] = blue[(row + lr) * width + (col + lc)];
      }
    }
  }
}
