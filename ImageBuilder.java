import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;
import java.io.IOException;
import java.awt.Color;


class ImageBuilder {
  protected int width, height;
  protected BufferedImage image;

  int width() { return this.width; }
  int height() { return this.height; }

  ImageBuilder(int width, int height) {
    this.width = width;
    this.height = height;

    // Make a new image
    image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
  }

  /// flushes image buffer and creates a new one
  void flush() {
    image = new BufferedImage(this.width, this.height, BufferedImage.TYPE_INT_RGB);
  }

  /// resizes the image
  void resize(int width, int height) {
    this.width = width;
    this.height = height;
    image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
  }

  /// Write a single pixel at (x,y) given 3 integer values
  void WritePixelBuffer(int x, int y, int r, int g, int b) {
    Color c = new Color(r, g, b);
    image.setRGB(x, y, c.getRGB());
  }

  /// Write a single pixel at (x,y) given a size 3 vector
  void WritePixelBuffer(int x, int y, Vec rgb) {
    if(rgb.size() != 3)
      throw new IllegalArgumentException("Vec is not rbg format: " + rgb.size());

    for(int i = 0; i < rgb.size(); ++i) {
      if(rgb.get(i) < 0.0) {
        rgb.set(i, 0.0);
      } else if(rgb.get(i) > 255.0) {
        rgb.set(i, 255.0);
      }
    }

    Color c = new Color((int)rgb.get(0), (int)rgb.get(1), (int)rgb.get(2));
    image.setRGB(x, y, c.getRGB());
  }

  void outputToPNG(String outputFilePath) {
    try {
      // Write the image to a PNG file
      ImageIO.write(image, "png", new File(outputFilePath));
    } catch(IOException e) {
      throw new RuntimeException(e);
    }
  }

}
