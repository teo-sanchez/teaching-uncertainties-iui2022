
import { Image } from 'image-js';

export class ImageUtility {

  public static async fetchImage(url: string): Promise<Image> {
    const image = await Image.load(url);
    return image;
  }

  public static cropCentreImage(
    image: Image,
    opt: {
      width?: number,
      height?: number
  }): Image {
    if (!opt.width) {
      opt.width = image.width;
    }
    if (!opt.height) {
      opt.height = image.height;
    }
    return image.crop({
      x: (image.width - opt.width) / 2,
      y: (image.height - opt.height) / 2,
      width: opt.width,
      height: opt.height
    });
  }

  public static getImageData(image: Image): ImageData {
    const arr = image.getRGBAData({ clamped: true }) as Uint8ClampedArray;
    const imageData = new ImageData(arr, image.width, image.height);
    return imageData;
  }
}
