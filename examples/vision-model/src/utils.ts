export function getImageDataFromURL(url: string): Promise<ImageData> {
  return new Promise((resolve, reject) => {
    // Converts img to any, and later `as CanvasImageSource`, otherwise build complains
    const img: any = new Image();
    img.crossOrigin = "anonymous"; // Important for CORS
    img.onload = () => {
      const canvas: HTMLCanvasElement = document.createElement("canvas");
      const ctx: CanvasRenderingContext2D = canvas.getContext("2d")!;
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img as CanvasImageSource, 0, 0);

      const imageData = ctx.getImageData(0, 0, img.width, img.height);
      resolve(imageData);
    };
    img.onerror = () => reject(new Error("Failed to load image"));
    img.src = url;
  });
}

export async function imageURLToBase64(url: string): Promise<string> {
  const imageData: ImageData = await getImageDataFromURL(url);
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  canvas.width = imageData.width;
  canvas.height = imageData.height;

  ctx!.putImageData(imageData, 0, 0);

  return canvas.toDataURL();
}
