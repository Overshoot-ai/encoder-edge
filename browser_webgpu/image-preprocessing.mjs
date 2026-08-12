import { floatToHalf } from "./fp16.mjs";
import {
  FIT_HEIGHT,
  FIT_WIDTH,
  INPUT_HEIGHT,
  INPUT_SHAPE,
  INPUT_WIDTH,
  PATCH_SIZE,
} from "./viewer-config.mjs";

function centerCrop(bitmap) {
  const targetRatio = FIT_WIDTH / FIT_HEIGHT;
  const sourceRatio = bitmap.width / bitmap.height;
  let x = 0;
  let y = 0;
  let width = bitmap.width;
  let height = bitmap.height;

  if (sourceRatio > targetRatio) {
    width = bitmap.height * targetRatio;
    x = (bitmap.width - width) / 2;
  } else if (sourceRatio < targetRatio) {
    height = bitmap.width / targetRatio;
    y = (bitmap.height - height) / 2;
  }
  return { x, y, width, height };
}

function drawInputPixels(bitmap) {
  const crop = centerCrop(bitmap);
  const fitted = new OffscreenCanvas(FIT_WIDTH, FIT_HEIGHT);
  const fitContext = fitted.getContext("2d", { alpha: false });
  fitContext.imageSmoothingEnabled = true;
  fitContext.imageSmoothingQuality = "high";
  fitContext.drawImage(
    bitmap,
    crop.x,
    crop.y,
    crop.width,
    crop.height,
    0,
    0,
    FIT_WIDTH,
    FIT_HEIGHT,
  );
  bitmap.close();

  const canvas = new OffscreenCanvas(INPUT_WIDTH, INPUT_HEIGHT);
  const context = canvas.getContext("2d", { alpha: false, willReadFrequently: true });
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(fitted, 0, 0, INPUT_WIDTH, INPUT_HEIGHT);
  return context.getImageData(0, 0, INPUT_WIDTH, INPUT_HEIGHT).data;
}

function packPatches(pixels) {
  const output = new Uint16Array(INPUT_SHAPE.reduce((product, size) => product * size, 1));
  const patchRows = INPUT_HEIGHT / PATCH_SIZE;
  const patchColumns = INPUT_WIDTH / PATCH_SIZE;
  let outputIndex = 0;

  for (let patchY = 0; patchY < patchRows; patchY += 1) {
    for (let patchX = 0; patchX < patchColumns; patchX += 1) {
      for (let y = 0; y < PATCH_SIZE; y += 1) {
        const row = (patchY * PATCH_SIZE + y) * INPUT_WIDTH;
        for (let x = 0; x < PATCH_SIZE; x += 1) {
          const pixel = (row + patchX * PATCH_SIZE + x) * 4;
          output[outputIndex] = floatToHalf(pixels[pixel] / 255);
          output[outputIndex + 1] = floatToHalf(pixels[pixel + 1] / 255);
          output[outputIndex + 2] = floatToHalf(pixels[pixel + 2] / 255);
          outputIndex += 3;
        }
      }
    }
  }
  return output;
}

export function preprocessBitmap(bitmap) {
  const started = performance.now();
  const input = packPatches(drawInputPixels(bitmap));
  return { input, preprocessMs: performance.now() - started };
}
