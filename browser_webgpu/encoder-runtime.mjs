import { createEmbeddingPreview } from "./embedding-preview.mjs";
import { preprocessBitmap } from "./image-preprocessing.mjs";
import { OUTPUT_SHAPE } from "./viewer-config.mjs";
import { getEncoderSession, runEncoder } from "./webgpu-session.mjs";

export async function prepareEncoder() {
  const loaded = await getEncoderSession();
  return loaded.diagnostics;
}

export async function encodeBitmap(bitmap) {
  const prepared = preprocessBitmap(bitmap);
  const encoded = await runEncoder(prepared.input);
  const previewStarted = performance.now();
  const preview = createEmbeddingPreview(encoded.bits);
  return {
    preprocessMs: prepared.preprocessMs,
    inferenceMs: encoded.inferenceMs,
    previewMs: performance.now() - previewStarted,
    outputShape: OUTPUT_SHAPE,
    diagnostics: encoded.diagnostics,
    preview,
  };
}
