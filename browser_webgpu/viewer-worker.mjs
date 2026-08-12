import { encodeBitmap, prepareEncoder } from "./encoder-runtime.mjs";
import { resetEncoderSession } from "./webgpu-session.mjs";

let queue = Promise.resolve();

async function handleMessage({ id, type, bitmap }) {
  try {
    if (type === "prepare") {
      self.postMessage({ id, type: "ready", diagnostics: await prepareEncoder() });
      return;
    }
    if (type !== "encode" || !bitmap) throw new Error("Invalid viewer worker request");

    const result = { id, type: "encoded", ...await encodeBitmap(bitmap) };
    const { preview } = result;
    self.postMessage(result, [preview.noveltyPixels, preview.vectorColors, preview.vectorBits]);
  } catch (error) {
    resetEncoderSession();
    self.postMessage({ id, type: "error", error: error.stack || String(error) });
  }
}

self.onmessage = ({ data }) => {
  queue = queue.then(() => handleMessage(data));
};
