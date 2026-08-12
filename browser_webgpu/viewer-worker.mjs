import { encodeFrame, prepareEncoder, resetEncoder } from "./webgpu-encoder.mjs";

async function handle(message) {
  try {
    const { type, bitmap } = message || {};
    if (type === "prepare") {
      const { diagnostics } = await prepareEncoder();
      self.postMessage({ type: "ready", diagnostics });
      return;
    }
    if (type !== "encode" || !bitmap) throw new Error("Invalid viewer worker request");

    const result = { type: "encoded", ...await encodeFrame(bitmap) };
    const { preview } = result;
    self.postMessage(result, [preview.noveltyPixels, preview.vectorColors, preview.vectorBits]);
  } catch (error) {
    resetEncoder();
    self.postMessage({ type: "error", error: error.stack || String(error) });
  }
}

self.onmessage = ({ data }) => void handle(data);
