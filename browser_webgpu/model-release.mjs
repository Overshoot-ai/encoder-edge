import { join, resolve } from "node:path";

export const MODEL_FILENAME = "gemma4-e4b-web-fp16-fused-rmsnorm-rope-fastgelu-matmulclip.onnx";
export const MODEL_BYTES = 307689273;
export const MODEL_SHA256 = "a706b29b0d586714f09125bb535e937bd5838d11b1c6ff5036862f4242e81b2b";
export const MODEL_RELEASE = "webgpu-encoder-v1";
export const MODEL_URL = `https://github.com/Overshoot-ai/encoder-edge/releases/download/${MODEL_RELEASE}/${MODEL_FILENAME}`;

export function modelPath(workspace) {
  return resolve(
    process.env.VIEWER_MODEL
      || join(workspace, "artifacts/browser-webgpu", MODEL_FILENAME),
  );
}
