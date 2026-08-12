import * as ort from "/runtime/ort.mjs";
import { createEmbeddingPreview } from "./embedding-preview.mjs";
import {
  floatToHalf,
  INPUT_SHAPE,
  MODEL_BYTES,
  MODEL_CACHE_NAME,
  OUTPUT_SHAPE,
} from "./viewer-shared.mjs";

const FIT_WIDTH = 854;
const FIT_HEIGHT = 480;
const INPUT_WIDTH = 1056;
const INPUT_HEIGHT = 576;
const PATCH_SIZE = 16;

let sessionPromise;

// Model and WebGPU session

async function loadModel() {
  const started = performance.now();
  try {
    const root = await navigator.storage.getDirectory();
    const handle = await root.getFileHandle(MODEL_CACHE_NAME);
    const file = await handle.getFile();
    if (file.size === MODEL_BYTES) {
      return { bytes: await file.arrayBuffer(), source: "opfs", loadMs: performance.now() - started };
    }
  } catch (error) {
    if (error.name !== "NotFoundError") console.warn("OPFS model read failed", error);
  }

  const response = await fetch("/model.onnx");
  if (!response.ok) throw new Error(`Encoder model: HTTP ${response.status}`);
  const bytes = await response.arrayBuffer();
  if (bytes.byteLength !== MODEL_BYTES) {
    throw new Error(`Encoder model has ${bytes.byteLength} bytes; expected ${MODEL_BYTES}`);
  }

  try {
    const root = await navigator.storage.getDirectory();
    const handle = await root.getFileHandle(MODEL_CACHE_NAME, { create: true });
    const writable = await handle.createWritable();
    await writable.write(bytes);
    await writable.close();
  } catch (error) {
    console.warn("OPFS model write failed", error);
  }
  return { bytes, source: "network", loadMs: performance.now() - started };
}

function readableProperties(value) {
  const properties = {};
  const keys = new Set([...Object.keys(value), ...Object.getOwnPropertyNames(Object.getPrototypeOf(value))]);
  for (const key of keys) {
    if (key === "constructor") continue;
    try {
      if (["bigint", "boolean", "number", "string"].includes(typeof value[key])) properties[key] = value[key];
    } catch {
      // Some WebIDL getters cannot be inspected.
    }
  }
  return properties;
}

async function createSession() {
  if (!navigator.gpu) throw new Error("WebGPU is unavailable in this browser");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("WebGPU did not return an adapter");
  if (!adapter.features.has("shader-f16")) throw new Error("This GPU does not support shader-f16");

  ort.env.logLevel = "warning";
  ort.env.wasm.wasmPaths = "/runtime/";
  ort.env.webgpu.adapter = adapter;

  const model = await loadModel();
  const started = performance.now();
  const session = await ort.InferenceSession.create(model.bytes, {
    executionProviders: ["webgpu"],
    graphOptimizationLevel: "all",
    executionMode: "sequential",
  });
  const device = await ort.env.webgpu.device;
  return {
    session,
    diagnostics: {
      executionProviders: ["webgpu"],
      adapterInfo: adapter.info ? readableProperties(adapter.info) : null,
      adapterFeatures: [...adapter.features].sort(),
      deviceFeatures: [...device.features].sort(),
      isFallbackAdapter: adapter.info?.isFallbackAdapter ?? null,
      modelBytes: model.bytes.byteLength,
      modelSource: model.source,
      modelLoadMs: model.loadMs,
      sessionCreateMs: performance.now() - started,
    },
  };
}

export function prepareEncoder() {
  sessionPromise ||= createSession();
  return sessionPromise;
}

export function resetEncoder() {
  sessionPromise = undefined;
}

// Camera frame preprocessing

function cropToAspectRatio(bitmap) {
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

function preprocess(bitmap) {
  const started = performance.now();
  const crop = cropToAspectRatio(bitmap);
  const fitted = new OffscreenCanvas(FIT_WIDTH, FIT_HEIGHT);
  const fitContext = fitted.getContext("2d", { alpha: false });
  fitContext.imageSmoothingEnabled = true;
  fitContext.imageSmoothingQuality = "high";
  fitContext.drawImage(bitmap, crop.x, crop.y, crop.width, crop.height, 0, 0, FIT_WIDTH, FIT_HEIGHT);
  bitmap.close();

  const canvas = new OffscreenCanvas(INPUT_WIDTH, INPUT_HEIGHT);
  const context = canvas.getContext("2d", { alpha: false, willReadFrequently: true });
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(fitted, 0, 0, INPUT_WIDTH, INPUT_HEIGHT);
  const pixels = context.getImageData(0, 0, INPUT_WIDTH, INPUT_HEIGHT).data;

  const input = new Uint16Array(INPUT_SHAPE.reduce((product, size) => product * size, 1));
  let output = 0;
  for (let patchY = 0; patchY < INPUT_HEIGHT / PATCH_SIZE; patchY += 1) {
    for (let patchX = 0; patchX < INPUT_WIDTH / PATCH_SIZE; patchX += 1) {
      for (let y = 0; y < PATCH_SIZE; y += 1) {
        const row = (patchY * PATCH_SIZE + y) * INPUT_WIDTH;
        for (let x = 0; x < PATCH_SIZE; x += 1) {
          const pixel = (row + patchX * PATCH_SIZE + x) * 4;
          input[output] = floatToHalf(pixels[pixel] / 255);
          input[output + 1] = floatToHalf(pixels[pixel + 1] / 255);
          input[output + 2] = floatToHalf(pixels[pixel + 2] / 255);
          output += 3;
        }
      }
    }
  }
  return { input, preprocessMs: performance.now() - started };
}

// Inference

export async function encodeFrame(bitmap) {
  const loaded = await prepareEncoder();
  const prepared = preprocess(bitmap);
  const tensor = new ort.Tensor("float16", prepared.input, INPUT_SHAPE);
  const started = performance.now();
  const outputs = await loaded.session.run({ pixel_values: tensor });
  const inferenceMs = performance.now() - started;
  const output = outputs.image_features || outputs[loaded.session.outputNames[0]];
  const bits = output.data instanceof Uint16Array
    ? output.data
    : new Uint16Array(output.data.buffer, output.data.byteOffset, output.data.byteLength / 2);
  if (bits.length !== OUTPUT_SHAPE.reduce((product, size) => product * size, 1)) {
    throw new Error(`Unexpected encoder output length ${bits.length}`);
  }

  const previewStarted = performance.now();
  const preview = createEmbeddingPreview(bits);
  return {
    preprocessMs: prepared.preprocessMs,
    inferenceMs,
    previewMs: performance.now() - previewStarted,
    outputShape: OUTPUT_SHAPE,
    diagnostics: loaded.diagnostics,
    preview,
  };
}
