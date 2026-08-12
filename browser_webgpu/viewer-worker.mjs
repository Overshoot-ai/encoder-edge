import * as ort from "/runtime/ort.mjs";

const INPUT_SHAPE = [1, 2376, 768];
const OUTPUT_SHAPE = [1, 264, 768];
const MODEL_BYTES = 307689273;
const MODEL_CACHE_NAME = "gemma4-e4b-webgpu-a706b29b0d586714.onnx";

let sessionPromise;
let queue = Promise.resolve();

const floatBuffer = new ArrayBuffer(4);
const floatValue = new Float32Array(floatBuffer);
const floatInteger = new Uint32Array(floatBuffer);

function floatToHalf(value) {
  floatValue[0] = value;
  const bits = floatInteger[0];
  const sign = (bits >>> 16) & 0x8000;
  let exponent = ((bits >>> 23) & 0xff) - 127 + 15;
  let mantissa = bits & 0x7fffff;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa = (mantissa | 0x800000) >>> (1 - exponent);
    return sign | ((mantissa + 0x1000) >>> 13);
  }
  if (exponent >= 31) return sign | 0x7c00;
  mantissa += 0x1000;
  if (mantissa & 0x800000) {
    mantissa = 0;
    exponent += 1;
  }
  return sign | (exponent << 10) | (mantissa >>> 13);
}

function halfToFloat(value) {
  const sign = (value & 0x8000) << 16;
  let exponent = (value >>> 10) & 0x1f;
  let mantissa = value & 0x03ff;
  let bits;
  if (exponent === 0) {
    if (mantissa === 0) bits = sign;
    else {
      exponent = 1;
      while ((mantissa & 0x0400) === 0) {
        mantissa <<= 1;
        exponent -= 1;
      }
      bits = sign | ((exponent + 112) << 23) | ((mantissa & 0x03ff) << 13);
    }
  } else if (exponent === 0x1f) bits = sign | 0x7f800000 | (mantissa << 13);
  else bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
  floatInteger[0] = bits >>> 0;
  return floatValue[0];
}

function webIdlRecord(value) {
  const keys = new Set([
    ...Object.keys(value),
    ...Object.getOwnPropertyNames(Object.getPrototypeOf(value)),
  ]);
  const result = {};
  for (const key of keys) {
    if (key === "constructor") continue;
    try {
      if (["bigint", "boolean", "number", "string"].includes(typeof value[key])) {
        result[key] = value[key];
      }
    } catch {
      // Some WebIDL getters reject inspection even when the adapter is valid.
    }
  }
  return result;
}

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

async function createSession() {
  if (!navigator.gpu) throw new Error("WebGPU is unavailable in this browser");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("WebGPU did not return an adapter");
  if (!adapter.features.has("shader-f16")) {
    throw new Error("The selected WebGPU adapter does not support shader-f16");
  }
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
      adapterInfo: adapter.info ? webIdlRecord(adapter.info) : null,
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

function getSession() {
  sessionPromise ||= createSession();
  return sessionPromise;
}

function preprocess(bitmap) {
  const started = performance.now();
  const fitWidth = 854;
  const fitHeight = 480;
  const targetRatio = fitWidth / fitHeight;
  const sourceRatio = bitmap.width / bitmap.height;
  let sx = 0;
  let sy = 0;
  let sw = bitmap.width;
  let sh = bitmap.height;
  if (sourceRatio > targetRatio) {
    sw = bitmap.height * targetRatio;
    sx = (bitmap.width - sw) / 2;
  } else if (sourceRatio < targetRatio) {
    sh = bitmap.width / targetRatio;
    sy = (bitmap.height - sh) / 2;
  }

  const fitted = new OffscreenCanvas(fitWidth, fitHeight);
  const fitContext = fitted.getContext("2d", { alpha: false });
  fitContext.imageSmoothingEnabled = true;
  fitContext.imageSmoothingQuality = "high";
  fitContext.drawImage(bitmap, sx, sy, sw, sh, 0, 0, fitWidth, fitHeight);
  bitmap.close();

  const canvas = new OffscreenCanvas(1056, 576);
  const context = canvas.getContext("2d", { alpha: false, willReadFrequently: true });
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(fitted, 0, 0, canvas.width, canvas.height);
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
  const output = new Uint16Array(INPUT_SHAPE.reduce((product, size) => product * size, 1));
  let outputIndex = 0;
  for (let patchY = 0; patchY < 36; patchY += 1) {
    for (let patchX = 0; patchX < 66; patchX += 1) {
      for (let y = 0; y < 16; y += 1) {
        const row = (patchY * 16 + y) * canvas.width;
        for (let x = 0; x < 16; x += 1) {
          const pixel = (row + patchX * 16 + x) * 4;
          output[outputIndex] = floatToHalf(pixels[pixel] / 255);
          output[outputIndex + 1] = floatToHalf(pixels[pixel + 1] / 255);
          output[outputIndex + 2] = floatToHalf(pixels[pixel + 2] / 255);
          outputIndex += 3;
        }
      }
    }
  }
  return { input: output, preprocessMs: performance.now() - started };
}

function heatColor(value) {
  const stops = [
    [0, [8, 5, 30]],
    [0.25, [65, 18, 93]],
    [0.5, [160, 42, 99]],
    [0.75, [238, 105, 45]],
    [1, [252, 245, 137]],
  ];
  const upper = stops.findIndex(([position]) => value <= position);
  if (upper <= 0) return stops[0][1];
  const [lowPosition, low] = stops[upper - 1];
  const [highPosition, high] = stops[upper];
  const fraction = (value - lowPosition) / (highPosition - lowPosition);
  return low.map((channel, index) => Math.round(channel + (high[index] - channel) * fraction));
}

function visualize(source) {
  const tokenCount = OUTPUT_SHAPE[1];
  const dimensions = OUTPUT_SHAPE[2];
  const mean = new Float32Array(dimensions);
  let globalSquared = 0;
  for (let token = 0; token < tokenCount; token += 1) {
    const offset = token * dimensions;
    for (let dimension = 0; dimension < dimensions; dimension += 1) {
      const value = halfToFloat(source[offset + dimension]);
      mean[dimension] += value;
      globalSquared += value * value;
    }
  }
  for (let dimension = 0; dimension < dimensions; dimension += 1) mean[dimension] /= tokenCount;

  const novelty = new Float32Array(tokenCount);
  const vectorColors = new Uint8Array(tokenCount * dimensions);
  const vectorScale = 2 * Math.sqrt(globalSquared / (tokenCount * dimensions));
  for (let token = 0; token < tokenCount; token += 1) {
    const offset = token * dimensions;
    let squaredDistance = 0;
    for (let dimension = 0; dimension < dimensions; dimension += 1) {
      const value = halfToFloat(source[offset + dimension]);
      const delta = value - mean[dimension];
      squaredDistance += delta * delta;
      vectorColors[offset + dimension] = Math.round(128 + Math.tanh(value / vectorScale) * 127);
    }
    novelty[token] = Math.sqrt(squaredDistance / dimensions);
  }

  const sorted = [...novelty].sort((a, b) => a - b);
  const low = sorted[Math.floor(tokenCount * 0.05)];
  const high = sorted[Math.floor(tokenCount * 0.95)];
  const noveltyPixels = new Uint8ClampedArray(tokenCount * 4);
  for (let token = 0; token < tokenCount; token += 1) {
    const normalized = Math.max(0, Math.min(1, (novelty[token] - low) / (high - low || 1)));
    const color = heatColor(normalized);
    noveltyPixels.set([...color, 255], token * 4);
  }
  return {
    width: 22,
    height: 12,
    vectorColumns: 32,
    vectorRows: 24,
    noveltyPixels: noveltyPixels.buffer,
    vectorColors: vectorColors.buffer,
    vectorBits: new Uint16Array(source).buffer,
  };
}

async function handleMessage(data) {
  const { id, type } = data;
  try {
    const loaded = await getSession();
    if (type === "prepare") {
      self.postMessage({ id, type: "ready", diagnostics: loaded.diagnostics });
      return;
    }
    if (type !== "encode" || !data.bitmap) throw new Error("Invalid viewer worker request");
    const prepared = preprocess(data.bitmap);
    const tensor = new ort.Tensor("float16", prepared.input, INPUT_SHAPE);
    const started = performance.now();
    const outputs = await loaded.session.run({ pixel_values: tensor });
    const inferenceMs = performance.now() - started;
    const output = outputs.image_features || outputs[loaded.session.outputNames[0]];
    const outputBits = output.data instanceof Uint16Array
      ? output.data
      : new Uint16Array(output.data.buffer, output.data.byteOffset, output.data.byteLength / 2);
    if (outputBits.length !== OUTPUT_SHAPE.reduce((product, size) => product * size, 1)) {
      throw new Error(`Unexpected encoder output length ${outputBits.length}`);
    }
    const previewStarted = performance.now();
    const preview = visualize(outputBits);
    const result = {
      id,
      type: "encoded",
      preprocessMs: prepared.preprocessMs,
      inferenceMs,
      previewMs: performance.now() - previewStarted,
      outputShape: OUTPUT_SHAPE,
      diagnostics: loaded.diagnostics,
      preview,
    };
    self.postMessage(result, [preview.noveltyPixels, preview.vectorColors, preview.vectorBits]);
  } catch (error) {
    sessionPromise = undefined;
    self.postMessage({ id, type: "error", error: error.stack || String(error) });
  }
}

self.onmessage = ({ data }) => {
  queue = queue.then(() => handleMessage(data));
};
