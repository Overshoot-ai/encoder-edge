import * as ort from "/runtime/ort.mjs";
import {
  INPUT_SHAPE,
  MODEL_BYTES,
  MODEL_CACHE_NAME,
  OUTPUT_SHAPE,
} from "./viewer-config.mjs";

let sessionPromise;

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

async function readCachedModel() {
  try {
    const root = await navigator.storage.getDirectory();
    const handle = await root.getFileHandle(MODEL_CACHE_NAME);
    const file = await handle.getFile();
    if (file.size === MODEL_BYTES) return await file.arrayBuffer();
  } catch (error) {
    if (error.name !== "NotFoundError") console.warn("OPFS model read failed", error);
  }
  return null;
}

async function cacheModel(bytes) {
  try {
    const root = await navigator.storage.getDirectory();
    const handle = await root.getFileHandle(MODEL_CACHE_NAME, { create: true });
    const writable = await handle.createWritable();
    await writable.write(bytes);
    await writable.close();
  } catch (error) {
    console.warn("OPFS model write failed", error);
  }
}

async function loadModel() {
  const started = performance.now();
  const cached = await readCachedModel();
  if (cached) {
    return { bytes: cached, source: "opfs", loadMs: performance.now() - started };
  }

  const response = await fetch("/model.onnx");
  if (!response.ok) throw new Error(`Encoder model: HTTP ${response.status}`);
  const bytes = await response.arrayBuffer();
  if (bytes.byteLength !== MODEL_BYTES) {
    throw new Error(`Encoder model has ${bytes.byteLength} bytes; expected ${MODEL_BYTES}`);
  }
  await cacheModel(bytes);
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

export function getEncoderSession() {
  sessionPromise ||= createSession();
  return sessionPromise;
}

export function resetEncoderSession() {
  sessionPromise = undefined;
}

export async function runEncoder(input) {
  const loaded = await getEncoderSession();
  const tensor = new ort.Tensor("float16", input, INPUT_SHAPE);
  const started = performance.now();
  const outputs = await loaded.session.run({ pixel_values: tensor });
  const inferenceMs = performance.now() - started;
  const output = outputs.image_features || outputs[loaded.session.outputNames[0]];
  const bits = output.data instanceof Uint16Array
    ? output.data
    : new Uint16Array(output.data.buffer, output.data.byteOffset, output.data.byteLength / 2);
  const expectedLength = OUTPUT_SHAPE.reduce((product, size) => product * size, 1);
  if (bits.length !== expectedLength) {
    throw new Error(`Unexpected encoder output length ${bits.length}`);
  }
  return { bits, inferenceMs, diagnostics: loaded.diagnostics };
}
