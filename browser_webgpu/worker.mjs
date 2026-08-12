import * as ort from "/node_modules/onnxruntime-web/dist/ort.mjs";

function percentile(values, fraction) {
  const ordered = [...values].sort((a, b) => a - b);
  return ordered[Math.max(0, Math.ceil(ordered.length * fraction) - 1)];
}

function summarize(values) {
  return {
    mean_ms: values.reduce((sum, value) => sum + value, 0) / values.length,
    p50_ms: percentile(values, 0.5),
    p90_ms: percentile(values, 0.9),
    min_ms: Math.min(...values),
    max_ms: Math.max(...values),
    samples_ms: values,
  };
}

function summarizeProfile(events) {
  const programs = {};
  let totalKernelMs = 0;
  for (const event of events) {
    const duration = (event.endTime - event.startTime) / 1_000_000;
    totalKernelMs += duration;
    const name = event.programName || event.kernelType || "unknown";
    const entry = programs[name] || { count: 0, total_ms: 0 };
    entry.count += 1;
    entry.total_ms += duration;
    programs[name] = entry;
  }
  return {
    event_count: events.length,
    total_kernel_ms: totalKernelMs,
    programs,
  };
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
      // Some WebIDL getters can reject inspection even though the object is valid.
    }
  }
  return result;
}

function halfToFloat(value) {
  const sign = (value & 0x8000) << 16;
  let exponent = (value >>> 10) & 0x1f;
  let mantissa = value & 0x03ff;
  let bits;
  if (exponent === 0) {
    if (mantissa === 0) {
      bits = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400) === 0) {
        mantissa <<= 1;
        exponent -= 1;
      }
      bits = sign | ((exponent + 112) << 23) | ((mantissa & 0x03ff) << 13);
    }
  } else if (exponent === 0x1f) {
    bits = sign | 0x7f800000 | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }
  const buffer = new ArrayBuffer(4);
  new Uint32Array(buffer)[0] = bits;
  return new Float32Array(buffer)[0];
}

function compare(reference, candidate) {
  if (candidate.length !== reference.length) {
    throw new Error(`Output length ${candidate.length} != reference ${reference.length}`);
  }
  let dot = 0;
  let referenceSquared = 0;
  let candidateSquared = 0;
  let deltaSquared = 0;
  let sumAbs = 0;
  let maxAbs = 0;
  let finite = true;
  for (let index = 0; index < reference.length; index += 1) {
    const expected = halfToFloat(reference[index]);
    const actual = halfToFloat(candidate[index]);
    const delta = actual - expected;
    finite &&= Number.isFinite(actual);
    dot += expected * actual;
    referenceSquared += expected * expected;
    candidateSquared += actual * actual;
    deltaSquared += delta * delta;
    sumAbs += Math.abs(delta);
    maxAbs = Math.max(maxAbs, Math.abs(delta));
  }
  return {
    finite,
    max_abs: maxAbs,
    mean_abs: sumAbs / reference.length,
    rmse: Math.sqrt(deltaSquared / reference.length),
    relative_l2: Math.sqrt(deltaSquared / referenceSquared),
    cosine_similarity: dot / Math.sqrt(referenceSquared * candidateSquared),
  };
}

function bytesToBase64(bytes) {
  let binary = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
}

const floatBitsBuffer = new ArrayBuffer(4);
const floatBitsValue = new Float32Array(floatBitsBuffer);
const floatBitsInteger = new Uint32Array(floatBitsBuffer);

function floatToHalf(value) {
  floatBitsValue[0] = value;
  const bits = floatBitsInteger[0];
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

function compareBits(reference, candidate) {
  if (reference.length !== candidate.length) {
    throw new Error(`Input length ${candidate.length} != reference ${reference.length}`);
  }
  let equal = 0;
  let maxAbs = 0;
  let deltaSquared = 0;
  let referenceSquared = 0;
  for (let index = 0; index < reference.length; index += 1) {
    equal += reference[index] === candidate[index] ? 1 : 0;
    const expected = halfToFloat(reference[index]);
    const actual = halfToFloat(candidate[index]);
    const delta = actual - expected;
    maxAbs = Math.max(maxAbs, Math.abs(delta));
    deltaSquared += delta * delta;
    referenceSquared += expected * expected;
  }
  return {
    bitwise_equal_fraction: equal / reference.length,
    max_abs: maxAbs,
    relative_l2: Math.sqrt(deltaSquared / referenceSquared),
  };
}

async function preprocessImage(url, referenceBits, shape) {
  const started = performance.now();
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const blob = await response.blob();
  const decodedStarted = performance.now();
  const bitmap = await createImageBitmap(blob);
  const canvas = new OffscreenCanvas(1056, 576);
  const context = canvas.getContext("2d", { alpha: false, willReadFrequently: true });
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  const decodedMs = performance.now() - decodedStarted;
  const imageData = context.getImageData(0, 0, canvas.width, canvas.height).data;
  const packedStarted = performance.now();
  const output = new Uint16Array(shape.reduce((product, dimension) => product * dimension, 1));
  let outputIndex = 0;
  for (let patchY = 0; patchY < 36; patchY += 1) {
    for (let patchX = 0; patchX < 66; patchX += 1) {
      for (let y = 0; y < 16; y += 1) {
        const row = (patchY * 16 + y) * canvas.width;
        for (let x = 0; x < 16; x += 1) {
          const pixel = (row + patchX * 16 + x) * 4;
          output[outputIndex] = floatToHalf(imageData[pixel] / 255);
          output[outputIndex + 1] = floatToHalf(imageData[pixel + 1] / 255);
          output[outputIndex + 2] = floatToHalf(imageData[pixel + 2] / 255);
          outputIndex += 3;
        }
      }
    }
  }
  const packedMs = performance.now() - packedStarted;
  return {
    bits: output,
    metrics: {
      total_ms: performance.now() - started,
      decode_resize_readback_ms: decodedMs,
      pack_fp16_ms: packedMs,
      vs_fixture: compareBits(referenceBits, output),
    },
  };
}

async function fetchBytes(path) {
  const started = performance.now();
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  const bytes = await response.arrayBuffer();
  return { bytes, elapsed_ms: performance.now() - started };
}

async function loadModel(path, cacheMode) {
  if (cacheMode === "none") return { ...(await fetchBytes(path)), source: "network" };
  if (cacheMode.startsWith("opfs-")) {
    const started = performance.now();
    const root = await navigator.storage.getDirectory();
    const filename = `model-${path.split("/").at(-1)}`;
    if (cacheMode === "opfs-read") {
      const handle = await root.getFileHandle(filename);
      const file = await handle.getFile();
      return {
        bytes: await file.arrayBuffer(),
        elapsed_ms: performance.now() - started,
        source: "opfs",
      };
    }
    try {
      const existingHandle = await root.getFileHandle(filename);
      const existingFile = await existingHandle.getFile();
      return {
        bytes: await existingFile.arrayBuffer(),
        elapsed_ms: performance.now() - started,
        source: "opfs",
      };
    } catch (error) {
      if (error.name !== "NotFoundError") throw error;
    }
    const response = await fetch(path);
    if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
    const bytes = await response.arrayBuffer();
    const handle = await root.getFileHandle(filename, { create: true });
    const writable = await handle.createWritable();
    await writable.write(bytes);
    await writable.close();
    return {
      bytes,
      elapsed_ms: performance.now() - started,
      source: "network+opfs-write",
    };
  }
  const cacheStarted = performance.now();
  const cache = await caches.open("overshoot-model-v1");
  if (cacheMode === "cache-read") {
    const cached = await cache.match(path);
    if (!cached) throw new Error(`Model cache miss for ${path}`);
    return {
      bytes: await cached.arrayBuffer(),
      elapsed_ms: performance.now() - cacheStarted,
      source: "cache",
    };
  }
  const cached = await cache.match(path);
  if (cached) {
    return {
      bytes: await cached.arrayBuffer(),
      elapsed_ms: performance.now() - cacheStarted,
      source: "cache",
    };
  }
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  await cache.put(path, response.clone());
  return {
    bytes: await response.arrayBuffer(),
    elapsed_ms: performance.now() - cacheStarted,
    source: "network+cache-write",
  };
}

let sharedAdapter = null;

self.onmessage = async ({ data: options }) => {
  const diagnostics = {};
  try {
    if (!self.navigator.gpu) throw new Error("WebGPU is unavailable in this Worker");
    if (!Number.isInteger(options.rounds) || options.rounds < 1) {
      throw new Error("Benchmark rounds must be a positive integer");
    }
    ort.env.logLevel = options.verbose ? "verbose" : "warning";
    ort.env.wasm.wasmPaths = "/node_modules/onnxruntime-web/dist/";

    const adapter = sharedAdapter || await self.navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("WebGPU did not return an adapter");
    sharedAdapter = adapter;
    const adapterInfo = adapter.info ? webIdlRecord(adapter.info) : null;
    const limits = webIdlRecord(adapter.limits);
    const features = [...adapter.features].sort();
    diagnostics.adapter_info = adapterInfo;
    diagnostics.adapter_limits = limits;
    diagnostics.adapter_features = features;
    if (!adapter.features.has("shader-f16")) {
      throw new Error("The selected WebGPU adapter does not expose shader-f16");
    }
    if (ort.env.webgpu.adapter !== adapter) ort.env.webgpu.adapter = adapter;
    const profileEvents = [];
    if (options.profile) {
      ort.env.webgpu.profiling = {
        mode: "default",
        ondata: (event) => profileEvents.push(event),
      };
    }
    let deviceLost = null;

    const [model, fixtureResponse] = await Promise.all([
      loadModel(options.modelUrl, options.cacheMode || "none"),
      fetch(options.fixtureUrl),
    ]);
    if (!fixtureResponse.ok) throw new Error(`Fixture: HTTP ${fixtureResponse.status}`);
    const fixture = await fixtureResponse.json();
    const fixtureUrl = new URL(options.fixtureUrl, self.location.href);
    const [input, reference] = await Promise.all([
      fetchBytes(new URL(fixture.input.path, fixtureUrl).pathname),
      fixture.output
        ? fetchBytes(new URL(fixture.output.path, fixtureUrl).pathname)
        : Promise.resolve(null),
    ]);

    const referenceInputBits = new Uint16Array(input.bytes);
    const preprocessing = options.imageUrl
      ? await preprocessImage(options.imageUrl, referenceInputBits, fixture.input.shape)
      : null;
    const inputBits = preprocessing?.bits || referenceInputBits;
    let session;
    const sessionCreateSamples = [];
    for (let cycle = 0; cycle < options.sessionCycles; cycle += 1) {
      const sessionStarted = performance.now();
      session = await ort.InferenceSession.create(model.bytes, {
        executionProviders: ["webgpu"],
        graphOptimizationLevel: "all",
        executionMode: "sequential",
        enableProfiling: options.profile,
      });
      sessionCreateSamples.push(performance.now() - sessionStarted);
      if (cycle + 1 < options.sessionCycles) await session.release();
    }
    const session_ms = sessionCreateSamples.at(-1);
    const device = await ort.env.webgpu.device;
    diagnostics.device_features = [...device.features].sort();
    device.lost.then((info) => {
      deviceLost = { reason: info.reason, message: info.message };
    });
    const tensor = new ort.Tensor(
      "float16",
      inputBits,
      fixture.input.shape,
    );
    const feeds = { [fixture.input.name]: tensor };

    const coldStarted = performance.now();
    let output = await session.run(feeds);
    const cold_ms = performance.now() - coldStarted;
    const warmLatencies = [];
    for (let index = 0; index < options.rounds; index += 1) {
      const started = performance.now();
      output = await session.run(feeds);
      warmLatencies.push(performance.now() - started);
    }
    if (options.profile) session.endProfiling();

    const outputTensor = output[fixture.output?.name || session.outputNames[0]];
    const outputBits = outputTensor.data instanceof Uint16Array
      ? outputTensor.data
      : new Uint16Array(
        outputTensor.data.buffer,
        outputTensor.data.byteOffset,
        outputTensor.data.byteLength / Uint16Array.BYTES_PER_ELEMENT,
      );
    const result = {
      type: "result",
      user_agent: self.navigator.userAgent,
      adapter_info: adapterInfo,
      adapter_limits: limits,
      adapter_features: features,
      device_features: diagnostics.device_features,
      device_lost: deviceLost,
      webgpu_profile: options.profile ? summarizeProfile(profileEvents) : null,
      model_bytes: model.bytes.byteLength,
      model_fetch_ms: model.elapsed_ms,
      model_source: model.source,
      input_fetch_ms: input.elapsed_ms,
      reference_fetch_ms: reference?.elapsed_ms ?? null,
      preprocessing: preprocessing?.metrics ?? null,
      session_create_ms: session_ms,
      session_create_samples_ms: sessionCreateSamples,
      cold_inference_ms: cold_ms,
      warm_inference: summarize(warmLatencies),
      output: {
        type: outputTensor.type,
        dims: outputTensor.dims,
        ...(reference ? compare(new Uint16Array(reference.bytes), outputBits) : {}),
        ...(options.captureOutput ? {
          f16_base64: bytesToBase64(
            new Uint8Array(
              outputBits.buffer,
              outputBits.byteOffset,
              outputBits.byteLength,
            ),
          ),
        } : {}),
      },
    };
    self.postMessage(result);
  } catch (error) {
    self.postMessage({
      type: "error",
      error: error.stack || String(error),
      diagnostics,
    });
  }
};
