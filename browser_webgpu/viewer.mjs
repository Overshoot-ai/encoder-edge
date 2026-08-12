const video = document.querySelector("#camera");
const canvas = document.querySelector("#embedding-canvas");
const context = canvas.getContext("2d");
const status = document.querySelector("#status");
const errorBox = document.querySelector("#error");
const pauseButton = document.querySelector("#pause");
const frameCount = document.querySelector("#frame-count");
const gpuTime = document.querySelector("#gpu-time");
const gpuName = document.querySelector("#gpu-name");
const emptyState = document.querySelector("#empty-state");
const capturedAt = document.querySelector("#captured-at");
const spatialTab = document.querySelector("#spatial-tab");
const vectorsTab = document.querySelector("#vectors-tab");
const legendLow = document.querySelector("#legend-low");
const legendHigh = document.querySelector("#legend-high");
const legendBar = document.querySelector("#legend-bar");
const description = document.querySelector("#description");
const tokenPanel = document.querySelector("#token-panel");
const tokenTitle = document.querySelector("#token-title");
const tokenPosition = document.querySelector("#token-position");
const tokenValues = document.querySelector("#token-values");

const worker = new Worker("/viewer-worker.mjs", { type: "module" });
const pending = new Map();
let requestId = 1;
let stream;
let timer;
let running = true;
let frames = 0;
let view = "spatial";
let preview;
let selectedToken = null;

worker.onmessage = ({ data }) => {
  const request = pending.get(data.id);
  if (!request) return;
  pending.delete(data.id);
  if (data.type === "error") request.reject(new Error(data.error));
  else request.resolve(data);
};
worker.onerror = ({ message }) => fail(new Error(message || "WebGPU worker failed"));

function request(type, bitmap) {
  const id = requestId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    if (bitmap) worker.postMessage({ id, type, bitmap }, [bitmap]);
    else worker.postMessage({ id, type });
  });
}

function setStatus(text, state = "") {
  status.className = `status ${state}`.trim();
  status.querySelector("span").textContent = text;
}

function fail(error) {
  running = false;
  clearTimeout(timer);
  pauseButton.disabled = true;
  setStatus("Encoding stopped");
  errorBox.hidden = false;
  errorBox.textContent = error.message || String(error);
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
  const buffer = new ArrayBuffer(4);
  new Uint32Array(buffer)[0] = bits >>> 0;
  return new Float32Array(buffer)[0];
}

function signedColor(value) {
  const signed = (value - 128) / 127;
  if (signed < 0) {
    const amount = -signed;
    return [9 + 28 * amount, 9 + 90 * amount, 11 + 224 * amount];
  }
  return [9 + 211 * signed, 9 + 29 * signed, 11 + 27 * signed];
}

function draw() {
  if (!preview) return;
  if (view === "spatial") {
    canvas.width = preview.width;
    canvas.height = preview.height;
    const image = context.createImageData(preview.width, preview.height);
    image.data.set(preview.noveltyPixels);
    context.putImageData(image, 0, 0);
    return;
  }
  const gap = 1;
  canvas.width = preview.width * (preview.vectorColumns + gap) - gap;
  canvas.height = preview.height * (preview.vectorRows + gap) - gap;
  const image = context.createImageData(canvas.width, canvas.height);
  for (let token = 0; token < preview.width * preview.height; token += 1) {
    const tokenX = token % preview.width;
    const tokenY = Math.floor(token / preview.width);
    for (let dimension = 0; dimension < 768; dimension += 1) {
      const x = tokenX * 33 + dimension % 32;
      const y = tokenY * 25 + Math.floor(dimension / 32);
      const offset = (y * canvas.width + x) * 4;
      const color = signedColor(preview.vectorColors[token * 768 + dimension]);
      image.data[offset] = color[0];
      image.data[offset + 1] = color[1];
      image.data[offset + 2] = color[2];
      image.data[offset + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
}

function formatValue(value) {
  if (!Number.isFinite(value)) return String(value);
  if (value === 0) return "0";
  if (Math.abs(value) >= 1000 || Math.abs(value) < .001) return value.toExponential(4);
  return value.toPrecision(6);
}

function showToken(token) {
  if (!preview) return;
  selectedToken = token;
  tokenTitle.textContent = `Token ${token}`;
  tokenPosition.textContent = `x=${token % preview.width}, y=${Math.floor(token / preview.width)} / 768 FP16 values`;
  const values = [];
  const offset = token * 768;
  for (let dimension = 0; dimension < 768; dimension += 1) {
    const value = halfToFloat(preview.vectorBits[offset + dimension]);
    const polarity = value < 0 ? "negative" : value > 0 ? "positive" : "";
    values.push(`<div><label>d${dimension}</label><output class="${polarity}">${formatValue(value)}</output></div>`);
  }
  tokenValues.innerHTML = values.join("");
  tokenPanel.hidden = false;
}

function setView(next) {
  view = next;
  const vectors = view === "vectors";
  spatialTab.classList.toggle("active", !vectors);
  vectorsTab.classList.toggle("active", vectors);
  spatialTab.setAttribute("aria-selected", String(!vectors));
  vectorsTab.setAttribute("aria-selected", String(vectors));
  canvas.classList.toggle("vectors", vectors);
  canvas.setAttribute("aria-label", vectors ? "All 264 raw embedding vectors" : "Spatial visualization of local image embeddings");
  legendBar.classList.toggle("vectors", vectors);
  legendLow.textContent = vectors ? "negative" : "similar";
  legendHigh.textContent = vectors ? "positive" : "distinctive";
  description.textContent = vectors
    ? "All 264 embeddings are shown in spatial order. Each mini-matrix is one actual 768-value vector reshaped to 32 x 24. Click one to inspect all FP16 values."
    : "Each tile is one image region. Brighter tiles differ more from the frame-wide average embedding. This shows feature novelty, not a reconstructed image.";
  draw();
}

async function encodeNext() {
  if (!running) return;
  const started = performance.now();
  try {
    const bitmap = await createImageBitmap(video);
    const result = await request("encode", bitmap);
    preview = {
      ...result.preview,
      noveltyPixels: new Uint8ClampedArray(result.preview.noveltyPixels),
      vectorColors: new Uint8Array(result.preview.vectorColors),
      vectorBits: new Uint16Array(result.preview.vectorBits),
    };
    window.__gemmaEmbeddingDiagnostics = { ...result.diagnostics, lastEncode: result };
    frames += 1;
    frameCount.textContent = `${frames} ${frames === 1 ? "frame" : "frames"}`;
    gpuTime.textContent = `${result.inferenceMs.toFixed(0)} ms GPU`;
    capturedAt.textContent = new Date().toLocaleTimeString();
    emptyState.hidden = true;
    draw();
    if (selectedToken !== null) showToken(selectedToken);
    if (running) {
      setStatus("Encoding the latest frame every second", "running");
      timer = setTimeout(encodeNext, Math.max(0, 1000 - (performance.now() - started)));
    }
  } catch (error) {
    fail(error);
  }
}

async function start() {
  try {
    const [media, prepared] = await Promise.all([
      navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      }),
      request("prepare"),
    ]);
    stream = media;
    video.srcObject = stream;
    await video.play();
    if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      await new Promise((resolve) => video.addEventListener("loadeddata", resolve, { once: true }));
    }
    window.__gemmaEmbeddingDiagnostics = prepared.diagnostics;
    gpuName.textContent = prepared.diagnostics.adapterInfo?.vendor === "apple" ? "Apple GPU" : "WebGPU";
    pauseButton.disabled = false;
    setStatus("Encoding the latest frame every second", "running");
    encodeNext();
  } catch (error) {
    fail(error);
  }
}

pauseButton.addEventListener("click", () => {
  running = !running;
  pauseButton.textContent = running ? "Pause" : "Resume";
  if (running) {
    setStatus("Encoding the latest frame every second", "running");
    encodeNext();
  } else {
    clearTimeout(timer);
    setStatus("Encoding paused", "paused");
  }
});
spatialTab.addEventListener("click", () => setView("spatial"));
vectorsTab.addEventListener("click", () => setView("vectors"));
document.querySelector("#close-token").addEventListener("click", () => {
  selectedToken = null;
  tokenPanel.hidden = true;
});
canvas.addEventListener("click", (event) => {
  if (!preview) return;
  const bounds = canvas.getBoundingClientRect();
  const x = (event.clientX - bounds.left) * canvas.width / bounds.width;
  const y = (event.clientY - bounds.top) * canvas.height / bounds.height;
  if (view === "spatial") {
    showToken(Math.min(11, Math.floor(y)) * 22 + Math.min(21, Math.floor(x)));
    return;
  }
  const tokenX = Math.floor(x / 33);
  const tokenY = Math.floor(y / 25);
  if (tokenX < 22 && tokenY < 12 && x % 33 < 32 && y % 25 < 24) showToken(tokenY * 22 + tokenX);
});
window.addEventListener("beforeunload", () => {
  clearTimeout(timer);
  stream?.getTracks().forEach((track) => track.stop());
  worker.terminate();
});

start();
