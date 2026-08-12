import {
  EMBEDDING_DIMENSIONS,
  halfToFloat,
  TOKEN_COLUMNS,
  TOKEN_COUNT,
  TOKEN_ROWS,
  tokenDisplayPosition,
} from "./viewer-shared.mjs";

const FRAME_INTERVAL_MS = 1000;
const VECTOR_GAP = 1;

// DOM and application state

function find(selector) {
  const element = document.querySelector(selector);
  if (!element) throw new Error(`Missing viewer element: ${selector}`);
  return element;
}

const ui = {
  video: find("#camera"),
  canvas: find("#embedding-canvas"),
  status: find("#status"),
  error: find("#error"),
  pause: find("#pause"),
  frameCount: find("#frame-count"),
  gpuTime: find("#gpu-time"),
  gpuName: find("#gpu-name"),
  emptyState: find("#empty-state"),
  capturedAt: find("#captured-at"),
  spatialTab: find("#spatial-tab"),
  vectorsTab: find("#vectors-tab"),
  legendLow: find("#legend-low"),
  legendHigh: find("#legend-high"),
  legendBar: find("#legend-bar"),
  description: find("#description"),
  tokenPanel: find("#token-panel"),
  tokenTitle: find("#token-title"),
  tokenPosition: find("#token-position"),
  tokenValues: find("#token-values"),
  closeToken: find("#close-token"),
};

const state = {
  disposed: false,
  encoding: false,
  frames: 0,
  preview: null,
  running: true,
  selectedToken: null,
  stream: null,
  timer: null,
  view: "spatial",
};

// Worker requests

const worker = new Worker("/viewer-worker.mjs", { type: "module" });
let pendingRequest = null;

function rejectPending(error) {
  pendingRequest?.reject(error);
  pendingRequest = null;
}

function workerRequest(type, bitmap) {
  if (state.disposed) return Promise.reject(new Error("Encoder worker disposed"));
  if (pendingRequest) return Promise.reject(new Error("The encoder is already processing a request"));
  return new Promise((resolve, reject) => {
    pendingRequest = { resolve, reject };
    if (bitmap) worker.postMessage({ type, bitmap }, [bitmap]);
    else worker.postMessage({ type });
  });
}

worker.onmessage = ({ data }) => {
  const request = pendingRequest;
  if (!request) return;
  pendingRequest = null;
  if (data.type === "error") request.reject(new Error(data.error));
  else request.resolve(data);
};

worker.onerror = ({ message }) => {
  const error = new Error(message || "WebGPU worker failed");
  rejectPending(error);
  stopWithError(error);
};

// Status and controls

function setStatus(text, style = "") {
  ui.status.className = `status ${style}`.trim();
  ui.status.querySelector("span").textContent = text;
}

function showRunningState() {
  ui.pause.textContent = state.running ? "Pause" : "Resume";
  setStatus(
    state.running ? "Encoding the latest frame every second" : "Encoding paused",
    state.running ? "running" : "paused",
  );
}

function stopCamera() {
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
}

function stopWithError(error) {
  if (state.disposed) return;
  state.running = false;
  clearTimeout(state.timer);
  stopCamera();
  rejectPending(error);
  worker.terminate();
  ui.pause.disabled = true;
  ui.error.hidden = false;
  ui.error.textContent = error.message || String(error);
  setStatus("Encoding stopped");
}

// Embedding canvas and token inspector

function decodePreview(raw) {
  return {
    ...raw,
    noveltyPixels: new Uint8ClampedArray(raw.noveltyPixels),
    vectorColors: new Uint8Array(raw.vectorColors),
    vectorBits: new Uint16Array(raw.vectorBits),
  };
}

function signedColor(value) {
  const signed = (value - 128) / 127;
  if (signed < 0) {
    const amount = -signed;
    return [9 + 28 * amount, 9 + 90 * amount, 11 + 224 * amount];
  }
  return [9 + 211 * signed, 9 + 29 * signed, 11 + 27 * signed];
}

function drawSpatial(context) {
  ui.canvas.width = state.preview.width;
  ui.canvas.height = state.preview.height;
  const image = context.createImageData(state.preview.width, state.preview.height);
  for (let token = 0; token < TOKEN_COUNT; token += 1) {
    const { x, y } = tokenDisplayPosition(token);
    image.data.set(state.preview.noveltyPixels.subarray(token * 4, token * 4 + 4), (y * TOKEN_COLUMNS + x) * 4);
  }
  context.putImageData(image, 0, 0);
}

function drawVectors(context) {
  const cellWidth = state.preview.vectorColumns + VECTOR_GAP;
  const cellHeight = state.preview.vectorRows + VECTOR_GAP;
  ui.canvas.width = state.preview.width * cellWidth - VECTOR_GAP;
  ui.canvas.height = state.preview.height * cellHeight - VECTOR_GAP;
  const image = context.createImageData(ui.canvas.width, ui.canvas.height);

  for (let token = 0; token < TOKEN_COUNT; token += 1) {
    const { x: tokenX, y: tokenY } = tokenDisplayPosition(token);
    for (let dimension = 0; dimension < EMBEDDING_DIMENSIONS; dimension += 1) {
      const x = tokenX * cellWidth + dimension % state.preview.vectorColumns;
      const y = tokenY * cellHeight + Math.floor(dimension / state.preview.vectorColumns);
      const output = (y * ui.canvas.width + x) * 4;
      const color = signedColor(state.preview.vectorColors[token * EMBEDDING_DIMENSIONS + dimension]);
      image.data[output] = color[0];
      image.data[output + 1] = color[1];
      image.data[output + 2] = color[2];
      image.data[output + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
}

function drawPreview() {
  if (!state.preview) return;
  const context = ui.canvas.getContext("2d");
  if (state.view === "vectors") drawVectors(context);
  else drawSpatial(context);
}

function formatValue(value) {
  if (!Number.isFinite(value)) return String(value);
  if (value === 0) return "0";
  if (Math.abs(value) >= 1000 || Math.abs(value) < 0.001) return value.toExponential(4);
  return value.toPrecision(6);
}

function showToken(token) {
  state.selectedToken = token;
  const position = tokenDisplayPosition(token);
  ui.tokenTitle.textContent = `Token ${token}`;
  ui.tokenPosition.textContent = `x=${position.x}, y=${position.y} / ${EMBEDDING_DIMENSIONS} FP16 values`;

  const offset = token * EMBEDDING_DIMENSIONS;
  const rows = [];
  for (let dimension = 0; dimension < EMBEDDING_DIMENSIONS; dimension += 1) {
    const value = halfToFloat(state.preview.vectorBits[offset + dimension]);
    const row = document.createElement("div");
    const label = document.createElement("label");
    const output = document.createElement("output");
    label.textContent = `d${dimension}`;
    output.textContent = formatValue(value);
    output.className = value < 0 ? "negative" : value > 0 ? "positive" : "";
    row.append(label, output);
    rows.push(row);
  }
  ui.tokenValues.replaceChildren(...rows);
  ui.tokenPanel.hidden = false;
}

function selectToken(event) {
  if (!state.preview) return;
  const bounds = ui.canvas.getBoundingClientRect();
  const x = (event.clientX - bounds.left) * ui.canvas.width / bounds.width;
  const y = (event.clientY - bounds.top) * ui.canvas.height / bounds.height;

  if (state.view === "spatial") {
    const tokenX = TOKEN_COLUMNS - 1 - Math.min(TOKEN_COLUMNS - 1, Math.floor(x));
    const tokenY = Math.min(TOKEN_ROWS - 1, Math.floor(y));
    showToken(tokenY * TOKEN_COLUMNS + tokenX);
    return;
  }

  const cellWidth = state.preview.vectorColumns + VECTOR_GAP;
  const cellHeight = state.preview.vectorRows + VECTOR_GAP;
  const displayTokenX = Math.floor(x / cellWidth);
  const tokenY = Math.floor(y / cellHeight);
  const insideVector = x % cellWidth < state.preview.vectorColumns && y % cellHeight < state.preview.vectorRows;
  if (displayTokenX < TOKEN_COLUMNS && tokenY < TOKEN_ROWS && insideVector) {
    const tokenX = TOKEN_COLUMNS - 1 - displayTokenX;
    showToken(tokenY * TOKEN_COLUMNS + tokenX);
  }
}

function setView(view) {
  state.view = view;
  const vectors = view === "vectors";
  ui.spatialTab.classList.toggle("active", !vectors);
  ui.vectorsTab.classList.toggle("active", vectors);
  ui.spatialTab.setAttribute("aria-selected", String(!vectors));
  ui.vectorsTab.setAttribute("aria-selected", String(vectors));
  ui.canvas.classList.toggle("vectors", vectors);
  ui.canvas.setAttribute("aria-label", vectors ? "All 264 raw embedding vectors" : "Spatial visualization of local image embeddings");
  ui.legendBar.classList.toggle("vectors", vectors);
  ui.legendLow.textContent = vectors ? "negative" : "similar";
  ui.legendHigh.textContent = vectors ? "positive" : "distinctive";
  ui.description.textContent = vectors
    ? "All 264 embeddings are shown in spatial order. Each mini-matrix is one actual 768-value vector reshaped to 32 x 24. Click one to inspect all FP16 values."
    : "Each tile is one image region. Brighter tiles differ more from the frame-wide average embedding. This shows feature novelty, not a reconstructed image.";
  drawPreview();
}

// Camera and one-frame-per-second loop

function showFrame(result) {
  state.preview = decodePreview(result.preview);
  state.frames += 1;
  ui.frameCount.textContent = `${state.frames} ${state.frames === 1 ? "frame" : "frames"}`;
  ui.gpuTime.textContent = `${result.inferenceMs.toFixed(0)} ms GPU`;
  ui.capturedAt.textContent = new Date().toLocaleTimeString();
  ui.emptyState.hidden = true;
  drawPreview();
  if (state.selectedToken !== null) showToken(state.selectedToken);
}

async function encodeNext() {
  if (!state.running || state.encoding || state.disposed) return;
  state.encoding = true;
  const started = performance.now();
  try {
    const bitmap = await createImageBitmap(ui.video);
    const result = await workerRequest("encode", bitmap);
    window.__gemmaEmbeddingDiagnostics = { ...result.diagnostics, lastEncode: result };
    showFrame(result);
  } catch (error) {
    stopWithError(error);
  } finally {
    state.encoding = false;
  }

  if (state.running) {
    const delay = Math.max(0, FRAME_INTERVAL_MS - (performance.now() - started));
    state.timer = setTimeout(encodeNext, delay);
  }
}

async function start() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    if (state.disposed) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    state.stream = stream;
    ui.video.srcObject = stream;
    await ui.video.play();
    if (ui.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      await new Promise((resolve) => ui.video.addEventListener("loadeddata", resolve, { once: true }));
    }

    const prepared = await workerRequest("prepare");
    window.__gemmaEmbeddingDiagnostics = prepared.diagnostics;
    ui.gpuName.textContent = prepared.diagnostics.adapterInfo?.vendor === "apple" ? "Apple GPU" : "WebGPU";
    ui.pause.disabled = false;
    showRunningState();
    encodeNext();
  } catch (error) {
    stopWithError(error);
  }
}

function toggleRunning() {
  if (state.disposed) return;
  state.running = !state.running;
  clearTimeout(state.timer);
  showRunningState();
  if (state.running && !state.encoding) encodeNext();
}

function dispose() {
  if (state.disposed) return;
  state.disposed = true;
  state.running = false;
  clearTimeout(state.timer);
  stopCamera();
  rejectPending(new Error("Encoder worker disposed"));
  worker.terminate();
}

ui.pause.addEventListener("click", toggleRunning);
ui.spatialTab.addEventListener("click", () => setView("spatial"));
ui.vectorsTab.addEventListener("click", () => setView("vectors"));
ui.canvas.addEventListener("click", selectToken);
ui.closeToken.addEventListener("click", () => {
  state.selectedToken = null;
  ui.tokenPanel.hidden = true;
});
window.addEventListener("beforeunload", dispose);

start();
