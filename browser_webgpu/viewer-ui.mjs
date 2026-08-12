import {
  decodePreview,
  drawEmbedding,
  tokenAtPoint,
  tokenValues,
} from "./embedding-canvas.mjs";
import { EMBEDDING_DIMENSIONS } from "./viewer-config.mjs";

const VIEW_COPY = {
  spatial: {
    canvasLabel: "Spatial visualization of local image embeddings",
    description: "Each tile is one image region. Brighter tiles differ more from the frame-wide average embedding. This shows feature novelty, not a reconstructed image.",
    low: "similar",
    high: "distinctive",
  },
  vectors: {
    canvasLabel: "All 264 raw embedding vectors",
    description: "All 264 embeddings are shown in spatial order. Each mini-matrix is one actual 768-value vector reshaped to 32 x 24. Click one to inspect all FP16 values.",
    low: "negative",
    high: "positive",
  },
};

function element(selector) {
  const value = document.querySelector(selector);
  if (!value) throw new Error(`Missing viewer element: ${selector}`);
  return value;
}

function formatValue(value) {
  if (!Number.isFinite(value)) return String(value);
  if (value === 0) return "0";
  if (Math.abs(value) >= 1000 || Math.abs(value) < 0.001) return value.toExponential(4);
  return value.toPrecision(6);
}

export class ViewerUi {
  #capturedAt = element("#captured-at");
  #canvas = element("#embedding-canvas");
  #description = element("#description");
  #emptyState = element("#empty-state");
  #error = element("#error");
  #frameCount = element("#frame-count");
  #frames = 0;
  #gpuName = element("#gpu-name");
  #gpuTime = element("#gpu-time");
  #legendBar = element("#legend-bar");
  #legendHigh = element("#legend-high");
  #legendLow = element("#legend-low");
  #pause = element("#pause");
  #preview;
  #selectedToken = null;
  #spatialTab = element("#spatial-tab");
  #status = element("#status");
  #tokenPanel = element("#token-panel");
  #tokenPosition = element("#token-position");
  #tokenTitle = element("#token-title");
  #tokenValues = element("#token-values");
  #vectorsTab = element("#vectors-tab");
  #view = "spatial";
  video = element("#camera");

  constructor({ onPauseToggle }) {
    this.#pause.addEventListener("click", onPauseToggle);
    this.#spatialTab.addEventListener("click", () => this.setView("spatial"));
    this.#vectorsTab.addEventListener("click", () => this.setView("vectors"));
    element("#close-token").addEventListener("click", () => this.closeToken());
    this.#canvas.addEventListener("click", (event) => {
      if (!this.#preview) return;
      const token = tokenAtPoint(this.#canvas, this.#preview, this.#view, event);
      if (token !== null) this.#showToken(token);
    });
  }

  setStatus(text, state = "") {
    this.#status.className = `status ${state}`.trim();
    this.#status.querySelector("span").textContent = text;
  }

  showError(error) {
    this.#pause.disabled = true;
    this.#error.hidden = false;
    this.#error.textContent = error.message || String(error);
    this.setStatus("Encoding stopped");
  }

  setReady(diagnostics) {
    this.#gpuName.textContent = diagnostics.adapterInfo?.vendor === "apple" ? "Apple GPU" : "WebGPU";
    this.#pause.disabled = false;
  }

  setRunning(running) {
    this.#pause.textContent = running ? "Pause" : "Resume";
    this.setStatus(
      running ? "Encoding the latest frame every second" : "Encoding paused",
      running ? "running" : "paused",
    );
  }

  showFrame(result) {
    this.#preview = decodePreview(result.preview);
    this.#frames += 1;
    this.#frameCount.textContent = `${this.#frames} ${this.#frames === 1 ? "frame" : "frames"}`;
    this.#gpuTime.textContent = `${result.inferenceMs.toFixed(0)} ms GPU`;
    this.#capturedAt.textContent = new Date().toLocaleTimeString();
    this.#emptyState.hidden = true;
    drawEmbedding(this.#canvas, this.#preview, this.#view);
    if (this.#selectedToken !== null) this.#showToken(this.#selectedToken);
  }

  setView(view) {
    this.#view = view;
    const vectors = view === "vectors";
    this.#spatialTab.classList.toggle("active", !vectors);
    this.#vectorsTab.classList.toggle("active", vectors);
    this.#spatialTab.setAttribute("aria-selected", String(!vectors));
    this.#vectorsTab.setAttribute("aria-selected", String(vectors));
    this.#canvas.classList.toggle("vectors", vectors);
    this.#legendBar.classList.toggle("vectors", vectors);

    const copy = VIEW_COPY[view];
    this.#canvas.setAttribute("aria-label", copy.canvasLabel);
    this.#legendLow.textContent = copy.low;
    this.#legendHigh.textContent = copy.high;
    this.#description.textContent = copy.description;
    drawEmbedding(this.#canvas, this.#preview, view);
  }

  #showToken(token) {
    this.#selectedToken = token;
    this.#tokenTitle.textContent = `Token ${token}`;
    this.#tokenPosition.textContent = `x=${token % this.#preview.width}, y=${Math.floor(token / this.#preview.width)} / ${EMBEDDING_DIMENSIONS} FP16 values`;
    this.#tokenValues.replaceChildren(...tokenValues(this.#preview, token).map((value, dimension) => {
      const row = document.createElement("div");
      const label = document.createElement("label");
      const output = document.createElement("output");
      label.textContent = `d${dimension}`;
      output.textContent = formatValue(value);
      output.className = value < 0 ? "negative" : value > 0 ? "positive" : "";
      row.append(label, output);
      return row;
    }));
    this.#tokenPanel.hidden = false;
  }

  closeToken() {
    this.#selectedToken = null;
    this.#tokenPanel.hidden = true;
  }
}
