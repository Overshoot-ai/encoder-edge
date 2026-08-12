import { EncoderClient } from "./encoder-client.mjs";
import { ViewerUi } from "./viewer-ui.mjs";

const FRAME_INTERVAL_MS = 1000;

export class ViewerApp {
  #client;
  #disposed = false;
  #encoding = false;
  #running = true;
  #stream;
  #timer;
  #ui;

  constructor() {
    this.#ui = new ViewerUi({ onPauseToggle: () => this.#toggleRunning() });
    this.#client = new EncoderClient("/viewer-worker.mjs", (error) => this.#fail(error));
  }

  async start() {
    let startupFailed = false;
    const mediaPromise = navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    }).then((stream) => {
      if (startupFailed || this.#disposed) stream.getTracks().forEach((track) => track.stop());
      return stream;
    });
    try {
      const [stream, prepared] = await Promise.all([
        mediaPromise,
        this.#client.prepare(),
      ]);
      if (this.#disposed) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      this.#stream = stream;
      this.#ui.video.srcObject = stream;
      await this.#ui.video.play();
      if (this.#ui.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        await new Promise((resolve) => {
          this.#ui.video.addEventListener("loadeddata", resolve, { once: true });
        });
      }

      window.__gemmaEmbeddingDiagnostics = prepared.diagnostics;
      this.#ui.setReady(prepared.diagnostics);
      this.#ui.setRunning(true);
      this.#encodeNext();
    } catch (error) {
      startupFailed = true;
      if (this.#disposed) return;
      this.#fail(error);
    }
  }

  #toggleRunning() {
    if (this.#disposed) return;
    this.#running = !this.#running;
    clearTimeout(this.#timer);
    this.#ui.setRunning(this.#running);
    if (this.#running && !this.#encoding) this.#encodeNext();
  }

  async #encodeNext() {
    if (!this.#running || this.#encoding) return;
    this.#encoding = true;
    const started = performance.now();
    try {
      const result = await this.#client.encode(this.#ui.video);
      window.__gemmaEmbeddingDiagnostics = { ...result.diagnostics, lastEncode: result };
      this.#ui.showFrame(result);
    } catch (error) {
      this.#fail(error);
      return;
    } finally {
      this.#encoding = false;
    }

    if (this.#running) {
      const delay = Math.max(0, FRAME_INTERVAL_MS - (performance.now() - started));
      this.#timer = setTimeout(() => this.#encodeNext(), delay);
    }
  }

  #fail(error) {
    if (this.#disposed) return;
    this.#running = false;
    clearTimeout(this.#timer);
    this.#stream?.getTracks().forEach((track) => track.stop());
    this.#stream = undefined;
    this.#ui.showError(error);
  }

  dispose() {
    if (this.#disposed) return;
    this.#disposed = true;
    this.#running = false;
    clearTimeout(this.#timer);
    this.#stream?.getTracks().forEach((track) => track.stop());
    this.#client.dispose();
  }
}
