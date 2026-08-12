export class EncoderClient {
  #nextRequestId = 1;
  #pending = new Map();
  #disposed = false;
  #worker;

  constructor(url, onError) {
    this.#worker = new Worker(url, { type: "module" });
    this.#worker.onmessage = ({ data }) => this.#handleMessage(data);
    this.#worker.onerror = ({ message }) => {
      const error = new Error(message || "WebGPU worker failed");
      this.#rejectPending(error);
      onError(error);
    };
  }

  #handleMessage(data) {
    const request = this.#pending.get(data.id);
    if (!request) return;
    this.#pending.delete(data.id);
    if (data.type === "error") request.reject(new Error(data.error));
    else request.resolve(data);
  }

  #rejectPending(error) {
    for (const request of this.#pending.values()) request.reject(error);
    this.#pending.clear();
  }

  #request(type, bitmap) {
    if (this.#disposed) return Promise.reject(new Error("Encoder worker disposed"));
    const id = this.#nextRequestId++;
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject });
      if (bitmap) this.#worker.postMessage({ id, type, bitmap }, [bitmap]);
      else this.#worker.postMessage({ id, type });
    });
  }

  prepare() {
    return this.#request("prepare");
  }

  async encode(video) {
    return this.#request("encode", await createImageBitmap(video));
  }

  dispose() {
    if (this.#disposed) return;
    this.#disposed = true;
    this.#rejectPending(new Error("Encoder worker disposed"));
    this.#worker.terminate();
  }
}
