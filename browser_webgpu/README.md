# Gemma 4 E4B WebGPU Benchmark

This isolated harness loads the fixed-shape FP16 ONNX vision tower in an ONNX
Runtime Web Worker and runs it on Chromium's hardware WebGPU adapter. It records
download and session setup time, cold and warm inference latency, output error,
device-loss status, adapter capabilities, and browser-process-tree RSS.

## Live camera embedding viewer

The same WebGPU encoder can run as a local camera website. The site does not use
an API key, upload camera frames, or require an H200 decoder. It requires macOS
with an Apple GPU, Node.js 20 or newer, and current Google Chrome.

From a clone of this repository:

```bash
cd browser_webgpu
npm install
npm run viewer
```

On the first run, the command downloads the pinned 308 MB encoder release,
verifies its byte size and SHA-256 digest while streaming it to disk, and caches
it under `artifacts/browser-webgpu/`. Later runs verify and reuse that file.

Open [http://localhost:3000](http://localhost:3000) in Chrome and allow camera
access. Keep the terminal running while using the site and press `Ctrl+C` to
stop it.

Use another port when needed:

```bash
VIEWER_PORT=3001 npm run viewer
```

The model is published separately from Git source under the Apache License 2.0;
see [`MODEL_LICENSE`](MODEL_LICENSE) and [`MODEL_NOTICE`](MODEL_NOTICE). The
browser also caches the verified file in OPFS after the local server delivers it.
Maintainers can override `VIEWER_MODEL` or `VIEWER_MODEL_URL` for release tests.

The viewer keeps one WebGPU session resident and processes the latest camera
frame at most once per second without overlapping GPU work. `Spatial novelty`
shows the `22 x 12` visual-token map. `264 raw vectors` shows every `768`-value
FP16 token; click a mini-matrix to inspect dimensions `d0` through `d767`.

To inspect runtime details, open Chrome DevTools and evaluate:

```js
window.__gemmaEmbeddingDiagnostics
```

A local hardware run reports `executionProviders: ["webgpu"]`,
`isFallbackAdapter: false`, and the selected GPU adapter. Run the automated
camera and interaction smoke test while the viewer server is running with:

```bash
npm run smoke:viewer
```

### Viewer code structure

The viewer uses native browser modules and keeps each runtime concern separate:

- `viewer.mjs` starts the application.
- `viewer-app.mjs` owns camera and one-frame-per-second scheduling.
- `viewer-ui.mjs` owns DOM state and token inspection.
- `embedding-canvas.mjs` draws previews and maps clicks to tokens.
- `encoder-client.mjs` provides the page-to-Worker request interface.
- `viewer-worker.mjs` serializes Worker jobs.
- `encoder-runtime.mjs` coordinates preprocessing, inference, and previews.
- `webgpu-session.mjs` owns model caching and the WebGPU-only ORT session.
- `image-preprocessing.mjs` packs camera pixels into the model input.
- `embedding-preview.mjs` computes novelty and raw-vector colors.
- `viewer-config.mjs` and `fp16.mjs` contain shared constants and numeric tools.

Run the fast unit checks for the pure modules with `npm test`. The browser smoke
test remains the end-to-end check for camera, WebGPU, visualization, selection,
and pause behavior.

## Generate the encoder

Generate the model and deterministic FP16 fixture:

```bash
uv run --extra web-export python3 -m optimized_v2.onnx_gemma4_e4b_vision \
  --artifact artifacts/gemma-4-e4b/client \
  --image artifacts/gemma-4-12b/sample.png \
  --output artifacts/browser-webgpu/gemma4-e4b-web-fp16.onnx \
  --fixture-dir artifacts/browser-webgpu/fixture
```

Build the accepted optimized graph from that export. This produces the default
model used by the viewer:

```bash
uv run --extra web-export python3 -m optimized_v2.fuse_onnx_gemma4_e4b_rmsnorm \
  --input artifacts/browser-webgpu/gemma4-e4b-web-fp16.onnx \
  --output artifacts/browser-webgpu/gemma4-e4b-web-fp16-fused-rmsnorm.onnx \
  --fixture-dir artifacts/browser-webgpu/fixture --skip-runtime
uv run --extra web-export python3 -m optimized_v2.fuse_onnx_gemma4_e4b_rope \
  --input artifacts/browser-webgpu/gemma4-e4b-web-fp16-fused-rmsnorm.onnx \
  --output artifacts/browser-webgpu/gemma4-e4b-web-fp16-fused-rmsnorm-rope.onnx \
  --fixture-dir artifacts/browser-webgpu/fixture --skip-runtime
uv run --extra web-export python3 -m optimized_v2.fuse_onnx_gemma4_e4b_fast_gelu \
  --input artifacts/browser-webgpu/gemma4-e4b-web-fp16-fused-rmsnorm-rope.onnx \
  --output artifacts/browser-webgpu/gemma4-e4b-web-fp16-fused-rmsnorm-rope-fastgelu.onnx \
  --fixture-dir artifacts/browser-webgpu/fixture --skip-runtime
uv run --extra web-export python3 -m optimized_v2.fuse_onnx_gemma4_e4b_matmul_clip \
  --input artifacts/browser-webgpu/gemma4-e4b-web-fp16-fused-rmsnorm-rope-fastgelu.onnx \
  --output artifacts/browser-webgpu/gemma4-e4b-web-fp16-fused-rmsnorm-rope-fastgelu-matmulclip.onnx
```

## Run the benchmark

Install and run the unprofiled latency benchmark:

```bash
cd browser_webgpu
npm install
npx playwright install chromium
npm run benchmark
```

Set `BENCHMARK_MODEL`, `BENCHMARK_FIXTURE`, `BENCHMARK_ROUNDS`,
`BENCHMARK_OUTPUT`, `BENCHMARK_OUTPUT_F16`, `BENCHMARK_PROFILE=1`, or
`BENCHMARK_VERBOSE=1` to override the defaults. `BENCHMARK_OUTPUT_F16` opts into
capturing the actual raw FP16 output for composed quality tests. Profiling emits
an ONNX Runtime node trace and WebGPU timestamp queries, so profile results must
not be used as latency measurements.

The default model is the accepted RMSNorm, RoPE, FastGELU, and MatMul+Clip graph.
It requires the pinned ONNX Runtime patch applied by `npm install` and every
`npm run benchmark`. Set `BENCHMARK_MODEL=gemma4-e4b-web-fp16.onnx` to run the
original export instead.

Additional optimization experiments use `BENCHMARK_PREPROCESS_IMAGE`,
`BENCHMARK_CACHE_MODE`, `BENCHMARK_SESSION_CYCLES`,
`BENCHMARK_REPEAT_MESSAGES`, and `BENCHMARK_MODEL_ENCODING`. MatMul row counts
can be selected per known GEMM family with the `ORT_MATMUL_ROWS_*` variables;
`node sweep-matmul.mjs` runs the bounded row-4/8/16 sweep.

The RSS metric includes the Node harness and every Chromium child process. On
Apple silicon it observes host/unified-memory pressure, but it is not a dedicated
WebGPU allocation counter.
