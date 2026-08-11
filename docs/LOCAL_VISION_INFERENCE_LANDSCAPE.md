# Local and Split Vision Inference: What Exists and How to Build It Properly

Research cutoff: 2026-08-10

Status: evidence review, not a rollout decision

Related project documents:

- `docs/STEP_1_RESEARCH.md` contains the earlier model-by-model split-inference survey and encoder FLOP table.
- `docs/STEP_2_POC.md` documents the first physical client/server proof of concept.
- `docs/BROWSER_WEBGPU_POC.md` documents the measured ONNX export.
- `docs/PACKAGING_PLAN.md` is a decision-oriented proposal. This report intentionally steps back from that proposal and asks what public systems have actually done.

## 1. Question and research method

The question is broader than "can an ONNX model run in a browser?" It is:

> How have people delivered local or split AI computation to end users, which parts are proven in deployed software, and what engineering is required to do the same responsibly for a client-side vision encoder feeding a cloud VLM?

This review covers four bodies of evidence:

1. **Deployed browser runtimes:** ONNX Runtime Web, Transformers.js, WebLLM, and the llama.cpp WebGPU work in LlamaWeb.
2. **Deployed native/local patterns:** Ollama, Chrome built-in model management, Sparkle/macOS distribution, and local-first applications such as Screenpipe.
3. **Split-VLM research:** complete edge encoders, feature compression, progressive representations, query-aware token selection, routing, and nearby-device designs.
4. **Security and privacy research:** semantic recovery, image reconstruction, exact-text recovery, and in-transit token manipulation.

Sources were checked against papers, official documentation, and public repositories rather than relying on product summaries. Papers are described as papers unless an implementation was found. A reported result is not treated as a production result, and a browser API being present is not treated as proof that a particular runtime supports it.

The evidence labels used below are:

- **Measured here:** reproduced in this repository.
- **Deployed public software:** a maintained runtime or product with public documentation/code.
- **Research implementation:** code exists, but it is not a hardened end-user product.
- **Paper result:** reported by authors and not reproduced here.
- **Engineering inference:** a conclusion drawn from the evidence, not a directly reported result.

The search cannot establish what private companies have built internally. "No product found" means no public product, API, paper, or repository was found in this search.

## 2. Executive findings

### 2.1 The main finding

There is no single established product pattern for the exact architecture in this repository. Instead, public systems cluster into four patterns:

| Pattern | What users install/download | Where inference runs | Public examples | Maturity |
|---|---|---|---|---|
| Browser-local | Web application plus model weights | Browser WebGPU/WASM | WebLLM, Transformers.js, ORT Web demos | Real and deployed, but hardware/runtime support is heterogeneous |
| Native local engine | Signed app/daemon plus separately managed weights | Native Metal/CUDA/CPU | Ollama, llama.cpp apps, Screenpipe | Most mature for reliable local AI |
| Nearby-device local | Small sensor/client plus a nearby laptop/phone host | User-controlled nearby computer | OpenGlass | Working research implementation; useful architectural precedent |
| Client encoder plus cloud LLM | Encoder/features on client, projector/LLM in cloud | Split across trust boundary | Distributed VLMs, TOFC, Progressive Semantic Communication, LAST | Active research; no complete hardened consumer product found |

The exact split has strong research precedent and serving-engine support, but public work still leaves the product layer to the implementer: artifact delivery, version negotiation, untrusted tensor validation, user consent, fallback behavior, privacy claims, and support across heterogeneous devices.

### 2.2 What is genuinely proven

- Browsers can run substantial transformer models with WebGPU. WebLLM measured 71-80% of native MLC decode throughput for two 4-bit LLMs on an M3 Max. LlamaWeb evaluated 10 models on 16 GPUs from eight vendors and found large differences across browsers, devices, model phases, and precisions. These results prove feasibility, not the latency of this vision tower. [WebLLM](https://arxiv.org/abs/2412.15803) [LlamaWeb](https://arxiv.org/abs/2605.20706)
- Production-like browser model delivery uses immutable remote artifacts, a manifest/config, local persistent caches, background workers, explicit progress, and runtime capability checks. WebLLM and ORT Web both document these pieces. [WebLLM deployment](https://llm.mlc.ai/docs/deploy/webllm.html) [WebLLM cache example](https://github.com/mlc-ai/web-llm/tree/main/examples/cache-usage) [ORT large models](https://onnxruntime.ai/docs/tutorials/web/large-models.html)
- Native local AI products normally separate the application/runtime from model weights. Ollama downloads models into a user data directory, exposes a loopback API, updates the application independently, and provides an explicit local-only mode. [Ollama FAQ](https://docs.ollama.com/faq)
- A complete vision encoder can run on the edge while an LLM runs elsewhere. Distributed VLMs, TOFC, and Progressive Semantic Communication all use that basic boundary. LAST additionally selects query-relevant encoder tokens before transmission. [Distributed VLMs](https://par.nsf.gov/biblio/10639785-distributed-vlms-efficient-vision-language-processing-through-cloud-edge-collaboration) [TOFC](https://arxiv.org/abs/2503.12926) [Progressive Semantic Communication](https://arxiv.org/abs/2604.26508) [LAST](https://arxiv.org/abs/2607.27952)
- The current Gemma 4 E4B tower can be exported as a fixed-shape ONNX graph. The measured artifact is 308,190,557 bytes and produces the required `[1,264,768]` pre-projector output. This proves exportability and CPU execution, not WebGPU performance or final model quality. See `docs/BROWSER_WEBGPU_POC.md`.

### 2.3 What is not proven

- No public consumer service was found that ships this exact interaction: arbitrary users download the provider's vision encoder, run it in a browser or native client, and submit visual features to a hosted matching LLM through a hardened public API.
- No browser benchmark has been run for this repository's model. Operator compatibility on paper is not enough; actual provider assignment, memory use, device loss, and warm/cold latency still need measurement.
- The planned INT4 browser model does not exist yet. Its size and quality are estimates until the actual graph is quantized, coverage is counted, and composed VLM quality is tested.
- Raw-pixel avoidance is not semantic privacy. Multiple attacks recover labels, captions, images, or text from representations used in split inference.
- No general token-reduction method preserves every workload. OCR, charts, documents, small objects, and fine spatial reasoning fail earlier than coarse scene understanding.

### 2.4 The newest relevant work

The newest papers answer different parts of the problem and should not be conflated:

- **Token Communication for MLLMs, August 7, 2026:** the newest related communication paper found. It sends neural-codec latents, reconstructs a pixel prior at the receiver, and injects adapted features into a server-side vision tokenizer. It is not a complete target-encoder-on-client split. [paper](https://arxiv.org/abs/2608.07279)
- **Inverting the Hidden, August 2, 2026:** the newest privacy result found. It reconstructs images and text from hidden states after roughly two-thirds of Qwen3-VL-8B or LLaVA-1.5-7B's LLM layers, under a knowledgeable honest-but-curious server. It strengthens the case that moving the split deeper does not make representations private. [paper](https://arxiv.org/abs/2608.01020)
- **LAST, July 30, 2026:** the newest direct client-encoder/cloud-LLM proposal found. It runs a shared vision encoder and a small proxy VLM at the edge, then sends a query-selected subset of pre-connector encoder tokens. It is evaluated on one A800 rather than a physical edge/cloud testbed. [paper](https://arxiv.org/abs/2607.27952)
- **Progressive Semantic Communication, April 29, 2026:** the newest clean physical testbed found for a complete target encoder on a genuinely constrained edge device and target LLM on a separate GPU server. [paper](https://arxiv.org/abs/2604.26508)
- **OpenGlass, July 3, 2026:** a useful public end-to-end local-first product precedent, but it sends JPEG frames from glasses to a nearby user-controlled laptop and runs the complete VLM there. It does not send target-model features to a cloud LLM. [paper](https://arxiv.org/abs/2607.03213) [code](https://github.com/OpenSQZ/OpenGlass)

## 3. The architectures people have tried

### 3.1 Full browser-local inference

The browser downloads model artifacts and executes all relevant model work locally.

```text
CDN/model host
  -> versioned model files
  -> browser persistent cache
  -> Web Worker
  -> WebGPU/WASM runtime
  -> local output
```

What public systems do:

- **WebLLM** compiles a supported model architecture ahead of time into converted/quantized weight shards plus a model-specific WASM/WebGPU library. It exposes an OpenAI-compatible streaming API and can place the engine in a Web Worker or Service Worker. Its paper evaluates decode, not a standalone ViT encoder. [paper](https://arxiv.org/abs/2412.15803) [docs](https://webllm.mlc.ai/docs/)
- **Transformers.js** loads ONNX artifacts and delegates execution to ONNX Runtime Web. It supports WebGPU through `device: 'webgpu'`, quantized variants, hosted or custom model locations, and common preprocessing pipelines. It is convenient when an architecture is already supported by the library. A custom fixed-shape graph can also use ORT Web directly and avoid adapting the Transformers.js model abstraction. [WebGPU guide](https://huggingface.co/docs/transformers.js/en/guides/webgpu) [custom models](https://huggingface.co/docs/transformers.js/en/custom_usage)
- **ONNX Runtime Web** is the lowest-friction match for this repository because an ONNX graph already exists. It supports external data, explicit execution providers, WebGPU profiling, graph capture for static all-WebGPU graphs, GPU-resident tensors, and Cache API/OPFS storage. [Web support matrix](https://onnxruntime.ai/docs/get-started/with-javascript/web.html) [WebGPU EP](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html) [large models](https://onnxruntime.ai/docs/tutorials/web/large-models.html)
- **LlamaWeb** adds WebGPU to llama.cpp/GGUF. It demonstrates an important systems lesson: functional portability is not performance portability. It uses static memory planning, streaming from OPFS to GPU, tunable kernels, and many quantization formats. It is less direct for this standalone ONNX vision tower, but its cross-device results are valuable evidence. [paper](https://arxiv.org/abs/2605.20706)

What transfers to this project:

- Use ORT Web directly unless MLC or llama.cpp gains a compelling, measured advantage for this exact graph.
- Put inference and preprocessing in a dedicated Worker. Do not perform hundreds of milliseconds of model work on the UI thread.
- Keep the execution-provider list strict during qualification. `['webgpu', 'wasm']` can silently produce a system that "works" while material compute runs on CPU; qualification should initially use `['webgpu']` and fail if unsupported.
- Treat browser and GPU combinations as deployment targets, not a single platform. The same WebGPU code can differ by orders of magnitude. LlamaWeb measured about 1 token/s in Firefox versus about 52 tokens/s in Chrome for one M3 configuration.
- Measure the model phase that resembles the workload. Vision encoding is matrix-matrix/prefill-like work. LlamaWeb's decode wins do not imply a win for this encoder; its browser prefill was slower than WebLLM and Transformers.js in its comparison.

### 3.2 Native local engine

The user installs an application or helper. The helper runs the model through native APIs and may expose a local API to a web UI.

```text
signed app/helper
  -> native runtime (MLX/Metal/CUDA/CPU)
  -> separately versioned model store
  -> local UI or authenticated loopback API
```

What public systems do:

- **Ollama** installs a local engine, stores model files outside the application (`~/.ollama/models` on macOS), binds its API to `127.0.0.1` by default, manages app updates separately, and supports disabling cloud features. This is the clearest general-purpose precedent for runtime/weight separation. [FAQ](https://docs.ollama.com/faq)
- **Screenpipe** documents a local-first application with a loopback-only, token-authenticated API, OS-keychain-backed secrets, explicit optional cloud paths, retention controls, and an inspectable data-flow description. It is a product/security precedent rather than a VLM split implementation. [security architecture](https://screenpipe.com/security/architecture)
- **Sparkle** is a common macOS update framework. Its security model uses HTTPS, Developer ID signing/notarization, and Ed25519-signed update archives/appcasts. Application updates and model updates should remain separate. [Sparkle documentation](https://sparkle-project.org/documentation/)

What transfers to this project:

- The already-qualified MLX/BF16 encoder is the known native execution path. It should not be rewritten merely to share code with the browser.
- Store weights in application support storage or another user-visible model directory. Use content hashes and a small manifest rather than treating an installer filename as the model version.
- A web-to-localhost bridge is not automatically safe because it is bound to loopback. Browsers can be coerced into calling local services through CSRF or DNS rebinding.
- Chrome's Local Network Access work adds a user permission boundary for public-site requests to loopback/private addresses. Its rollout started with opt-in testing in Chrome 138; exact stable behavior must be rechecked when implementing the helper. [Chrome LNA](https://developer.chrome.com/blog/local-network-access)
- Alternatives to a localhost HTTP server are a browser extension with native messaging or a fully native UI. Both avoid some web-to-loopback exposure, but add installation/review/platform work.

### 3.3 Nearby-device local inference

OpenGlass separates a cheap sensing device from a user-controlled nearby computer:

```text
ESP32 glasses --JPEG/Wi-Fi--> nearby laptop --local VLM/TTS--> user
```

Its public implementation uses an ESP32-S3/OV5640 camera, an RTX 5060 laptop, INT4 MiniCPM-V/o through llama.cpp, Whisper, and local text-to-speech. The paper reports 993 ms median query-ready-to-audio latency with resized inputs and 97.5% of trials below two seconds. It also logs stage timings, uses streaming speech, applies explicit abstention rules, and attributes tail failures. [paper](https://arxiv.org/abs/2607.03213)

The transferable lessons are more important than its model choice:

- Nearby user-owned compute can be a practical third tier between browser and cloud.
- Resize/input adaptation was the dominant latency lever.
- A median is insufficient; the paper reports p95 and below-target pass rates and identifies capture versus inference tails.
- Safety-sensitive local systems need abstention, retake guidance, timeouts, and auditable event logs, not only a faster model.
- "Local-first" describes data flow, not a formal privacy guarantee.

### 3.4 Complete client encoder plus cloud LLM

This is the repository's architecture:

```text
CLIENT                                      SERVER
image -> processor -> vision encoder          receive versioned tensor
      -> optional reduction/codec             validate/decode
      -> binary features ------------------>  projector -> LLM -> response
```

The closest public work is summarized below.

| Work | Client/edge work | What crosses the boundary | Main reported result | Important limitation |
|---|---|---|---|---|
| Distributed VLMs (2025) | Complete vision encoder and projector | Projected visual tensor plus prompt/request ID | Up to 33.54% higher sustained throughput by overlapping edge encoding with server generation | Throughput result, not a 33.54% single-request latency reduction; no public code found |
| TOFC (2025) | Complete SigLIP/CLIP encoder, feature clustering/merging, learned entropy encoder | Entropy-coded pre-projector features | Up to 52% lower transmitted data and 63% lower system latency than ELIC at matched task performance | Trains about 300M parameters for one epoch (8.1 hours on 8 RTX 4090s); public repo omits end-to-end transport |
| Progressive Semantic Communication (2026) | Complete 4-bit SmolVLM vision encoder and a two-layer MetaAE | Ordered latent chunks in a progressive protocol | 6.94 s end to end versus 9.32 s full cloud at 1 Mbps in the full-latent case | Uses JSON for a 192.3 KiB latent, evaluates answer semantic similarity rather than broad task accuracy, code not yet usable when checked |
| Co-VStream (2026) | Complete video encoder, temporal condensation, separate captioner | Condensed features plus captions | 87.59% less communication and near cloud-only LVBench accuracy | Captions/entity graph confound the contribution of features; edge test uses desktop GPUs; no code found |
| LAST (2026) | Shared vision encoder plus small proxy VLM and query-aware selection | Selected pre-connector encoder tokens | 95.4% of full-token average performance at 12.5% token retention on 11 benchmarks | Selection roughly doubles measured edge-stage latency in the default setup; evaluated on an A800, not a physical edge/cloud link |

Detailed observations:

- **Split point matters.** TOFC deliberately transmits before the projector because encoder width is smaller than LLM width. That is also the measured choice in this repository.
- **Pipelining and compression solve different problems.** Distributed VLMs improves aggregate service throughput by overlapping different requests. TOFC and Progressive Semantic Communication target wire time and downstream token cost.
- **Adaptive representations add protocol round trips.** Progressive transmission can stop early, but every server request for another chunk adds latency. It is attractive on unstable/slow links only if the saved bytes exceed feedback cost.
- **Query-aware reduction is not reusable across questions.** LAST needs the user's query and a proxy VLM before selecting tokens. A query-independent encoder result can otherwise be cached and reused for several questions about the same image.
- **Aggressive reduction is workload-specific.** LAST's 12.5% setting preserves 95.4% of its normalized average, but ChartQA still falls from 73.0 to 64.8. The same paper shows much smaller losses at 37.5% and 62.5% retention.
- **The latest related codec is not the same split.** Token Communication for MLLMs sends neural-codec latents; the receiver reconstructs pixels and runs the target vision tokenizer while also injecting adapted latent features. It reports rate savings across MME, POPE, SeedBench, and captioning, but is a learned image/feature codec rather than a client copy of the target vision encoder. [paper](https://arxiv.org/abs/2608.07279)

### 3.5 Routing between complete edge and cloud models

Another line of work does not partition one model. It runs a small complete VLM locally and sends selected requests, normally including pixels, to a larger cloud VLM.

- **INAR-VL** routes among two INT8 2B edge VLMs and two FP16 8B cloud VLMs using image-quality and question-complexity features. On an RTX 4060 edge and RTX PRO 6000 cloud setup, it runs 36% of requests locally, reports 24% lower latency and 26% lower energy than cloud-only, and retains 97% of cloud accuracy. Its cloud path still uploads the image. [paper](https://arxiv.org/abs/2605.18853)
- **edgeVLM** runs a small VLM continuously and reuses delayed cloud answers as context for future video frames. It is designed for real-time streams where cloud answers become stale. It also uploads selected frames and therefore does not provide raw-pixel minimization. [paper](https://arxiv.org/abs/2508.12638)

Routing is a valid alternative when a local small model is good enough for easy requests. It is not interchangeable with this split: the local and cloud answers can differ, and cloud-routed requests usually expose pixels.

### 3.6 What can be reused today

The public implementation landscape is fragmented. The serving pieces are stronger than the end-user client pieces:

| Project | Reusable capability | Public status | Suitability for an Internet-facing client split |
|---|---|---|---|
| vLLM | Accepts precomputed `image_embeds`; also supports trusted-cluster encoder disaggregation | Maintained open source; embedding input is documented as trusted-user functionality | Use behind a validating gateway, not as the public parser |
| TensorRT-LLM | External multimodal embedding tensors and a standalone encoder/embedding-handle path | Features merged with tests | External tensor path is relevant; shared-memory handle path is cluster-internal |
| llm-d | Kubernetes encoder/prefill/decode disaggregation around vLLM | Experimental public recipes | Useful for server-side scaling, not client feature ingress by itself |
| NVIDIA Dynamo | Multimodal routing and encoder disaggregation across backends | Maintained public platform | Useful inside the server cluster; input behavior depends on backend |
| TOFC | Feature merger, entropy model, and checkpoint | Archived partial research code | Compression components only; transport/service path is omitted |
| VisionZip / LAST-style pruning | Token selection/merging ideas | VisionZip code is public; no LAST implementation was found | Requires model-specific integration and quality validation |
| Progressive Semantic Communication | Progressive MetaAE protocol | Paper promises code; repository was not usable when checked | Research design, not a dependency to adopt today |
| OpenGlass | Complete ESP32-to-local-host application and evaluation harness | Public code, prompts, logs, and setup | Strong systems reference, but uses pixels and a complete local VLM |

No checked hosted-model API from Fireworks, Together AI, or Hugging Face Inference Providers exposes arbitrary client-generated visual tokens as a standard request type. Modal and Baseten can host custom code/containers, so they can run a project-owned gateway plus vLLM or another engine; that is infrastructure hosting, not a provider-supplied tensor contract. [Modal vLLM deployment](https://modal.com/blog/how-to-deploy-vllm) [Baseten custom server](https://docs.baseten.co/development/model/custom-server)

The practical reusable stack is therefore:

```text
project-owned browser/native client
  -> project-owned authenticated validation gateway
  -> private vLLM/TensorRT-LLM/custom model worker
```

The project still owns the processor, model artifact, wire schema, compatibility policy, privacy behavior, and quality gates.

## 4. Runtime findings for this project

### 4.1 Current measured boundary

| Property | Measured value | Evidence |
|---|---:|---|
| Source client artifact | 338.7 MB BF16, including tower/projector material | `artifacts/gemma-4-e4b/client/` |
| Browser-oriented graph | 308,190,557 bytes, tower only | `docs/BROWSER_WEBGPU_POC.md` |
| ONNX input | `[1,2376,768]` FP16 preprocessed patches | Same |
| ONNX output | `[1,264,768]` FP16 pre-projector features | Same |
| Graph | Opset 18; 4,651 nodes; 459 initializers; 145 MatMul; 16 Softmax | Same |
| ORT CPU execution | 7.84 s, used only as correctness/operator check | Same |
| Native MLX p50 | About 319-352 ms on base M4 | `benchmark-results/projector-shift/REPORT.md` |
| Current wire tensor | `[264,768]` BF16, about 405,504 tensor bytes | `optimized_v2/protocol.py` |

The browser graph starts after image preprocessing. A production browser client still needs to reproduce the exact Python image processor: decode, orientation, resize, crop/pad, channel order, normalization, patch order, position/grid metadata, and numeric rounding.

### 4.2 Browser support is runtime-specific

The official ORT Web matrix currently supports its WebGPU execution provider in Chrome/Edge on desktop and Android, but not Safari, Firefox, iOS, or Node.js. Float16 has additional browser-version requirements. WASM is more broadly supported. [ORT Web support matrix](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)

This does not mean Safari or Firefox lack the WebGPU API. LlamaWeb demonstrates Safari/iOS execution with a different runtime. It means a product using **ORT Web WebGPU** cannot infer support from `navigator.gpu` alone.

WebNN remains an experimental/future option for this use case. ORT's official matrix requires a Chrome feature flag on Windows, and its WebNN import is marked experimental. It should not be the sole production path until stable browser/runtime support and this graph are measured.

### 4.3 Correct qualification sequence

For the current graph, the proper browser proof is:

1. Create a minimal Chromium page and dedicated Worker.
2. Load a content-hashed model revision from local development hosting.
3. Request only the WebGPU execution provider.
4. Record session creation time, graph initialization, first inference, and repeated warm inferences separately.
5. Enable ORT WebGPU profiling and inspect every node/provider assignment.
6. Measure browser process memory and GPU allocation across load, warmup, repeated runs, disposal, and model replacement.
7. Compare the output tensor to the native reference and then run the composed H200 quality gate.
8. Test `GPUDevice.lost`, tab backgrounding, sleep/wake, cancellation, concurrent tabs, and recovery. WebGPU devices can be lost at any time and all resources must then be recreated. [MDN `GPUDevice.lost`](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/lost)
9. Repeat on the supported hardware/browser matrix. Do not generalize from one M4 result.

Graph capture is worth testing because the graph is fixed-shape, but ORT only supports it when all compute kernels are on WebGPU. It is an optimization after provider coverage is established, not a substitute for coverage. [ORT session options](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html)

## 5. How browser model delivery is done properly

### 5.1 Artifact structure

A robust browser artifact is not one mutable `model.onnx` URL. Use a versioned manifest and immutable content-addressed files:

```json
{
  "schema_version": 1,
  "model_id": "gemma-4-e4b-vision",
  "model_revision": "sha256:...",
  "processor_revision": "sha256:...",
  "runtime": {"name": "onnxruntime-web", "min_version": "..."},
  "input": {"shape": [1, 2376, 768], "dtype": "float16"},
  "output": {"shape": [1, 264, 768], "dtype": "float16"},
  "required_features": ["webgpu", "shader-f16"],
  "files": [
    {"url": "/models/<hash>/graph.onnx", "sha256": "...", "bytes": 1234},
    {"url": "/models/<hash>/weights-00.bin", "sha256": "...", "bytes": 8388608}
  ]
}
```

The exact fields may differ, but the properties should not:

- Model and processor versions are independently named.
- URLs are immutable and cacheable for a long time.
- Every file has an expected byte count and cryptographic digest.
- The manifest activates only after every required file verifies.
- An interrupted update does not destroy the previously working version.
- Garbage collection deletes old revisions only after a new revision has loaded successfully.

ONNX supports external data files, and ORT Web accepts their URLs, blobs, or byte arrays explicitly. [ONNX external data](https://github.com/onnx/onnx/blob/main/docs/ExternalData.md) [ORT large models](https://onnxruntime.ai/docs/tutorials/web/large-models.html)

### 5.2 Sharding and download behavior

Sharding is useful even below ONNX's 2 GB protobuf limit because it enables bounded retries, parallelism, progress reporting, and atomic per-file verification. The right shard size is an empirical CDN/browser tradeoff; a range such as 8-32 MB is a reasonable experiment, not a standard.

The web application must own interruption handling. Chrome's built-in AI models continue downloads after tab/browser closure because Chrome itself manages them. A normal website does not receive that privileged lifecycle automatically. Chrome's model manager is still a useful design reference: capability checks before download, resumable download, background updates, hot swap, and purge under disk pressure. [Chrome built-in model management](https://developer.chrome.com/docs/ai/understand-built-in-model-management)

Required UX:

- Show exact total bytes and downloaded bytes.
- Explain why the model is being downloaded and where it is stored.
- Allow cancellation and retry.
- Do not start a large download merely because a page rendered.
- Reuse the server path, if available and consented to, while the local model downloads.
- Distinguish "downloaded," "initializing GPU," and "ready." They are different failure stages.
- Expose a delete-local-model control.

### 5.3 Browser storage

Cache API, IndexedDB, and OPFS are all used by public browser ML systems. WebLLM supports all three cache backends; LlamaWeb streams weights from OPFS into GPU buffers to avoid extra whole-model copies. [WebLLM cache example](https://github.com/mlc-ai/web-llm/tree/main/examples/cache-usage) [LlamaWeb](https://arxiv.org/abs/2605.20706)

Storage facts:

- Data is scoped per origin; a model cached by one origin is not generally shared with another.
- Cache/IndexedDB/OPFS data is best-effort by default and can be evicted under storage pressure.
- `navigator.storage.persist()` requests persistent treatment, but browsers decide whether to grant it.
- Chromium permits an origin up to roughly 60% of total disk in policy terms, but that is not a promise that space is currently available.
- Private browsing has different limits and normally deletes data when the private session ends.
- Writes must handle `QuotaExceededError` and partial failure.
- WebKit can proactively evict inactive origins and uses origin-wide eviction. Installed web apps and persistent status can affect treatment, but code should always tolerate a complete cache miss.

Sources: [MDN quotas and eviction](https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria) [MDN persistence](https://developer.mozilla.org/en-US/docs/Web/API/StorageManager/persist) [WebKit storage policy](https://webkit.org/blog/14403/updates-to-storage-policy/)

### 5.4 Quantization

Large browser-local transformer systems commonly quantize model weights because download size and memory traffic matter more than preserving a server checkpoint verbatim. WebLLM commonly ships 4-bit weights; LlamaWeb supports many llama.cpp formats; Transformers.js/ORT Web support several ONNX quantized paths.

For this graph, first-party ORT `MatMulNBits` weight-only quantization is the most direct experiment because the exported graph contains 145 `MatMul` nodes and ORT Web has a corresponding WebGPU operator. The quantizer converts only eligible constant-weight inputs, so the converted node and byte coverage remain unknown until the artifact is produced. [quantizer source](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/matmul_nbits_quantizer.py) [WebGPU operator](https://github.com/microsoft/onnxruntime/blob/main/js/web/lib/wasm/jsep/webgpu/ops/matmulnbits.ts)

What is known and unknown:

| Item | Status |
|---|---|
| FP16 ONNX graph exists | Measured here |
| Symmetric RTN, block size 32 is a plausible first INT4 configuration | Engineering candidate based on ORT/Transformers.js practice |
| Most graph weight will convert to `MatMulNBits` | Unknown until node coverage is counted |
| Raw INT4 artifact will be about 85-98 MB | Estimate, not measured |
| Compressed transfer will be about 58-66 MB | Estimate from prior lossless tests, not a produced artifact |
| INT4 output will preserve composed VLM quality | Unknown; prior Q4 and even FP16 experiments make this a serious risk |
| INT4 will be faster than FP16 | Unknown; lower weight traffic can be offset by dequantization and kernel quality |

Proper quantization evaluation requires:

- Count converted and unconverted weight bytes, not only output file size.
- List every remaining high-cost FP16 node and why it was excluded.
- Use real calibration images if moving beyond round-to-nearest; random noise is not representative of vision activations.
- Test black, white, low-light, overexposed, text-heavy, chart, and high-detail inputs.
- Compare complete VLM behavior, not only encoder cosine similarity.
- Measure both startup and warm latency. A smaller file can still initialize slowly.

## 6. How native packaging is done properly

### 6.1 Separate three versioned objects

Treat these as independent releases:

1. **Application/helper:** executable code, UI, local API, update mechanism.
2. **Runtime:** MLX and other native libraries, normally bundled with the helper.
3. **Model package:** weights, processor configuration, model manifest, and compatibility metadata.

The model manifest should declare the minimum/maximum helper protocol versions. A helper update must not silently make an already downloaded model incompatible, and a model update must not activate until its complete hash and compatibility are verified.

### 6.2 macOS distribution

A direct-download macOS app should use Developer ID signing, hardened runtime, and Apple notarization to avoid Gatekeeper friction. Notarization can be automated in CI. Sparkle is a reasonable updater if its update archives and feeds are signed and the private signing key is isolated from the update host. [Apple notarization](https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution) [Sparkle](https://sparkle-project.org/documentation/)

Model choices:

- **Bundle weights in the app/dmg:** simplest first run and strongest artifact atomicity, but every application update can become large unless the updater understands component separation.
- **Download weights after install:** smaller installer and independent model cadence, but adds another resumable/download verification flow.
- **Hybrid:** bundle one known model revision and allow content-addressed model updates later.

There is no universally correct choice at 339 MB. Both bundling and first-run download are normal at this scale.

### 6.3 Localhost bridge security

If a web page controls a native helper, the minimum design is:

- Bind only to `127.0.0.1` and `::1`; never `0.0.0.0` by default.
- Generate a high-entropy per-install credential and store it in the OS keychain or a user-only file.
- Pair the browser and helper through an explicit user action. Do not place a long-lived secret in a public web page or analytics event.
- Require authentication on every endpoint, including WebSocket/SSE upgrade paths.
- Validate `Origin` and `Host`; CORS headers alone are not an authentication mechanism.
- Reject DNS-rebinding hostnames and unexpected IP families.
- Use POST plus a required custom header for state-changing operations, forcing a CORS preflight.
- Never perform state changes from GET requests.
- Apply strict request-size, concurrency, and timeout limits.
- Do not accept arbitrary filesystem paths, shell commands, model URLs, or bind addresses from the browser.
- Return a helper/protocol/model compatibility handshake before accepting inference.
- Remove or disable the service during uninstall.
- Log metadata required for diagnosis, not image/features by default.

The security precedent is not theoretical: Zoom, Ollama, Docker Desktop, and other local services have had vulnerabilities involving unauthenticated localhost access, origin trust, or network exposure. The correct lesson is not "never use localhost"; it is to treat localhost as a real network service exposed to every local process and potentially every visited website.

## 7. The split protocol and serving boundary

### 7.1 Versioned contract

The tensor is meaningful only under an exact contract. Every request must bind:

| Field | Reason |
|---|---|
| Protocol revision | Parser and compatibility behavior |
| Model/weight revision | Different weights produce incompatible features |
| Processor revision | Resize, normalization, patch order, and grid must match |
| Split point | Identifies the next server operation |
| Projector location/revision | Distinguishes encoder-width from LLM-width values |
| Shape and axis order | Prevents ambiguous interpretation and allocation abuse |
| Numeric format and byte order | BF16, FP16, FP32, INT8, and packed INT4 differ |
| Token/grid metadata | Required by dynamic-resolution/spatial models |
| Quantizer/codec revision | Required to decode compressed representations |
| Request ID, user/session, expiry | Replay protection and prompt/tensor association |

The existing binary protocol is a sound starting point because it avoids base64 and decimal-float JSON overhead. It should continue to reject malformed lengths before allocating arrays or GPU buffers.

### 7.2 Public gateway, private model server

Do not expose vLLM's raw embedding-input feature directly to untrusted clients. vLLM warns that incorrect multimodal embedding shapes can crash the engine and recommends the feature only for trusted users. [vLLM multimodal inputs](https://docs.vllm.ai/en/latest/features/multimodal_inputs/)

The public gateway should:

- Authenticate and authorize the user before parsing a large body.
- Enforce compressed and uncompressed byte limits.
- Decode into a bounded buffer.
- Validate exact shape, element count, dtype, finite values, and reasonable norms.
- Verify model/processor/split revisions against an allowlist.
- Bind prompt and feature payload to the same authenticated request.
- Apply per-user concurrency and rate limits.
- Forward an internal typed object to a private inference worker.
- Keep user tensors out of exception traces, metrics labels, and default request logs.

Validation protects availability; it does not prove the tensor came from a real image. A malicious but shape-valid tensor can still manipulate model behavior or bypass image-side safety checks.

### 7.3 Integrity and confidentiality

- Authenticated TLS protects against outside interception and modification while the request remains inside that TLS connection.
- If TLS terminates at a proxy and the tensor then passes through queues or services, bind the user, request ID, prompt hash, model revision, and payload digest with an application-level MAC/signature across those hops.
- Encrypt stored retries/queues and use short retention.
- Do not cache visual features across users.

VTM-Attack demonstrates the integrity risk: changing only 10% of visual tokens reduced Qwen2.5-VL-72B MMBench accuracy from 88.39% to 0.08% in the paper's strongest setting. Model families varied substantially in robustness. [paper](https://arxiv.org/abs/2607.02819)

## 8. Privacy findings

### 8.1 What the split does provide

- The service need not receive the original image file or its EXIF metadata.
- Raw image copies can be eliminated from server ingress, logs, object stores, and debugging workflows.
- A deep encoder plus spatial pooling can remove some pixel-local and exact-character detail.
- The client can perform local crop, redaction, frame selection, and policy checks before encoding.

This is meaningful **raw-pixel minimization**. It is not meaningless merely because embeddings leak information.

### 8.2 What it does not provide

The cloud must receive enough information to answer the question, so semantic content remains available by design.

Public attacks show several distinct leakage endpoints:

| Work | Representation attacked | Attacker knowledge | Recovered information | What it establishes |
|---|---|---|---|---|
| CapRecover (2025) | Vision-encoder intermediate/final features | Encoder architecture plus auxiliary image/caption data | CIFAR-10 labels up to 92.71% top-1; COCO captions around 0.52-0.53 ROUGE-L for CLIP/ResNet | High-level semantics can be recovered without reconstructing pixels |
| Vision Encoder as Privacy Boundary (2026) | Encoder-free visual tokens vs encoder-based outputs | Trained matched decoders | Exact access codes from Gemma4/Fuyu; 0/48 exact for Qwen3-VL, InternVL, LLaVA controls | Dense pixel-local token grids leak exact text more readily; encoder/spatial pooling helps for this endpoint |
| Image Prompt Reconstruction (2026) | Image token/LLM hidden states across distributed MLLMs | Passive black-box participant plus auxiliary image pairs | Pixel-level reconstruction in earlier layers and semantic diffusion reconstruction across deeper layers | Image information remains recoverable across several model families and layers |
| Inverting the Hidden (2026) | Hidden states after about two-thirds of the LLM | Honest-but-curious server with front-end model knowledge and auxiliary data | About 50% lower image MSE than baselines and up to 99% text-token accuracy | Moving the split deep into the LLM does not inherently make multimodal states private |

Sources: [CapRecover](https://arxiv.org/abs/2507.22828) [Vision Encoder as Privacy Boundary](https://arxiv.org/abs/2606.14783) [Image Prompt Reconstruction](https://arxiv.org/abs/2606.18710) [Inverting the Hidden](https://arxiv.org/abs/2608.01020)

These results are not contradictory. They use different representations, datasets, attackers, reconstruction targets, and success metrics:

- Encoder-based outputs can resist exact five-character OCR while still revealing objects, labels, captions, location, identity, or a semantically similar image.
- A failure to reconstruct exact pixels is not a failure to infer sensitive content.
- Spatial pooling can reduce character-level inversion but does not make semantics disappear.
- Quantization/noise may reduce one attack and leave another effective. In the encoder-free study, 3-bit quantization and moderate noise did not stop exact-code recovery; spatial pooling did.

### 8.3 Defensible product language

Accurate:

> Your image is processed by a vision encoder on your device. We send the resulting visual features, not the original image file, to our server so the language model can answer. Visual features can still contain sensitive information about the image.

Inaccurate without stronger controls and evidence:

- "Your data never leaves your device."
- "The server cannot know what is in the image."
- "Embeddings are anonymous."
- "The image cannot be reconstructed."
- "End-to-end encrypted" when the server must decrypt features for ordinary inference.

### 8.4 How to improve the privacy claim

From cheapest to strongest:

1. Minimize raw pixels and metadata, and never log features by default.
2. Use a deep semantic encoder and deliberate spatial pooling rather than an encoder-free/pixel-local representation.
3. Run explicit leakage tests against the exact shipped representation.
4. Publish the data flow, retention, and training policy with a network-inspection guide.
5. Provide fail-closed local-only behavior rather than silently uploading pixels on local failure.
6. Let users select a self-hosted/BYOK endpoint if compatible with the trust model.
7. Use confidential-computing/attested server execution if the goal is to reduce trust in server operators. Apple's Private Cloud Compute is one of the strongest publicly documented product patterns for enforceable cloud-AI privacy, although it is a different hardware/software stack and cannot simply be copied onto an ordinary service. [Apple PCC](https://security.apple.com/blog/private-cloud-compute/)
8. If the server truly must not see the features, ordinary split inference is the wrong primitive; investigate secure enclaves, secure multiparty computation, or homomorphic encryption and accept their current cost/complexity.

## 9. Quality evaluation

### 9.1 Why tensor similarity is not the gate

The repository has already observed that high feature cosine similarity can coexist with failed end-to-end quality. Mean tensor similarity averages away rare, high-impact token/channel errors and says nothing directly about the LLM's decision boundary.

Every client implementation should be evaluated at four levels:

1. **Preprocessing parity:** compare patches/grid/positions produced from identical source bytes.
2. **Feature diagnostics:** relative error, per-token cosine percentiles, token norms, channel-wise error, non-finite values, and outlier behavior.
3. **Model decisions:** next-token logit divergence, generated-answer agreement, and deterministic task-level comparisons against the reference.
4. **Application quality:** ChartQA plus the actual product workload, including OCR/documents separately from general visual questions.

The reference should be the qualified native/server model composition, not an isolated FP32 PyTorch tower if production uses different rounding and a server projector.

### 9.2 Required test sets

- Natural photos across lighting, aspect ratios, and image sizes.
- Documents, charts, screenshots, fine text, and small objects.
- Black, white, overexposed, underexposed, corrupt, and unusually large inputs.
- Repeated questions on the same image.
- Images known to trigger high-norm/outlier behavior.
- A fixed regression corpus with expected reference features/logits/answers tied to model and processor revisions.

### 9.3 Wire conversion

The current server contract consumes BF16. Browsers do not expose BF16 as a normal JavaScript/WebGPU storage type. A browser implementation should choose and test one explicit conversion route, for example output FP32 from ORT and perform one round-to-nearest-even FP32-to-BF16 conversion before serialization. Do not accidentally compute FP16, copy to FP32, then truncate to BF16 without documenting the double-rounding path.

## 10. Performance and reliability evaluation

Measure complete user-visible phases rather than one `session.run()`:

| Phase | Required metrics |
|---|---|
| Capability probe | pass/fail reason and elapsed time |
| Model download | bytes, throughput, retries, cache hit/miss |
| Verification | digest time and failures |
| Session creation | CPU/GPU time and memory |
| Warmup | first-run shader compilation/capture time |
| Preprocessing | decode, resize, normalize, patch construction |
| Encoder | p50/p90/p95/p99, not only average |
| Serialization | conversion, compression, payload bytes |
| Network | upload, server queue, response TTFT |
| Recovery | device loss, cancellation, sleep/wake, cache eviction |

Client telemetry should record hardware/runtime metadata and timings only with consent. It should not include pixels, features, prompts, model outputs, or raw exception buffers by default.

Energy and thermal behavior matter for repeated video frames even if one image is fast. A desktop M4 single-image result cannot establish battery/thermal suitability on a MacBook, phone, or low-end integrated GPU.

## 11. Three proper implementation options

The research supports three defensible implementations. The correct choice depends on reach, quality, and privacy requirements; the evidence does not collapse them into one universal answer.

### Option A: Browser client

Use when zero install is important and Chrome/Edge desktop support is acceptable.

```text
immutable ONNX shards -> Cache/OPFS -> Worker + ORT Web WebGPU
  -> exact preprocessing -> pre-projector features
  -> BF16 binary protocol -> validation gateway -> H200 projector/LLM
```

Required proof before product use:

- Actual quantized artifact and node coverage.
- Browser provider assignment and memory profile.
- Composed quality parity.
- Supported device matrix and explicit unsupported routing.
- Download/cache/recovery UX.
- Clear user consent for feature transmission and any cloud fallback.

Main risks: browser/runtime coverage, model download, GPU heterogeneity, model weight exposure, and quality drift.

### Option B: Native helper

Use when Apple Silicon is the primary platform and deterministic native performance matters more than zero-install reach.

```text
signed/notarized helper -> MLX/BF16 encoder
  -> authenticated loopback bridge or native UI
  -> existing binary protocol -> H200
```

Required proof before product use:

- Installer/update/model-version lifecycle.
- Loopback threat-model review and browser permission behavior.
- Clean install/uninstall and crash recovery.
- Same composed quality and protocol tests as the current client.

Main risks: install conversion, updater security, localhost attack surface, and platform-specific maintenance.

### Option C: Full local or nearby-host VLM

Use when features must not leave user-controlled hardware or network availability is poor.

```text
browser/native/sensor -> user-owned model host -> local answer
```

This can use a smaller VLM, a nearby laptop, or a self-hosted model. It changes the quality/cost model because the H200 LLM no longer answers every request. OpenGlass and Ollama are relevant precedents.

Main risks: weaker local model, larger local download/compute, fragmented hardware support, and difficult support burden.

## 12. Common failure modes to avoid

- Calling embeddings "private" without testing semantic recovery.
- Treating WebGPU API availability as ORT Web support.
- Allowing silent WebGPU-to-WASM fallback during qualification.
- Shipping one mutable model URL with no manifest, digest, or rollback.
- Assuming browser cache is permanent.
- Loading/inferencing on the main UI thread.
- Benchmarking only warm inference on one developer laptop.
- Reporting model bytes but not peak process/GPU memory.
- Comparing feature bytes with an arbitrary JPEG rather than the same images and task-quality target.
- Using base64/JSON float arrays for production tensors.
- Exposing raw vLLM embedding ingestion to untrusted users.
- Trusting CORS as authentication for a localhost helper.
- Allowing fallback to upload pixels without explicit mode/consent.
- Gating quantization on cosine similarity alone.
- Evaluating general VQA and assuming OCR/chart behavior follows.
- Updating client weights without a compatible server revision still available.
- Logging tensors in traces or crash reports.
- Hiding the model download, cloud transition, or current execution mode from users.

## 13. What a complete implementation should contain

### Client

- Exact processor implementation and golden test vectors.
- Capability probe with reasoned failures.
- Content-addressed artifact manager with progress, verification, cancellation, retry, rollback, and deletion.
- Worker-isolated runtime.
- Strict local/cloud routing semantics.
- Protocol encoder with model/processor/split revisions.
- Cancellation, timeout, and device-loss recovery.
- No sensitive telemetry by default.

### Gateway

- Authentication, authorization, TLS, replay protection, and rate limits.
- Streaming-safe body limits and bounded decompression.
- Strict schema/shape/dtype/value validation before GPU allocation.
- Private inference-worker interface.
- Revision compatibility table and staged retirement.
- Metrics for sizes, timing, failures, and model revision without tensor contents.

### Evaluation

- Golden preprocessing and feature tests.
- Composed answer/logit/task gates.
- Browser/native/server comparison on the same corpus.
- Device/browser matrix with p50/p95/p99 and memory.
- Network/cache/update/recovery fault injection.
- Leakage evaluation: labels, captions, OCR, attributes, nearest neighbors, and image inversion.
- Adversarial/malformed tensor and safety-bypass tests.

### Product and policy

- Data-flow page stating exactly what leaves the device.
- Per-request execution-mode indicator.
- Retention/training/logging policy for features and prompts.
- Explicit fallback behavior and local-only fail-closed mode.
- Model deletion and request-history controls.
- Incident response that classifies visual features as sensitive data.

## 14. Findings specific to this repository

1. The neural split itself is not the speculative part. It has direct precedent and this repository has already proven a compatible client/server boundary.
2. The native M4 path is the only qualified local implementation today. It has measured latency and composed quality evidence.
3. The browser path has passed exportability, not deployment. The largest unknowns are quantized quality, WebGPU provider coverage, memory, and warm/cold latency.
4. The current 405 KB feature payload is smaller than many published post-projector tensors but still larger than the brief's 50-70 KB compressed-frame estimate. Browser weight download, not per-request upload, is presently the larger distribution problem.
5. Query-aware pruning such as LAST is interesting if H200 prefill or wire cost becomes important, but it adds an edge proxy model and makes image features query-specific. It is not a free follow-up to the current encoder.
6. TOFC is the most directly relevant learned-codec baseline found if reducing wire bytes becomes a research goal, but it requires training and a matching server decoder.
7. The privacy claim should remain "raw-image minimization." CapRecover directly attacks conventional encoder outputs, while newer reconstruction work shows that deeper hidden states also leak.
8. Supporting arbitrary client tensors requires a validation gateway even if vLLM or TensorRT-LLM accepts embeddings internally.
9. A native helper and browser client can share the wire contract and server gateway without sharing the inference runtime or artifact format.
10. No implementation should be selected solely from literature numbers. The next meaningful evidence must come from this exact model, processor, quality corpus, M4/browser stack, and H200 composition.

## 15. Open questions

- Can an ORT `MatMulNBits` version of the exact tower pass composed ChartQA and answer-agreement gates?
- What fraction of graph compute remains outside WebGPU after actual session initialization?
- What are peak browser CPU and GPU memory during download, session creation, and inference?
- Does the browser's FP16 accumulation reproduce qualified BF16 behavior closely enough at the final answers?
- Which Apple Silicon generations and Chromium versions meet an acceptable p95 latency?
- What leakage can a CapRecover/RASR-style attacker extract from the exact pooled `[264,768]` tensor?
- Is 405 KB/request material in the expected product/network profile, or is model delivery the only size problem worth solving now?
- Do users need browser reach, native reliability, or full local-only answers? This is a product requirement, not a paper result.
- If native is used, is a localhost bridge acceptable, or should the UI be native/extension-based?
- If confidential H200 execution is desired, can the current vLLM/gateway stack run in a verifiably attested configuration without unacceptable overhead?

## 16. Source index

### Browser runtimes and delivery

- [WebLLM paper](https://arxiv.org/abs/2412.15803)
- [WebLLM documentation](https://webllm.mlc.ai/docs/)
- [WebLLM cache backends](https://github.com/mlc-ai/web-llm/tree/main/examples/cache-usage)
- [LlamaWeb paper](https://arxiv.org/abs/2605.20706)
- [ONNX Runtime Web support matrix](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)
- [ONNX Runtime WebGPU execution provider](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)
- [ONNX Runtime large-model guidance](https://onnxruntime.ai/docs/tutorials/web/large-models.html)
- [ONNX Runtime Web session options](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html)
- [Transformers.js WebGPU guide](https://huggingface.co/docs/transformers.js/en/guides/webgpu)
- [Transformers.js custom model configuration](https://huggingface.co/docs/transformers.js/en/custom_usage)
- [ONNX external data specification](https://github.com/onnx/onnx/blob/main/docs/ExternalData.md)
- [MDN storage quotas and eviction](https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria)
- [WebKit storage policy](https://webkit.org/blog/14403/updates-to-storage-policy/)
- [Chrome built-in model management](https://developer.chrome.com/docs/ai/understand-built-in-model-management)

### Split and edge/cloud systems

- [Distributed VLMs](https://par.nsf.gov/biblio/10639785-distributed-vlms-efficient-vision-language-processing-through-cloud-edge-collaboration)
- [TOFC](https://arxiv.org/abs/2503.12926) and [partial code](https://github.com/asdLeaving/TOFC)
- [Progressive Semantic Communication](https://arxiv.org/abs/2604.26508)
- [Co-VStream](https://arxiv.org/abs/2606.22804)
- [LAST](https://arxiv.org/abs/2607.27952)
- [Token Communication for MLLMs](https://arxiv.org/abs/2608.07279)
- [OpenGlass](https://arxiv.org/abs/2607.03213) and [code](https://github.com/OpenSQZ/OpenGlass)
- [INAR-VL](https://arxiv.org/abs/2605.18853)
- [edgeVLM](https://arxiv.org/abs/2508.12638)
- [ModServe](https://arxiv.org/abs/2502.00937)
- [VisionZip](https://arxiv.org/abs/2412.04467)
- [FastVLM](https://arxiv.org/abs/2412.13303)

### Serving infrastructure

- [vLLM multimodal embedding inputs](https://docs.vllm.ai/en/latest/features/multimodal_inputs/)
- [vLLM disaggregated encoder](https://docs.vllm.ai/en/latest/features/disagg_encoder/)
- [llm-d multimodal encoder disaggregation](https://github.com/llm-d/llm-d/tree/main/guides/multimodal-serving/e-disaggregation)
- [TensorRT-LLM external multimodal embeddings](https://github.com/NVIDIA/TensorRT-LLM/pull/6263)
- [TensorRT-LLM standalone multimodal encoder](https://github.com/NVIDIA/TensorRT-LLM/pull/6743)
- [NVIDIA Dynamo encoder disaggregation](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/features/multimodal/encoder-disaggregation.md)
- [Modal vLLM deployment](https://modal.com/blog/how-to-deploy-vllm)
- [Baseten custom model server](https://docs.baseten.co/development/model/custom-server)

### Privacy and security

- [CapRecover](https://arxiv.org/abs/2507.22828)
- [The Vision Encoder as a Privacy Boundary](https://arxiv.org/abs/2606.14783)
- [Image Prompt Reconstruction Attacks](https://arxiv.org/abs/2606.18710)
- [Inverting the Hidden](https://arxiv.org/abs/2608.01020)
- [Vision Token Manipulation Attacks](https://arxiv.org/abs/2607.02819)
- [Apple Private Cloud Compute](https://security.apple.com/blog/private-cloud-compute/)
- [Screenpipe security architecture](https://screenpipe.com/security/architecture)
- [Chrome Local Network Access](https://developer.chrome.com/blog/local-network-access)

### Native distribution

- [Ollama FAQ](https://docs.ollama.com/faq)
- [Apple notarization](https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution)
- [Sparkle update framework](https://sparkle-project.org/documentation/)
