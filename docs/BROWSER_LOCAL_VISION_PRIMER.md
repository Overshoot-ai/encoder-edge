# Half the Model Lives in Your Browser

## How we will package and serve a local vision encoder without pretending the browser is an H200

Research checked: 2026-08-10

Open the Overshoot Playground. Turn on your webcam. Ask:

> What is on my desk?

Today, answering that question means getting the image to a machine that can run a vision-language model. The normal cloud approach is straightforward:

```text
your image -> our server -> vision model -> language model -> answer
```

We are building a different route:

```text
your image -> local vision encoder -> visual features -> our server -> language model -> answer
```

In this local mode, the raw image stays on your machine. The server receives visual features instead.

That sentence sounds simple. Shipping it means answering three less-simple questions:

1. How do we put a neural network in a browser?
2. How do we update it without redeploying the entire Playground?
3. How do we make it reliable across browsers and GPUs that behave very differently?

This is the short technical walkthrough.

## The 30-second version

We package the system as three independently versioned things:

```text
1. Runtime
   The Playground, image processor, ONNX Runtime Web, downloader, and WebGPU Worker.

2. Encoder
   A downloadable Gemma vision model package, cached by the browser.

3. Rest of the model
   The projector and language model, kept on an H200.
```

The proposed deployment serves them from three places:

```text
playground.overshoot.ai  -> application/runtime
models.overshoot.ai      -> vision encoder package (proposed model CDN)
api.overshoot.ai         -> authenticated gateway to the H200
```

The user downloads the encoder once. Each image is then encoded locally into roughly 405 KB of visual features. Those features go to the H200, which runs the projector and language model and streams the answer back.

The packaging pattern is established. Browser execution of our exact model is not fully qualified yet. That distinction matters, and we will return to it.

## First: what are we actually splitting?

A vision-language model is not one indivisible blob. At a high level, ours looks like this:

```text
image
  -> image preprocessing
  -> vision encoder
  -> visual features [264, 768]
  -> projector block
       -> RMSNorm
       -> linear projection 768 -> 2560
  -> language model
  -> answer
```

The vision encoder recognizes visual patterns. It turns pixels into a sequence of numbers the rest of Gemma understands.

The projector block translates those 768-wide vision features into the 2,560-wide representation expected by the language model. The language model then reasons over the image and the user's question.

Our split is immediately before the projector:

```text
BROWSER                                      H200

image                                        [264, 768] features
  -> preprocessing                              -> RMSNorm
  -> vision encoder                             -> 768-to-2560 projection
  -> [264, 768] features -------------------->  -> language model
                                                  -> answer
```

Why there?

Because `[264,768]` in BF16 is:

```text
264 tokens * 768 values * 2 bytes = 405,504 bytes
```

After the projector it becomes:

```text
264 tokens * 2,560 values * 2 bytes = 1,351,680 bytes
```

Moving the projector to the server makes the request about 70% smaller. RMSNorm stays with it because Gemma implements both operations as one projector block, and normalizing on the server avoids another cross-runtime numerical difference.

## Three packages, not three downloads

Only two things reach the browser: the runtime and the encoder. The H200 model never does.

### Package one: the runtime

The runtime is ordinary web application code:

- React UI.
- Video and image capture.
- Exact Gemma image preprocessing.
- ONNX Runtime Web.
- A dedicated Web Worker.
- WebGPU capability checks.
- Model download and cache management.
- BF16 feature serialization.
- The client for the Overshoot streaming API.

This ships with the Playground through Vercel, just like its current JavaScript and CSS.

The runtime is the player. It is not the movie.

## Six browser concepts worth understanding first

The runtime list hides several different layers. They are related, but they are not synonyms and they are not additional top-level packages.

| Term | What it is | Which package owns it? |
|---|---|---|
| ONNX | A file format describing a neural-network graph and its tensors | Encoder package |
| ONNX Runtime Web | JavaScript/WASM software that reads and executes ONNX graphs | Runtime package |
| Web Worker | A background browser execution context | Runtime package |
| WebGPU | The browser API used to submit work to the local GPU | Browser/platform, accessed by the runtime |
| External `.data` | Optional files holding large ONNX weight tensors | Encoder package |
| Manifest | Our proposed package/version/checksum contract around the ONNX files | Encoder package, interpreted by the runtime and server |

### ONNX and ONNX Runtime Web are different things

An ONNX model is data. It says:

```text
input tensor
  -> matrix multiplication using weight tensor A
  -> normalization
  -> attention
  -> pooling
  -> output tensor
```

It contains the graph, input/output definitions, operator attributes, and either the model weights themselves or references to external weight data.

[ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html) is the software that opens that file in a browser. It validates the graph, chooses an implementation for each operator, allocates tensors, submits GPU work, and returns the output.

```text
vision.onnx                   ONNX Runtime Web
"run these operations"  ->  "I know how to run them here"
```

ONNX is comparable to a program file. ONNX Runtime Web is the execution engine. Exporting a valid ONNX graph does not prove that every browser can execute it quickly; the runtime still needs a supported GPU implementation for the important operators.

### A Web Worker is a browser background thread

JavaScript normally runs UI code on the browser's main thread. If that thread spends 800 ms preparing and launching a model, scrolling, typing, and button feedback can freeze for the same 800 ms.

A [Web Worker](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers) has a separate JavaScript execution context:

```text
MAIN THREAD                          VISION WORKER

render React UI                     load ONNX Runtime
handle buttons       postMessage    create model session
show progress       ------------->  preprocess frame
stream answer       <-------------  run encoder
                                     return features
```

Workers communicate by messages. They do not share ordinary JavaScript objects with the page. Large values should be transferred rather than copied when possible. A Worker does not make the GPU faster; it prevents CPU-side model work from blocking the interface.

The Worker is code inside the runtime package. It is not a fourth package.

### WebGPU capability checks are more than `navigator.gpu`

[WebGPU](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API) is the browser API that exposes GPU compute and graphics through a common interface over Metal, Direct3D 12, or Vulkan.

A basic check looks like this:

```javascript
if (!navigator.gpu) throw new Error("WebGPU is unavailable");

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error("No usable GPU adapter");

if (!adapter.features.has("shader-f16")) {
  throw new Error("This GPU/browser cannot run the FP16 model");
}
```

That is only the first gate. A real qualification check asks:

```text
1. Does the browser expose WebGPU?
2. Can it return a hardware adapter?
3. Does the adapter expose shader-f16 and sufficient limits?
4. Does ONNX Runtime Web officially support this browser?
5. Can ORT create a session for this exact model?
6. Are all expensive operators assigned to WebGPU?
7. Does a golden input produce an acceptable output quickly enough?
```

A machine can pass step one and fail step five. This is why "WebGPU available" and "our encoder supported" are different claims.

### External `.data` files are weight storage, not extra models

An ONNX file is serialized with Protocol Buffers. It can store each weight tensor directly inside the `.onnx` file. That is what our current measured 308 MB export does.

ONNX also supports [external data](https://github.com/onnx/onnx/blob/main/docs/ExternalData.md). In that form, the graph contains a record like:

```text
weight name: vision.layers.0.attention.q.weight
location: weights-00.data
offset: 8388608
length: 1179648
```

The bytes live in `weights-00.data`, while the graph explains where to find them. Splitting weight storage into files does not split the neural network into separate models. It is still one logical encoder and one ONNX session.

Why use external files or shards?

- A failed 16 MB download is cheaper to retry than a failed 300 MB download.
- Progress can be reported per file.
- CDN caching and verification happen in bounded pieces.
- ONNX itself has a 2 GB protobuf-file limit for very large models.

Our current export is one `.onnx` file. External shards are a proposed distribution format, not something already produced by the export harness.

### The manifest is our package contract, not an ONNX feature

The ONNX graph can identify its external weight filenames, offsets, and lengths. It does not know:

- Which image processor revision the Playground must use.
- Which browser runtime version we qualified.
- Which server projector revision accepts the output.
- Which protocol revision to put on the wire.
- The CDN byte count and SHA-256 digest for every file.
- Which browser/GPU features are required.

That is why we propose a separate `manifest.json`:

```json
{
  "schema_version": 1,
  "model_id": "gemma-4-e4b-vision",
  "model_revision": "sha256:encoder-package-hash",
  "processor_revision": "sha256:processor-hash",
  "split_point": "vision.pre_projector",
  "protocol_version": 3,
  "runtime": {
    "name": "onnxruntime-web",
    "minimum_version": "1.x"
  },
  "input": {
    "shape": [1, 2376, 768],
    "dtype": "float16"
  },
  "output": {
    "shape": [1, 264, 768],
    "wire_dtype": "bfloat16"
  },
  "server_compatibility": ["gemma-4-e4b-server-r7"],
  "required_features": ["webgpu", "shader-f16"],
  "files": [
    {
      "path": "graph.onnx",
      "bytes": 1245184,
      "sha256": "..."
    },
    {
      "path": "weights-00.data",
      "bytes": 16777216,
      "sha256": "..."
    }
  ]
}
```

This exact manifest does not exist yet. It is the proposed receipt for one encoder release. The downloader verifies the receipt; ONNX Runtime loads the graph and weights; the API checks that the declared encoder revision is compatible with the server projector.

An encoder package may contain ten files and still be one of the three top-level packages. "Package" here means one versioned logical release, not one physical file.

### An ImageBitmap is a transferable decoded image

The browser's `<video>` element owns a decoded current frame. We need to move that frame to the Worker without first turning it into a base64 string or a new JPEG.

[`createImageBitmap()`](https://developer.mozilla.org/en-US/docs/Web/API/Window/createImageBitmap) creates an [`ImageBitmap`](https://developer.mozilla.org/en-US/docs/Web/API/ImageBitmap): a browser-managed, decoded image object suitable for drawing and transferring.

```javascript
const bitmap = await createImageBitmap(videoElement);

worker.postMessage(
  { type: "encode", bitmap },
  [bitmap],
);
```

The second argument transfers ownership to the Worker instead of cloning all pixel data. The Worker can draw it onto an [`OffscreenCanvas`](https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas), perform preprocessing, and then release it with `bitmap.close()`.

```text
video decoder -> ImageBitmap -> Worker -> OffscreenCanvas -> model input tensor
```

An ImageBitmap is not the encoder input. It is an efficient vehicle for getting a decoded frame to the code that will construct the encoder input.

Canvas resizing is also not automatically identical to our Python/PIL processor. Matching resize, crop, normalization, patch order, and rounding remains a correctness task.

### Suggested prerequisite reading order

1. [ONNX Concepts](https://onnx.ai/onnx/intro/concepts.html): graph, nodes, tensors, initializers, and operators.
2. [ONNX Runtime Web: Get Started](https://onnxruntime.ai/docs/get-started/with-javascript/web.html): loading and running a model in JavaScript.
3. [ORT Web: Working with Large Models](https://onnxruntime.ai/docs/tutorials/web/large-models.html): external data, browser limits, and caching.
4. [Using Web Workers](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers): browser threading and message passing.
5. [Transferable Objects](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Transferable_objects): moving large buffers without copying.
6. [WebGPU API](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API): adapters, devices, features, and device loss.
7. [ImageBitmap](https://developer.mozilla.org/en-US/docs/Web/API/ImageBitmap) and [OffscreenCanvas](https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas): moving and processing frames outside the UI thread.
8. [Browser storage quotas and eviction](https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria): why downloaded models must be treated as an evictable cache.

### Package two: the encoder

The encoder is the large, separately downloaded model package:

```text
gemma-4-e4b-vision/<content-hash>/
├── manifest.json
├── graph.onnx
├── weights-00.data
├── weights-01.data
├── processor.json
└── checksums
```

The `.onnx` file describes the model's computation: matrix multiplications, attention, normalization, pooling, and the order in which they run. ONNX Runtime Web is the program that executes that description.

Large weights can be stored in external `.data` shards. The manifest says exactly which graph, weights, processor, runtime, and server revision belong together.

The current FP16 ONNX tower is 308,190,557 bytes. That is useful as a correctness artifact, but it is not the intended first-visit download.

The Gemma E4B vision tower has roughly 150 million parameters. The ideal bit-count math is:

```text
150M weights * 16 bits ~= 300 MB at FP16/BF16
150M weights *  4 bits ~=  75 MB at ideal packed INT4
```

Real weight-only INT4 is larger than 75 MB because each block needs scales and possibly zero points, some tensors remain FP16, and the graph has non-weight data. Based on that math, **85-100 MB raw is a reasonable experiment-planning hypothesis**.

It is not a measured package size. We have not produced the INT4 graph, counted converted bytes, or measured HTTP compression. The Playground must not display a 60 MB or 82 MB promise until the generated manifest contains the real artifact byte count.

### Package three: the rest of Gemma

The server package contains:

- The RMSNorm/projector block.
- The language model.
- vLLM and the H200 runtime.
- Compatibility support for active client encoder revisions.

This remains inside Overshoot's infrastructure. It is not browser code and is not downloadable by users.

## What happens on the first visit?

Suppose the user selects:

```text
Gemma 4 E4B
Local vision + cloud reasoning
```

The Playground does not immediately throw 100 MB at their connection. It first asks the browser what it can actually do:

```text
Does WebGPU exist?
Does this runtime support the browser?
Does the GPU support the required FP16 feature?
Can a small qualification graph run correctly?
Is there enough storage for the model package?
```

If the machine qualifies, the user sees an explicit download:

```text
Download the local vision model

This model processes images on your device.
Expected download: calculated from the generated manifest
Storage: this browser

[Download] [Use cloud mode]
```

The actual byte count comes from the manifest. No mystery spinner.

The browser downloads immutable, content-addressed shards from the model CDN. Each shard is checked against its SHA-256 digest before the model becomes active.

```text
download revision B
  -> verify every shard
  -> initialize WebGPU
  -> run a golden self-test
  -> mark revision B active
  -> eventually remove revision A
```

If a download fails halfway through, revision A still works. If the cache is later evicted, the Playground asks to download the model again. Browser storage is a cache, not a sacred vault.

## What happens when the user asks a question?

The existing Playground already has a video element containing the current source. For local vision mode, the request becomes:

```text
1. Capture the newest frame as an ImageBitmap.
2. Transfer it to the vision Worker.
3. Resize, normalize, and construct Gemma's patch tensor.
4. Run the ONNX encoder with WebGPU.
5. Convert the output to the BF16 wire format.
6. Send the question and approximately 405 KB of features.
7. Stream the answer back with SSE.
```

The Worker is important. Running a 150M-parameter vision tower on the browser's main thread would make the interface appear frozen. A Worker gives the model its own execution context while the UI continues to animate, accept cancellation, and show progress.

The request itself is binary:

```text
4 bytes   protocol magic
4 bytes   metadata length
8 bytes   tensor length
N bytes   JSON metadata
M bytes   raw BF16 features
```

The metadata identifies the model, processor, split point, shape, question, and active revision. The tensor stays binary. We do not turn 405 KB into a giant JSON list of decimal numbers.

The public request goes to:

```text
POST https://api.overshoot.ai/v1/chat/completions
Content-Type: application/x-cross-device-gemma
Authorization: Bearer <user API key>
Accept: text/event-stream
```

The public API authenticates the user, checks limits and revisions, and forwards the request over a private network to the H200 gateway. The browser never receives the H200's private address or credentials.

The response is the same kind of SSE token stream the Playground already knows how to render.

## One Playground-specific trap

Today, the Playground publishes webcam, screen, and uploaded-video tracks through LiveKit. Hosted models then refer to those frames with an `ovs://streams/...` URL.

That is correct for cloud mode:

```text
camera -> LiveKit -> Overshoot stream -> cloud vision model
```

It is not correct for privacy-oriented local vision mode. If we publish the track to LiveKit first, the raw frames have already left the browser.

Local mode needs a separate media path:

```text
camera/screen/file -> local video element -> local encoder -> features only
```

So the Playground will have two explicit modes rather than one path with a hidden optimization:

```text
Cloud
Raw frames are sent to Overshoot.

Local vision + cloud reasoning
Raw frames stay in the browser; visual features are sent to Overshoot.
```

RTSP is different. Browsers generally cannot consume RTSP directly, so Overshoot must pull that stream on the server. We cannot honestly label that source as raw-frame-local.

## Has anyone done this before?

Nobody we found has publicly shipped this exact consumer product end to end. But every major piece has precedent.

### WebLLM: models as web application assets

[WebLLM](https://arxiv.org/abs/2412.15803) downloads quantized model shards, caches them in the browser, executes them with WebGPU in a Worker, and exposes a streaming OpenAI-style API.

On an M3 Max, its paper measured about 71-80% of native MLC decode throughput for two 4-bit language models. That does not predict our encoder latency, but it establishes the packaging pattern: compiled runtime, remote model artifacts, local cache, Worker execution, and streaming output.

### LlamaWeb: the browser is not one computer

[LlamaWeb](https://arxiv.org/abs/2605.20706) brought llama.cpp/GGUF models to WebGPU and tested 10 models on 16 GPUs from eight vendors.

It statically plans memory and streams weights from the browser's Origin Private File System into GPU buffers. It also demonstrates the uncomfortable truth of browser ML: the same code can perform very differently across Chrome, Safari, Firefox, macOS, Windows, Linux, and mobile GPUs.

That is why we qualify device/browser combinations instead of checking only `navigator.gpu`.

### ONNX Runtime Web and Transformers.js: portable model execution

[ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html) and [Transformers.js](https://huggingface.co/docs/transformers.js/en/guides/webgpu) already run image, speech, and transformer models in browsers through WebGPU and WebAssembly.

ORT's documented WebGPU execution provider currently targets Chrome and Edge, not Safari or Firefox. The existence of WebGPU in a browser does not imply that ORT supports that browser.

### Ollama: keep the engine and weights separate

[Ollama](https://docs.ollama.com/faq) packages a native model engine separately from downloaded model weights. The application updates on one cadence; models live in a user data directory and update on another.

We are applying the same separation to the web:

```text
Playground/runtime updates frequently
Encoder package updates deliberately
H200 model updates independently, with compatibility overlap
```

### Distributed VLMs: run the vision side at the edge

[Distributed VLMs](https://par.nsf.gov/biblio/10639785-distributed-vlms-efficient-vision-language-processing-through-cloud-edge-collaboration) ran the complete vision encoder and projector on Jetson devices and sent visual tensors to a server running the language model.

It reported up to 33.54% higher sustained throughput by overlapping edge encoding with server generation. That is a throughput result from a research prototype, not a claim that every individual request became 33.54% faster.

### Progressive Semantic Communication: a physical split testbed

[Progressive Semantic Communication](https://arxiv.org/abs/2604.26508) put a 4-bit SmolVLM encoder on an NXP i.MX95 and the language model on an RTX 2080 Super. At a constrained 1 Mbps uplink, the full split path took 6.94 seconds versus 9.32 seconds for its cloud-only path.

It proves that a complete vision encoder can run on genuinely constrained edge hardware and communicate with a separate GPU server. Its measured representation used inefficient JSON and its public code was not usable when checked, so it is evidence, not a package we can drop in.

### TOFC: compress the features, not the image

[TOFC](https://arxiv.org/abs/2503.12926) ran a complete vision encoder on a Jetson AGX Orin, merged visual features, entropy-coded them, and decoded them before the server projector and LLM.

It reported up to 52% fewer transmitted bytes and 63% lower system latency than a learned image codec at matched task performance. It also required model-specific training, and its public repository omitted the complete two-device transport path.

### OpenGlass: nearby user-owned compute

[OpenGlass](https://arxiv.org/abs/2607.03213) built open-source camera glasses that send JPEG frames over local Wi-Fi to a nearby laptop running a complete local VLM. It reported 993 ms median query-ready-to-audio latency with resized frames.

That is not our neural split, but it demonstrates another useful product lesson: local-first systems need stage-level timings, p95 latency, timeouts, abstention, and explicit handling of stale frames. A good median is not enough.

## So, how reliable will ours be?

The honest answer has three parts.

### The three-part packaging can be delivered reliably

"Three packages" describes the system boundary:

```text
runtime | encoder | server model
```

"Reliable packaging" describes how each one is versioned, downloaded, verified, activated, and rolled back. It is a property of those three packages, not a fourth package.

For example, the encoder remains one logical package even if it contains a manifest, graph, processor configuration, and six weight shards.

Content-addressed files, CDN delivery, checksums, retryable/resumable downloads, browser caches, version manifests, and atomic activation are established web distribution techniques.

We know how to avoid half-installed models and incompatible updates.

### The server path is already substantially proven

This repository already has:

- A working pre-projector split.
- A binary BF16 protocol.
- A gateway that validates and forwards features to vLLM.
- A server-side projector.
- Streaming answers.
- A qualified native M4 encoder path around 319-352 ms p50.

The native path demonstrates that the model split itself works.

### The browser encoder is still a qualification project

We have exported the complete fixed-shape tower to ONNX and executed it with ONNX Runtime on CPU. The artifact produces the expected `[264,768]` boundary.

We have not yet proven:

- Actual WebGPU latency on the base M4.
- Peak browser and GPU memory.
- Complete WebGPU operator assignment without WASM fallback.
- INT4 artifact size and node coverage.
- INT4 composed VLM quality.
- Exact browser preprocessing parity.
- Device-loss, sleep/wake, background-tab, and cache-eviction recovery.

So the reliability table is:

| Layer | Status |
|---|---|
| Three-package architecture | Established pattern |
| CDN download, cache, checksums, rollback | Standard engineering |
| Native M4 encoder -> H200 | Qualified in this repository |
| FP16 ONNX export | Proven for one fixed shape and CPU execution |
| Browser WebGPU execution | Not yet measured |
| INT4 browser model | Not yet produced |
| Browser + H200 final-answer parity | Not yet qualified |

We should not turn the yellow and red rows green with marketing copy.

## What happens when something fails?

Reliability comes mostly from making failure boring.

### Unsupported browser

The local model is not offered. The user can choose cloud mode.

### Model download interrupted

Keep the previous revision active. Resume or retry the incomplete shards.

### Browser evicts the model

Ask to download it again. Do not pretend browser storage is permanent.

### WebGPU device is lost

Destroy the old session, request a new device, rebuild resources, and retry once. If local-only mode is enabled, fail closed after that.

### Local encoder is slower than the requested frame rate

Do not queue every frame. Keep one request in flight, discard stale frames, and encode the newest frame when ready.

```text
encoder busy
  -> frame 101 arrives: remember it
  -> frame 102 arrives: replace 101
  -> encoder ready: process 102
```

An answer about the newest frame is useful. An answer about a perfectly preserved queue from ten seconds ago is not.

### Local mode fails during a request

Do not silently upload the raw image. Either ask the user before switching modes or fail closed when they selected local-only behavior.

## What about privacy?

The split keeps raw images out of the normal server path. That removes raw pixels and image metadata from ingress, storage, logs, and debugging systems.

The features are still sensitive.

[CapRecover](https://arxiv.org/abs/2507.22828) recovered labels and captions from conventional encoder features without reconstructing pixels. Other work has reconstructed image semantics and, for some model architectures, readable text from intermediate visual states.

The accurate claim is:

> Raw images are processed on your device. The Playground sends visual features to Overshoot so the cloud language model can answer. Those features can still contain sensitive information about the image.

Not:

> Your data never leaves your device.

Features are data. They are just a more deliberate boundary than raw images.

## What the finished Playground experience looks like

```text
Open Playground
  -> choose webcam, screen, or local video
  -> select "Local vision + cloud reasoning"
  -> download encoder once
  -> wait for "Local vision ready"
  -> ask a question
  -> see "Encoding locally"
  -> see "Sending 405 KB of visual features"
  -> receive a streamed answer
```

Every request shows its route. The model can be deleted from browser storage. Cloud fallback is explicit. Unsupported sources are labeled honestly.

The implementation is not one enormous web bundle and it is not a tiny native application hiding behind the page. It is a normal web app, a separately managed model package, and a private H200 service joined by a strict feature contract.

```text
small code, large local encoder, enormous remote reasoner
```

That is the whole trick.

## Further reading

- [Local and Split Vision Inference: What Exists and How to Build It Properly](./LOCAL_VISION_INFERENCE_LANDSCAPE.md)
- [Browser WebGPU Export Feasibility](./BROWSER_WEBGPU_POC.md)
- [Cross-device VLM research survey](./STEP_1_RESEARCH.md)
- [WebLLM](https://arxiv.org/abs/2412.15803)
- [LlamaWeb](https://arxiv.org/abs/2605.20706)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)
- [Distributed VLMs](https://par.nsf.gov/biblio/10639785-distributed-vlms-efficient-vision-language-processing-through-cloud-edge-collaboration)
- [Progressive Semantic Communication](https://arxiv.org/abs/2604.26508)
- [TOFC](https://arxiv.org/abs/2503.12926)
- [OpenGlass](https://arxiv.org/abs/2607.03213)
- [CapRecover](https://arxiv.org/abs/2507.22828)
