# Browser-Local Vision, Cloud Language

## End-to-end system design for packaging, serving, and operating the split model

Status: proposed production architecture

Research and measurements checked: 2026-08-10

## 1. Purpose

The system processes an image with the Gemma vision encoder in the user's browser, then sends visual features to an H200 that runs the projector and language model.

```text
USER DEVICE                                  OVERSHOOT

image                                        visual features
  -> image processor                           -> RMSNorm
  -> vision encoder                            -> 768-to-2560 projector
  -> [264, 768] features ------------------->  -> language model
                                                   -> streamed answer
```

The design has three goals:

1. Raw images do not enter the normal server inference path.
2. The browser downloads only the vision encoder, not the complete language model.
3. The encoder, browser runtime, and server model can update independently under an explicit compatibility contract.

The design does not claim that visual features are anonymous or semantically private. The server receives enough visual information to answer the question.

## 2. System at a glance

The deployment consists of three logical packages:

| Package | Runs where | Contains |
|---|---|---|
| Client runtime | Browser | Application code, image processor, ONNX Runtime Web, Worker, WebGPU qualification, downloader, cache manager, wire encoder, API client |
| Vision encoder | Browser, downloaded separately | ONNX graph, weight data, processor configuration, manifest, checksums, golden test metadata |
| Server model | H200 infrastructure | RMSNorm/projector block, Gemma language model, vLLM, binary gateway, compatibility table |

Three packages does not mean three files. One logical encoder release may contain a manifest, graph, processor file, and several weight files.

The proposed hosting layout is:

```text
Web application host/CDN
  -> client runtime

Model object storage/CDN
  -> immutable encoder releases

Public authenticated API
  -> private feature gateway
  -> H200 projector and language model
```

## 3. Core terminology

### Model weights

Weights are the numerical parameters learned during training. A weight tensor might be:

```text
vision.layer.0.attention.q.weight
shape: [768, 768]
dtype: BF16
```

Weights alone do not specify the order of model operations. A native runtime such as PyTorch, vLLM, or MLX supplies model implementation code separately.

### ONNX

ONNX is a portable model representation. An `.onnx` file describes:

- Inputs and outputs.
- Operators and their order.
- Tensor shapes and data types.
- Constants.
- Learned weights, called initializers, or references to external weight data.

ONNX replaces the Python model class with a machine-readable computation graph.

### ONNX Runtime Web

ONNX Runtime Web is browser-compatible software that opens and executes an ONNX graph. It validates the graph, allocates tensors, maps operators to implementations, submits GPU work, and returns outputs.

```text
ONNX graph       = what operations to perform
ONNX Runtime Web = software that performs them
WebGPU           = browser interface used to reach the GPU
GPU              = hardware executing the shaders
```

ONNX is not required merely because the client has a GPU. It is required by our selected browser runtime. A custom WebGPU engine, MLC/WebLLM, or llama.cpp/GGUF would use a different model representation.

### Web Worker

A Web Worker is a background browser execution context. It isolates preprocessing and model execution from the main UI thread.

```text
MAIN THREAD                         VISION WORKER

render UI                           load model
capture user action                 preprocess frame
show progress       postMessage     run ONNX session
stream answer      <------------->  return features
```

The Worker is part of the client runtime package. It does not make inference faster; it keeps the interface responsive.

### WebGPU qualification

The browser must pass more than a `navigator.gpu` check:

1. WebGPU is exposed.
2. A hardware adapter is available.
3. Required features such as `shader-f16` are available.
4. Adapter limits are sufficient.
5. ONNX Runtime Web supports the browser.
6. The exact model session initializes.
7. Important nodes remain on WebGPU rather than falling back to WASM.
8. A golden input produces an acceptable result within the device threshold.

### ImageBitmap

An `ImageBitmap` is a browser-managed decoded image that can be transferred to a Worker without first creating a base64 string or JPEG.

```javascript
const bitmap = await createImageBitmap(imageOrVideoElement);
worker.postMessage({ type: "encode", bitmap }, [bitmap]);
```

The Worker draws it to an `OffscreenCanvas`, performs exact preprocessing, and releases it with `bitmap.close()`.

```text
decoded image -> ImageBitmap -> Worker -> processor -> model tensor
```

An ImageBitmap is transport for decoded pixels. It is not itself the encoder input.

## 4. Package one: client runtime

The runtime is served with the web application and includes:

```text
client-runtime/
|-- application UI
|-- source capture
|-- capability probe
|-- model downloader
|-- browser storage manager
|-- vision Worker
|   |-- ONNX Runtime Web
|   |-- image processor
|   `-- feature encoder
|-- BF16 wire serializer
`-- streaming API client
```

The runtime should be small relative to the model and update through the normal web deployment process.

The runtime owns behavior, not weights. It knows how to:

- Discover compatible encoder releases.
- Download and verify one release.
- Load it into ONNX Runtime Web.
- Convert an input image into the exact model tensor.
- Serialize the output into the feature protocol.
- Recover from cache eviction, device loss, cancellation, and incompatible revisions.

## 5. Package two: vision encoder

The encoder is a separately downloaded, versioned artifact:

```text
gemma-4-e4b-vision/<revision>/
|-- manifest.json
|-- graph.onnx
|-- weights-00.data
|-- weights-01.data
|-- processor.json
`-- golden-test.json
```

The current measured export is one 308,190,557-byte FP16 `.onnx` file. It contains the fixed-shape vision tower and stops before the server projector.

External `.data` files are a proposed delivery representation. In an external-data ONNX graph, a weight initializer identifies its byte source:

```text
name: vision.layers.0.attention.q.weight
location: weights-00.data
offset: 8388608
length: 1179648
```

External files do not create multiple models. They are storage pieces for one graph and one encoder session.

### Quantization status

The tower has roughly 150 million parameters:

```text
150M weights * 16 bits ~= 300 MB at FP16/BF16
150M weights *  4 bits ~=  75 MB at ideal packed INT4
```

Real weight-only INT4 will exceed 75 MB because quantization blocks require scales and possibly zero points, some tensors remain FP16, and graph/constants add bytes.

`85-100 MB raw` is an experiment-planning hypothesis, not a measured download. No user-facing size should be published until the actual quantized artifact exists and the generated manifest records its exact bytes.

The INT4 candidate must pass:

- Converted-node and converted-byte coverage.
- Browser session initialization.
- WebGPU provider assignment.
- Feature diagnostics.
- Composed H200 answer/logit tests.
- ChartQA and product workload gates.
- Browser memory and latency gates.

## 6. The encoder manifest

The manifest is an application-level contract. It is not part of the ONNX standard and has not been implemented yet.

The ONNX graph knows which weight bytes it needs. It does not know which processor, API protocol, or server projector is compatible. The manifest binds those deployment pieces:

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
  "required_features": ["webgpu", "shader-f16"],
  "input": {
    "shape": [1, 2376, 768],
    "dtype": "float16"
  },
  "output": {
    "shape": [1, 264, 768],
    "wire_dtype": "bfloat16"
  },
  "server_compatibility": ["gemma-4-e4b-server-r7"],
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

The manifest supports four checks:

1. **Download integrity:** every file has an expected byte count and SHA-256.
2. **Runtime compatibility:** the browser has the required runtime and GPU features.
3. **Neural compatibility:** graph, processor, shape, dtype, and split point match.
4. **Server compatibility:** the H200 still serves the matching projector/model revision.

## 7. Package three: server model

The server package remains private and contains:

```text
server-model/
|-- public API integration
|-- binary feature gateway
|-- revision compatibility table
|-- RMSNorm/projector block
|-- Gemma language model
`-- vLLM/H200 runtime
```

The server receives `[264,768]` BF16 features. It applies:

```text
[264, 768]
  -> RMSNorm
  -> linear projection 768 -> 2560
  -> [264, 2560] language-model visual embeddings
  -> language model
```

The projector stays on the server because post-projector transmission is larger:

```text
pre-projector:  264 * 768  * 2 =   405,504 bytes
post-projector: 264 * 2560 * 2 = 1,351,680 bytes
```

The server must retain compatibility with active encoder revisions until those client revisions are retired.

## 8. Build and publishing process

This happens before any user opens the application.

```text
1. Load the source Gemma checkpoint.
2. Select the split before the projector block.
3. Export the fixed-shape tower to ONNX.
4. Validate the ONNX graph and reference output.
5. Optionally quantize eligible weights.
6. Count converted and unconverted bytes.
7. Run composed quality tests through the H200.
8. Produce graph, external data, processor config, and golden test.
9. Calculate file sizes and SHA-256 hashes.
10. Generate manifest.json.
11. Upload files to immutable content-addressed CDN URLs.
12. Add the encoder/server compatibility pair to the public catalog.
```

No artifact becomes discoverable until the complete package passes its gates.

## 9. First-use browser process

### Step 1: load the runtime

The user opens the HTTPS web application. The normal HTML, JavaScript, CSS, Worker code, and ONNX Runtime Web load.

### Step 2: select local vision mode

The UI explains the boundary:

```text
Raw images are processed on this device.
Visual features and your question are sent to Overshoot.
```

### Step 3: qualify the browser

The runtime checks browser/runtime support, WebGPU adapter features, storage, and the small qualification probe.

### Step 4: request download consent

The application shows the exact generated package size from the manifest:

```text
Download local vision encoder
Size: <manifest total bytes>
Storage: this browser

[Download] [Cancel]
```

### Step 5: download and verify

Files use immutable content-hashed URLs. Each file is committed only after byte-count and SHA-256 verification.

```text
download -> verify -> cache -> mark file ready
```

### Step 6: create the model session

The Worker initializes ONNX Runtime Web with WebGPU only during qualification. Material WASM fallback is not accepted silently.

### Step 7: run the golden test

The Worker runs a known input and compares output diagnostics against the package's accepted bounds. Only then does the encoder become active.

## 10. Per-request browser process

For each image/question pair:

```text
1. Acquire the current image or frame.
2. Create an ImageBitmap.
3. Transfer the ImageBitmap to the Worker.
4. Perform exact resize, crop/pad, normalization, patch, and position processing.
5. Create the `[1,2376,768]` FP16 model input.
6. Run the ONNX session through WebGPU.
7. Receive `[1,264,768]` pre-projector features.
8. Convert to the qualified BF16 wire representation.
9. Build the binary request.
10. Send it to the public authenticated API.
11. Read the streamed answer.
```

Preprocessing is part of model correctness. Browser canvas output cannot be assumed equivalent to the reference Python/PIL processor; it must pass golden preprocessing and composed answer tests.

## 11. Feature wire protocol

The current experimental protocol has this envelope:

```text
4 bytes   magic: CDG2
4 bytes   metadata length
8 bytes   tensor length
N bytes   UTF-8 JSON metadata
M bytes   contiguous BF16 tensor data
```

The production revision should bind:

- Model and weight revision.
- Processor revision.
- Split point.
- Protocol revision.
- Tensor shape, dtype, byte order, and layout.
- User question and generation limits.
- Request ID and expiry/replay metadata.

The tensor remains binary. It is not base64 or a JSON float array.

The browser request is conceptually:

```http
POST /v1/chat/completions
Authorization: Bearer <user credential>
Content-Type: application/x-cross-device-gemma
Accept: text/event-stream

<binary feature envelope>
```

## 12. Public API and H200 process

The public API is the trust boundary. The browser never contacts a private H200 address directly.

```text
BROWSER
  -> public HTTPS API
       -> authenticate and authorize
       -> enforce compressed/uncompressed limits
       -> validate protocol and revision
       -> forward through private service identity
            -> feature gateway
                 -> validate shape/dtype/finiteness
                 -> create vLLM image embeddings input
                      -> H200 projector and language model
                           -> SSE answer
```

Validation occurs before large allocation or H200 work:

- Exact content type.
- Bounded request length.
- Known protocol revision.
- Allowed model/processor/split revision tuple.
- Exact rank, width, and element count.
- BF16 byte length.
- Finite values.
- Per-user rate and concurrency limits.

Shape validation protects availability. It does not prove that a valid-looking tensor came from a genuine image.

## 13. Permissions and user consent

The browser-local route does not require a special AI or WebGPU permission prompt.

| Capability | User permission/action |
|---|---|
| WebGPU | No prompt; automatic capability check |
| Model execution | No prompt |
| Model download | Explicit application confirmation |
| Cache API/OPFS storage | Usually no prompt; persistence request may be granted, denied, or prompted depending on browser |
| Webcam | Browser/OS camera permission |
| Screen capture | Browser source picker every session; OS screen-recording permission may also apply |
| Local file | User explicitly selects each file |
| Feature transmission | Explicit application disclosure/consent |
| Public API request | No browser prompt; normal authenticated HTTPS |

Cloud fallback is a separate consent decision:

```text
Local only
  -> never upload raw pixels; fail if local execution fails

Ask before cloud fallback
  -> prompt before changing the data path

Cloud allowed
  -> raw image upload permitted under the cloud-mode disclosure
```

Local failure must not silently change the route from features to raw pixels.

## 14. Model storage and updates

Browser storage is an evictable cache, not permanent installation.

The runtime should use Cache API or OPFS and support:

- Exact download progress.
- Bounded retries.
- Partial revision recovery.
- SHA-256 verification.
- Previous-version rollback.
- User-triggered model deletion.
- Full cache-miss recovery.

Update activation is atomic:

```text
revision A remains active
  -> download revision B
  -> verify all B files
  -> initialize B
  -> run B golden test
  -> mark B active
  -> remove A later
```

The server cannot retire projector revision A until active clients using encoder A have migrated or expired.

## 15. Reliability behavior

"Three packages" is system topology. "Reliable packaging" is how those packages are verified, activated, and recovered. Reliability is not a fourth package.

| Failure | Required behavior |
|---|---|
| Unsupported browser/GPU | Do not offer local mode |
| Download interrupted | Retry incomplete files; keep prior revision active |
| Hash mismatch | Delete the bad file and retry; never initialize it |
| Storage quota exceeded | Explain the failure and offer cleanup/cloud options |
| Cache evicted | Treat as first download again |
| Session creation fails | Mark device/revision incompatible; do not silently use WASM |
| WebGPU device lost | Recreate session/resources and retry once |
| Tab backgrounded or machine sleeps | Cancel/stale the request; requalify if required |
| Local encoder slower than frame source | Keep one request in flight and process the newest frame only |
| API unavailable | Preserve local-only policy; do not upload pixels elsewhere |
| Server rejects revision | Refresh compatibility catalog; never guess a projector |

For continuous frames, use latest-frame backpressure:

```text
encoder busy
  -> frame 101 arrives
  -> frame 102 replaces 101
  -> encoder becomes ready
  -> encode frame 102
```

Queuing every frame produces correct answers about stale images.

## 16. Security and privacy boundary

### What remains local

- Original image bytes.
- Image metadata such as EXIF, unless separately submitted.
- Browser preprocessing intermediates.
- Downloaded encoder weights.

### What leaves the device

- The text question.
- Model/version metadata.
- `[264,768]` visual features, roughly 405,504 raw tensor bytes for the current fixed shape.
- Authentication and request metadata.

### Accurate claim

> Raw images are processed by a vision encoder on your device. Visual features and your question are sent to Overshoot so the cloud language model can answer. Visual features can still contain sensitive information about the image.

### Inaccurate claims

- "Your data never leaves your device."
- "Embeddings are anonymous."
- "The server cannot infer image content."
- "Images cannot be reconstructed."

The server must treat features as sensitive data:

- No tensor logging by default.
- Short retention.
- Access control and encrypted internal transport.
- No feature values in traces, metrics labels, or crash reports.
- Explicit leakage evaluation against the shipped representation.

## 17. Observability

Measure each stage independently:

| Stage | Measurements |
|---|---|
| Capability | browser, OS, adapter class, feature/limit result |
| Download | exact bytes, throughput, retries, cache hit/miss |
| Verification | digest time and failure reason |
| Session | initialization time and memory |
| Preprocessing | decode, resize, normalization, patch construction |
| Encoder | cold and warm p50/p90/p95/p99 |
| Serialization | conversion time and payload bytes |
| Network | upload, API, gateway, H200 TTFT |
| Generation | TTFT, completion time, output tokens |
| Recovery | device loss, cache eviction, cancellation, stale frame count |

Telemetry should not include pixels, visual features, prompts, or model outputs by default.

## 18. Current evidence and unknowns

| Item | Status |
|---|---|
| Native M4 encoder to H200 split | Qualified; about 319-352 ms p50 encoder latency |
| Binary BF16 protocol and gateway | Implemented and tested |
| Server-side projector and vLLM path | Implemented and tested |
| Fixed-shape tower ONNX export | Measured; 308,190,557 bytes |
| ONNX Runtime CPU execution | Passed as graph/operator check |
| Browser preprocessing parity | Not implemented or qualified |
| Browser WebGPU execution | Not measured |
| Browser peak memory | Not measured |
| INT4 artifact | Not produced |
| INT4 download size | Unknown |
| INT4 composed VLM quality | Unknown |
| Browser/H200 final-answer parity | Not qualified |
| Device-loss/cache/update recovery | Not implemented |

The neural split is proven by the native client. The browser product is not proven until the unknown rows are measured on the exact artifact and target devices.

## 19. Implementation sequence

1. Implement exact browser preprocessing and golden tests.
2. Create a dedicated vision Worker.
3. Load the existing FP16 ONNX graph with WebGPU on a supported Chromium build.
4. Record operator assignment, memory, cold latency, and warm latency.
5. Serialize browser output into the binary feature protocol.
6. Route the binary request through the public authenticated API to the private H200 gateway.
7. Run feature, logit, answer, and ChartQA comparisons.
8. Produce the INT4 candidate and count converted bytes.
9. Repeat browser and composed quality qualification for INT4.
10. Define and generate the manifest only from a qualified artifact.
11. Implement CDN delivery, cache, atomic updates, and deletion.
12. Add permissions, disclosures, route indicators, and fail-closed behavior.
13. Qualify the supported browser/device matrix and publish its boundaries.

## 20. Reading material

1. [ONNX Concepts](https://onnx.ai/onnx/intro/concepts.html)
2. [ONNX External Data](https://github.com/onnx/onnx/blob/main/docs/ExternalData.md)
3. [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)
4. [ORT WebGPU Execution Provider](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)
5. [ORT Web Large Models](https://onnxruntime.ai/docs/tutorials/web/large-models.html)
6. [Using Web Workers](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers)
7. [Transferable Objects](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Transferable_objects)
8. [WebGPU API](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)
9. [ImageBitmap](https://developer.mozilla.org/en-US/docs/Web/API/ImageBitmap)
10. [OffscreenCanvas](https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas)
11. [Browser Storage Quotas and Eviction](https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria)
12. [Browser WebGPU export feasibility](./BROWSER_WEBGPU_POC.md)
13. [Full research landscape](./LOCAL_VISION_INFERENCE_LANDSCAPE.md)
