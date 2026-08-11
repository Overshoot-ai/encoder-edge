# Step 2 Proof of Concept: Gemma 4 12B Unified

Verified: 2026-07-29

## Goal

Provide one user-facing operation while ensuring the server does not receive image pixels or contain Gemma's image embedding weights:

```text
image + question -> client image embedder -> visual tensor + prompt metadata -> server language model -> answer
```

## Environment

- NVIDIA H200 SXM with 143,771 MiB GPU memory
- 16 vCPUs and 200 GiB system memory
- Ubuntu 24.04 with CUDA 13
- PyTorch 2.11.0+cu130
- Transformers 5.12.1
- Model: `google/gemma-4-12B-it`

## Physical Artifact Split

The exporter loaded the original Hugging Face checkpoint once and produced:

| Artifact | Contents | Measured size |
|---|---|---:|
| Client | Processor, model configuration, and `Gemma4UnifiedVisionEmbedder` weights | 126 MB |
| Server | Gemma language model and LM head, with image/audio configurations set to null | 23 GB |

The complete client image module contains 49,922,304 parameters. This is larger than the approximately 35M paper-level estimate because the complete module includes the final 3,840-to-3,840 multimodal projection.

The server checkpoint index was inspected after export:

```text
server tensor count: 666
model.embed_vision.* tensors: 0
model.embed_audio.* tensors: 0
server vision_config: null
server audio_config: null
```

The server therefore does not merely skip the image path at runtime; its saved artifact physically excludes those weights.

## Equivalence Test

A synthetic 640x360 image containing a red square and a blue circle was processed through two paths:

```text
NORMAL
pixels -> complete Gemma model -> answer

SPLIT
pixels -> client-only image embedder -> visual tensor -> server-only language model -> answer
```

Measured result:

```text
visual tokens: 264
maximum final-prefill-logit difference: 0.00000000
generated token sequences equal: true
normal answer: The image shows a red square and a blue circle.
split answer: The image shows a red square and a blue circle.
```

This proves exact numerical equivalence for the tested image, prompt, software versions, BF16 execution, and deterministic decoding settings. It does not prove equivalence across every image, runtime, quantization level, or hardware platform.

## HTTP Client/Server Test

The server-only artifact was started with Python's standard HTTP server. A separate client process accepted the image and question, ran only the exported image module, serialized the output with Safetensors, and sent it over HTTP.

Measured result:

```text
answer: The image shows a red square and a blue circle.
client hostname: <client-hostname>
client device: mps
client image parameters: 49,922,304
server hostname: <server-hostname>
server device: cuda (NVIDIA H200)
server has image embedder: false
visual tokens: 264
HTTP request body: 2,034,680 bytes
```

The request contained:

- BF16 visual features
- Tokenized prompt IDs
- Attention mask
- Multimodal token-type IDs
- No image pixels

The 2.03 MB request is much larger than the brief's 50-70 KB compressed-image estimate. This first proof establishes architectural correctness and raw-image separation, not bandwidth savings.

## Physical Cross-Device Test

The equivalence and first HTTP tests above ran as separate processes on the H200. A second test established the intended physical topology:

```text
APPLE SILICON MAC                         H200 VM
sample.png                                server artifact only
  -> 126 MB client artifact                 -> receive Safetensors
  -> image embedder on MPS                  -> cast features to BF16
  -> SSH tunnel --------------------------> -> Gemma language model
                                             -> answer
```

The Mac had no server weights and the H200 server artifact had no image embedding weights. The result was:

```text
answer: The image shows a red square and a blue circle.
visual tokens: 264
HTTP request body: 2,034,680 bytes
```

MPS FP16 initially produced NaN or infinity values, and the server rejected that request before inference. FP32 produced finite features and the correct answer but doubled the payload to 4,062,200 bytes. MPS BF16 was then tested successfully: it preserved FP32-like exponent range while returning to the 2,034,680-byte payload. This is a concrete client-runtime finding and reinforces that numeric formats must be evaluated rather than assumed safe.

## Security Checks Implemented

- Safetensors rather than pickle serialization
- Request-size limit
- Exact tensor-name allowlist
- Shape and dtype validation
- Prompt and visual-token limits
- NaN and infinity rejection
- Image-placeholder and feature-count matching

This remains a research implementation. It does not authenticate users, provide TLS itself, enforce application-level signatures, run content-safety checks, or prevent semantic recovery from visual features.

## Lean Implementation Boundary

The runtime intentionally contains only three responsibilities:

| File | Required responsibility |
|---|---|
| `client.py` | Load the image module, process image/text, create visual features, and send one HTTP request |
| `server.py` | Load the language model, validate one tensor request, generate, and return JSON |
| `modeling.py` | Replace image placeholders with received features using Gemma's text embedding layer |

`export_split.py` is a one-time artifact builder. `verify.py` is an optional proof tool and is not part of serving. The implementation has no database, queue, cache, container framework, model-serving engine, orchestration layer, authentication subsystem, or custom binary protocol. HTTP uses Python's standard library. The only runtime packages beyond Python are the four libraries required to load the model, process an image, represent tensors, and serialize tensors: PyTorch, Transformers, Pillow, and Safetensors.

The split-serving source is 165 lines: 58 client lines, 75 server lines, and 32 shared Gemma insertion lines. The one-time exporter is 61 lines. The optional equivalence verifier is 124 lines. The runtime has one request route and no frontend, streaming, telemetry, health route, device configuration, visual-token controls, or tunable generation settings.

This is a bounded engineering claim, not a mathematical proof that no shorter program can exist. Removing the remaining validation or separating fewer responsibilities could reduce line count but would make the demonstration less safe or less readable without changing the architecture.

## VM Cleanup

Before testing, approximately 87 GiB of old Hugging Face model caches and vLLM compilation caches were removed. Existing datasets, repositories, and unrelated experiment outputs were preserved. After verification:

- The test server was initially stopped and GPU memory returned to 0 MiB.
- The redundant complete Gemma checkpoint cache was removed.
- The verified 126 MB client and 23 GB server artifacts were retained under the local `artifacts/gemma-4-12b` directory.

The server was later restarted from those retained artifacts to verify the reduced runtime.

## Next Step

Run the client artifact on candidate devices and measure:

- Model download and initialization time
- Peak CPU/GPU memory
- Image embedding latency
- Energy use
- Browser, Mac, and Jetson runtime compatibility

Keep this implementation as the minimal architectural baseline. Add visual-token reduction, INT8 feature quantization, streaming, or optimized H200 serving only in separately measured iterations.
