# Browser WebGPU Export Feasibility

Verified: 2026-08-07

## Goal

Determine whether the Gemma 4 E4B vision tower can be represented as a
browser-oriented ONNX graph before building a JavaScript client.

The tested boundary is the current server-projector split:

```text
fixed preprocessed patches [1,2376,768] FP16
  -> 16-layer vision tower
  -> pooled pre-projector features [1,264,768] FP16
```

The server-side RMSNorm/projector is not included.

## Export Harness

[`optimized_v2/onnx_gemma4_e4b_vision.py`](../optimized_v2/onnx_gemma4_e4b_vision.py)
reuses the fixed-shape wrapper from the Core ML experiment. The wrapper
precomputes position embeddings and RoPE values for one rectangular patch grid
and replaces shape-dependent pooling with fixed reshapes and reductions.

Unlike the Core ML path, the ONNX export retains Hugging Face's original
FP32-accumulated RMSNorm. It does not use the Core ML-specific stabilized
RMSNorm substitution.

Install the optional export dependencies and run the export with:

```bash
uv sync --extra web-export

python -m optimized_v2.onnx_gemma4_e4b_vision \
  --artifact artifacts/gemma-4-e4b/client \
  --image artifacts/gemma-4-12b/sample.png \
  --output /tmp/gemma4-e4b-web-fp16.onnx
```

Generated model files belong in artifact storage or a CDN, not in Git.

## Full-Graph Result

| Property | Result |
|---|---:|
| ONNX opset | 18 |
| Layers | 16 |
| Input shape | `[1,2376,768]` |
| Output shape | `[1,264,768]` |
| Model size | 308,190,557 bytes |
| Export time | 41.54 s |
| ONNX checker time | 0.73 s |
| ONNX Runtime CPU load time | 1.11 s |
| ONNX Runtime CPU inference | 7.84 s |

CPU inference is a numerical and operator-coverage check. It is not a WebGPU
performance estimate.

The graph contains standard operators. Its main compute operators are 145
`MatMul` nodes, 16 `Softmax` nodes, and elementwise/reduction operations. The
complete graph has 4,651 nodes and 459 initializers.

## Numerical Result

The fixed FP32 wrapper reproduces the Hugging Face tower on the tested image:

| Comparison | Relative L2 | Cosine similarity | Maximum absolute error |
|---|---:|---:|---:|
| Hugging Face vs fixed FP32 | `7.09e-8` | `0.99999994` | `0.0078125` |
| Fixed FP32 vs PyTorch FP16 | `9.91e-4` | `0.99999928` | `59.7090` |
| Fixed FP32 vs ONNX Runtime FP16 | `6.63e-4` | `0.99999958` | `31.4844` |
| PyTorch FP16 vs ONNX Runtime FP16 | `9.11e-4` | `0.99999946` | `48.0` |

All tested outputs were finite. The maximum FP32 reference magnitude was
approximately 51,272, below FP16's finite limit for this image.

These measurements prove exportability for one image and one fixed shape. They
do not qualify FP16 model quality. ChartQA and mixed-image testing remain
required because FP16 tower execution previously failed the production quality
gate despite high feature cosine similarity.

## What Is Proven

- The complete pre-projector E4B vision tower exports to ONNX opset 18.
- ONNX Runtime can load and execute every exported operator.
- The resulting artifact is approximately 308 MB rather than an assumed 340 MB.
- Fixed-shape preprocessing removes dynamic position, mask, and pooling logic
  from the runtime graph.
- The graph produces the expected `[1,264,768]` feature boundary.

## What Is Not Proven

- ONNX Runtime Web can assign every important operator to its WebGPU provider.
- Chromium can allocate the model and intermediates under browser GPU limits.
- WebGPU latency is competitive with the 319-352 ms native MLX path.
- Safari, Firefox, Intel Macs, or mobile devices can run this graph.
- FP16 features retain acceptable end-to-end ChartQA quality.
- The existing BF16-only binary gateway can consume browser FP16 output.
- Browser preprocessing reproduces the Python processor byte-for-byte.

## Next Gate

The next experiment should be a Chromium-only local page using ONNX Runtime
Web in a dedicated Worker. It must record provider assignment, model load time,
peak GPU allocation behavior, warm and cold inference latency, device-loss
recovery, and the resulting feature tensor. Do not add a cloud request path
until local output passes the existing H200 quality gate.

If WebGPU falls back to WASM for material portions of the graph or exceeds a
one-second warm latency target on the M4, stop the browser-only path and retain
the signed native MLX helper as the performance route.
