# Bit-Identical Binary Transport

This version keeps the existing MPS vision encoder and vLLM engine but removes base64 from the Mac-to-H200 link. It does not quantize or otherwise alter the BF16 visual tensor.

```text
Mac BF16 tensor
  -> raw tensor bytes over persistent HTTP
  -> H200 binary gateway
  -> identical BF16 tensor
  -> vLLM over H200 loopback
  -> SSE response to Mac
```

The gateway translates the binary request into vLLM's existing `image_embeds` Chat Completions format. Base64 still exists over H200 loopback, but it is no longer transferred across the Mac-to-H200 network.

## Prove Bit Identity

Run on either machine:

```bash
python -m optimized_v2.prove
```

The proof compares the raw BF16 bits before transport, after binary reconstruction, and after the gateway's vLLM translation. Expected output includes:

```json
{
  "raw_tensor_bytes": 2027520,
  "binary_request_bytes": 2027666,
  "base64_tensor_bytes": 2703360,
  "mac_to_gateway_bits_equal": true,
  "gateway_to_vllm_bits_equal": true,
  "proof": "PASS"
}
```

## Run

Start the existing vLLM server on the H200:

```bash
./optimized/serve.sh artifacts/gemma-4-12b-vllm
```

In a second H200 terminal, start the binary gateway:

```bash
python -m optimized_v2.gateway \
  --upstream http://127.0.0.1:8001 \
  --host 127.0.0.1 \
  --port 8002
```

On the Mac, tunnel the gateway port:

```bash
ssh -N \
  -L 8002:127.0.0.1:8002 \
  <user>@<h200-host>
```

Run the binary client on the Mac:

```bash
python -m optimized_v2.client \
  --artifact artifacts/gemma-4-12b/client \
  --server http://127.0.0.1:8002 \
  --image /path/to/image.png \
  --question "What is shown in this image?"
```

Alternatively, start the browser frontend:

```bash
python -m optimized_v2.frontend \
  --artifact artifacts/gemma-4-12b/client \
  --server http://127.0.0.1:8002 \
  --port 3002
```

Then open [http://127.0.0.1:3002](http://127.0.0.1:3002).

## Compare With Base64 Transport

With vLLM, the gateway, and both tunnels running:

```bash
python -m optimized_v2.benchmark \
  --artifact artifacts/gemma-4-12b/client \
  --image /path/to/image.png \
  --rounds 5
```

The benchmark warms both clients, alternates request order, verifies identical generated text, and reports median payload and latency metrics.

## Quality Benchmark

The quality benchmark encodes each image once, sends the same visual tensor to Transformers, base64 vLLM, and binary vLLM, and scores required concepts plus forbidden hallucinations:

```bash
python -m optimized_v2.quality_benchmark \
  --artifact artifacts/gemma-4-12b/client \
  --cases /path/to/quality-cases.json \
  --minimal-server http://127.0.0.1:8000 \
  --base64-server http://127.0.0.1:8001 \
  --binary-server http://127.0.0.1:8002
```

Case files are JSON arrays with this structure:

```json
[
  {
    "name": "red_car",
    "image": "/path/to/car.png",
    "question": "What is the main object and its color?",
    "required": [["car", "vehicle"], ["red"]],
    "forbidden": ["football"]
  }
]
```

Prefix caching must remain disabled for precomputed visual embeddings. During testing it caused cache-state-dependent answers and base64/binary disagreement despite bit-identical tensors. `optimized/serve.sh` disables it.

## Full H200 Versus Split

Reconstruct the stock full-model vLLM artifact from the existing split weights:

```bash
python optimized/prepare_full_vllm_artifact.py \
  --client artifacts/gemma-4-12b/client \
  --server artifacts/gemma-4-12b/server \
  --output artifacts/gemma-4-12b-vllm-full
```

[`deployment_benchmark.py`](deployment_benchmark.py) runs the same labeled cases in either mode:

```bash
python -m optimized_v2.deployment_benchmark full \
  --cases /path/to/cases.json \
  --server http://127.0.0.1:8001 \
  --model gemma-4-12b-full \
  --rounds 2 \
  --output /tmp/full.json

python -m optimized_v2.deployment_benchmark split \
  --cases /path/to/cases.json \
  --server http://127.0.0.1:8002 \
  --model gemma-4-12b-optimized \
  --artifact artifacts/gemma-4-12b/client \
  --rounds 2 \
  --output /tmp/split.json
```

The measured comparison is recorded in [`deployment_benchmark.txt`](deployment_benchmark.txt).

The Overshoot-style latency runs and 200-sample ChartQA comparison are recorded
in [`../benchmark-results/`](../benchmark-results/) and
[`chartqa_benchmark.txt`](chartqa_benchmark.txt). See
[`vision_numerics.md`](vision_numerics.md) for the measured reason that full
CUDA vision execution and split MPS vision execution are deterministic but not
bit-identical. The version-matched PyTorch 2.11 rerun is recorded in
[`pytorch_upgrade_benchmark.txt`](pytorch_upgrade_benchmark.txt).

## Experimental Browser Export

Export the fixed 480p E4B pre-projector tower to FP16 ONNX with:

```bash
python -m optimized_v2.onnx_gemma4_e4b_vision \
  --artifact artifacts/gemma-4-e4b/client \
  --image artifacts/gemma-4-12b/sample.png \
  --output /tmp/gemma4-e4b-web-fp16.onnx
```

Install the optional dependencies with `uv sync --extra web-export`. The
measured full graph is 308 MB and executes successfully in ONNX Runtime CPU
validation. This does not yet qualify WebGPU performance or FP16 end-to-end
quality. See
[`../docs/BROWSER_WEBGPU_POC.md`](../docs/BROWSER_WEBGPU_POC.md).
