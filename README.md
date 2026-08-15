# Edge Encoder

## CLI

```bash
git clone https://github.com/Overshoot-ai/encoder-edge.git
cd encoder-edge
uv sync
```

Run an encoder:

```bash
uv run edge-encoder encode \
  --model google/gemma-4-E4B-it \
  --input docs/architecture.png \
  --output features.bin
```

This writes `features.bin` and `features.bin.json`. For gated Hugging Face
models, set `HF_TOKEN` first.

## How to Run E2E

Chat with a projector-aware decoder:

```bash
uv run edge-encoder chat \
  --server https://gateway.example.com \
  --image docs/architecture.png \
  --prompt "What does this diagram show?"
```

Set `EDGE_ENCODER_API_KEY` when the gateway requires authentication.

Pipeline independent image requests from JSONL:

```jsonl
{"id":"one","image":"images/one.jpg","prompt":"What is shown?"}
{"id":"two","image":"images/two.jpg","prompt":"Read the chart title."}
```

```bash
uv run edge-encoder batch \
  --server https://gateway.example.com \
  --input requests.jsonl \
  --output responses.jsonl \
  --max-in-flight 4
```

Run the gateway next to a projector-aware vLLM server:

```bash
EDGE_ENCODER_GATEWAY_API_KEY=change-me \
uv run edge-encoder gateway \
  --upstream http://127.0.0.1:8001 \
  --host 0.0.0.0 \
  --port 8002
```

The vLLM server must serve the matching `gemma-4-e4b-optimized` artifact with
multimodal embeddings enabled. Put HTTPS in front of the gateway when exposing
it publicly. Set `EDGE_ENCODER_SERVER_REVISION` when deploying a differently
versioned compatible server artifact.

Compare the automatic MLX optimization with PyTorch/MPS:

```bash
uv run edge-encoder benchmark \
  --model google/gemma-4-E4B-it \
  --input docs/architecture.png \
  --output benchmark.json
```

## WebGPU Viewer

```bash
git clone https://github.com/Overshoot-ai/encoder-edge.git
cd encoder-edge/browser_webgpu
npm install
npm run viewer
```

Then open [http://localhost:3000](http://localhost:3000) in Chrome.

On first launch, `npm run viewer` automatically:

- Downloads the pinned 308 MB encoder from the GitHub release
- Verifies its byte size and SHA-256
- Caches it locally
- Starts the website

Subsequent launches reuse the verified cached model.
