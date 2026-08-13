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
