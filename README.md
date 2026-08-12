# Run the WebGPU embedding viewer

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
