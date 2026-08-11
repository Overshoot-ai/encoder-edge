# Packaging Plan: Local-First Vision Encoding for Cross-Device Gemma

Status: decision document
Date: 2026-08-07
Supersedes/extends: `docs/STEP_1_RESEARCH.md`, `docs/STEP_2_POC.md`, `docs/BROWSER_WEBGPU_POC.md`

## Decision

**Ship server-first, then earn the local tier with a bounded quantization experiment, then add native as a retention tier — in that order.**

1. **Server-first web app** (now): full pipeline on the H200, image upload, streaming answers. Works in every browser, zero install, and becomes the permanent fallback tier.
2. **Browser-local tier** (bounded experiment, then build): quantize the already-exported ONNX vision tower to weight-only INT4 (`MatMulNBits`), cutting the download from 338.7 MB to **~58–98 MB**, and run it with ONNX Runtime Web on WebGPU inside a Worker — gated by strict quality and performance pass/fail criteria before any user sees it.
3. **Native macOS helper** (when repeat users exist): reuses the already-qualified 320 ms MLX/BF16 path verbatim; 2–4 engineer-weeks, $99/yr, with a known security checklist and one Chrome Local-Network-Access permission prompt.

This ordering copies the pattern every successful local-AI product converged on (Superwhisper, Granola, Draw Things, Chrome's own hybrid guidance): **zero-install value first, local processing as the upgrade, native as the power tier.** No successful 2025–26 product ships "download hundreds of MB in a browser tab" as the first-run experience, and install funnels lose 40–95% of users before they see value.

---

## 1. Current state (measured, in-repo)

| Fact | Value | Source |
|---|---|---|
| Production split | Browser/native client runs vision tower; H200 runs RMSNorm + projector + LLM | `optimized_v2/mlx_client.py`, `optimized/optimized_vllm/model.py` |
| Wire object | Pre-projector `[264,768]` BF16 ≈ 405,504 tensor bytes at 480p (70% smaller than post-projector) | `optimized_v2/protocol.py` |
| Native encoder latency (M4) | p50 319–352 ms, qualified | `benchmark-results/projector-shift/REPORT.md` |
| Source weights | BF16 `vision.safetensors`, 338.7 MB (tower + projector) | `artifacts/gemma-4-e4b/client/` |
| Browser export | Full 16-layer pre-projector tower → fixed-shape FP16 ONNX, opset 18, 308,190,557 bytes, `[1,2376,768]→[1,264,768]`, ONNX checker + ONNX Runtime CPU pass | `optimized_v2/onnx_gemma4_e4b_vision.py`, `docs/BROWSER_WEBGPU_POC.md` |
| FP16 numerical drift (single image) | ONNX Runtime FP16 vs fixed FP32: relative L2 6.63e-4, cosine 0.9999996 — all finite | same |
| Quality-gate history | FP16 tower previously failed the production quality gate despite high cosine similarity; Q4/INT4 Core ML failed (22–36% relative L2); token-reduction schemes failed (ToMe 73%→64%) | `benchmark-results/` |

The open question is not "can the tower export" (proven) but **"can a ≤100 MB browser artifact pass decision-level quality gates and run fast enough on WebGPU."**

## 2. What the research established

### 2.1 Runtime: WebGPU + ONNX Runtime Web is the only serious browser path

- ORT Web's WebGPU EP supports every op in our graph (145 MatMul, 16 Softmax, RMSNorm via decomposition or `SimplifiedLayerNormalization`, GELU-tanh, Conv), has shipped FP16 since ORT 1.17 (Chrome 121+/Edge 122+), and supports `MatMulNBits` blockwise INT4 — the same op that powers Microsoft's production Phi-3-mini-onnx-web models. ([ORT Web support matrix](https://onnxruntime.ai/docs/get-started/with-javascript/web.html), [WebGPU op list](https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md))
- **Browser support is Chrome/Edge-only for ORT Web's WebGPU EP** — even though Safari 26 and Firefox 141+ ship the WebGPU *API*. Do not infer runtime support from API support. Safari/Firefox get the server path.
- `shader-f16` is a required device feature (~92.5% support overall, ~67.9% on Linux; disabled on Vulkan+NVIDIA). Runtime detection is mandatory. ([web3dsurvey](https://web3dsurvey.com/webgpu/features/shader-f16))
- **WebNN is not viable in 2026**: origin trial slipped four times (now Chrome 156–160, ~Dec 2026+), zero Safari signal, Microsoft says not for production. Our ONNX graph is WebNN-compatible, so adoption later is an execution-provider switch, not a rewrite. Revisit when the OT actually ships. ([W3C CR Jan 2026](https://www.w3.org/TR/2026/CR-webnn-20260122/), [OT thread](https://groups.google.com/a/chromium.org/g/blink-dev/c/5CWKSChYo98))
- Performance expectations: WebLLM demonstrates ~80% of native for browser WebGPU; LlamaWeb (16 devices, 8 vendors) shows large cross-device variance and browser memory ceilings. Expect the browser encoder to be **~0.5–1.5 s**, not the native 320 ms. ([arXiv:2412.15803](https://arxiv.org/abs/2412.15803), [arXiv:2605.20706](https://arxiv.org/html/2605.20706))
- Chrome's built-in Gemini Nano (GA since Chrome 148) and Apple Foundation Models **cannot run our pipeline** (no model choice, text-only output / native-only API). Ignore as substitutes.

### 2.2 Size: the download blocker shrinks 4–5.8×

Measured on our actual artifact (all round-trips SHA-256 verified):

| Option | Download size | Quality risk | Notes |
|---|---|---|---|
| BF16 raw (today) | 338.7 MB | — | Not shippable as a web download |
| Lossless brotli `Content-Encoding` | ~196 MB | zero | Free via CDN config; no runtime-memory savings |
| INT8 weight-only (DQ→FP16 matmul) | ~143–172 MB | ~zero | INT8 SigLIP-class encoders ≈ ±0.3% MME across 4 VLMs ([arXiv:2607.08029](https://www.alphaxiv.org/abs/2607.08029)) |
| **INT4 `MatMulNBits` block 32 + zstd/br transport** | **~58–66 MB compressed, ~85–98 MB raw** | low–moderate | WebGPU-supported since ORT 1.17; production-proven by Phi-3-mini-web; compute stays FP16 |

Download times for the INT4 artifact (~58 MB): **~5 s at 100 Mbps, ~19 s at 25 Mbps, ~46 s at 10 Mbps** — tolerable with real progress UI, versus 1.7–4.5 minutes for the FP16 artifact. Encoder swap (SigLIP2-B/16 + INT4 → ~45 MB) exists but requires projector retraining (LLaVA-recipe scale: 595K pairs, <4 h on 8×A100) and is only worth it if <80 MB becomes a hard requirement.

Token pruning (LAST, [arXiv:2607.27952](https://www.alphaxiv.org/abs/2607.27952)) shrinks the **wire payload**, not the model download; our 406 KB payload is already small. Park it.

### 2.3 Quality gates: mean cosine similarity is disqualified

The literature explains our earlier FP16 failure precisely: cosine ≈ 1 − O(ε²) cannot detect the drift that matters; ViTs carry global information in ~2.4% high-norm outlier tokens; correctness agreement drops while similarity metrics stay green. ([arXiv:2607.08734](https://arxiv.org/abs/2607.08734), [arXiv:2309.16588](https://arxiv.org/abs/2309.16588), [arXiv:2212.08254](https://arxiv.org/abs/2212.08254)) All future candidates must pass:

- **Feature level (diagnostic):** per-token cosine *percentiles* (min, p0.1, p1), KL/KS divergence of feature histograms, token-norm KS test, per-channel error profile, inf/NaN counting — against the native BF16 pipeline on a representative image set.
- **Decision level (the gate):** correctness agreement vs. the native pipeline, next-token logit KL, and the existing ChartQA harness on the composed browser+H200 system. (Harnesses already in `optimized_v2/`: `quality_benchmark.py`, `mlx_token_budget_chartqa.py`, `overshoot_eval.py`.)
- **Wire level:** emit the ONNX output as **f32 and round f32→BF16 once in the browser** (BF16 = upper 16 bits of FP32, round-to-nearest-even). Same 405 KB payload, one rounding stage removed versus FP16-wire→BF16.

### 2.4 INT4 recipe (implementation-ready)

Tool: `onnxruntime.quantization` `MatMulNBitsQuantizer` (first-party; renamed from `matmul_4bits_quantizer` in 2025; Olive adds nothing since our graph already exists).

```python
from onnxruntime.quantization import matmul_nbits_quantizer as mmq

quant = mmq.MatMulNBitsQuantizer(
    model="gemma4-e4b-vision-fp16.onnx",
    algo_config=mmq.DefaultWeightOnlyQuantConfig(   # RTN, no calibration data
        block_size=32,        # sweet spot: Intel/Metal fast-path kernels; transformers.js default
        is_symmetric=True,    # INT4, no zero-point tensor; try False if fidelity misses
    ),
    nodes_to_exclude=[],      # add block-0 / final projection only if validation fails
)
quant.process()
quant.model.save_model_to_file("gemma4-e4b-vision-int4.onnx", True)  # external data
```

Gotchas (all evidenced):

- **Coverage check is mandatory.** The quantizer only converts `MatMul` nodes with constant B inputs; `Gemm` and fused `com.microsoft.Attention` weights are invisible to it. The published Gemma-3-4B ONNX q4 artifact is the documented casualty (738 MB vs 840 MB FP16 — attention weights silently left unquantized). Count `MatMulNBits` nodes after quantization.
- **Do not use HQQ** for the web target: ORT's HQQ writes unpacked FP16 zero-points, which the ORT-Web WebGPU kernel explicitly rejects.
- **Do not set `accuracy_level`**: the browser kernel ignores it and it changes numerics on other EPs.
- **`actorder=False`**: reordered weights need support that is doubtful on ORT-Web.
- Keep the patch-embed Conv in FP16 (it isn't a MatMul anyway; RTN-on-conv at 4-bit is fragile and patch-embed is the #2 most fragile component).
- The browser `MatMulNBits` kernel accumulates in FP16; validate against an FP16 (not FP32) reference and include all-black/overexposed edge inputs.
- If RTN misses: asymmetric first, then GPTQ with **128–512 real preprocessed domain images** (never random noise — off-manifold Hessians). Fragile-component ladder if still failing: exclude block 0 (+~10 MB) or set it to 8-bit via per-node config.
- Serve the final artifact behind brotli/gzip `Content-Encoding` or ship a tiny WASM zstd decoder; INT4 payloads compress a further ~31% (measured).

### 2.5 Native path: fully priced, no surprises

- **Cost:** 2–4 engineer-weeks one-time, ~0.5–2 days/month ongoing, $99/yr. Notarization is minutes, fully scriptable, no human review. macOS 15+ removed the Control-click bypass, so signing+notarizing is effectively mandatory (Homebrew casks require it from Sept 1, 2026).
- **Distribution convention:** every successful local-AI app (Ollama 180 MB, Jan 103 MB, LM Studio 544 MB, Draw Things 279 MB) ships the engine and downloads weights on first run into a user-owned directory (`~/.ollama/models`, `~/Library/Application Support/…`). At our size, **bundling the weights in the dmg is also viable** (~350 MB installer, Draw Things scale) and removes the whole first-download failure class.
- **New friction as of Chrome 142 (Oct 2025):** web→localhost requests are gated by a one-time **Local Network Access permission prompt** per browser profile (extended to WebSockets in Chrome 147). The native bridge no longer avoids prompts versus browser-only — it trades them for an install funnel.
- **Security checklist is non-negotiable** (Zoom CVE-2019-13450, Trend Micro CVE-2016-3987, Ollama CVE-2024-37032, Docker CVE-2025-9074 are all the same bug): bind `127.0.0.1` only; per-install auth token stored `0600`, paired via URL-scheme callback; CORS origin allowlist; `Host` header validation (kills DNS rebinding); custom header on state-changing endpoints (forces preflight, kills CSRF); no GET side effects; uninstall must remove the server.
- Auto-update: Sparkle (EdDSA-signed appcast, delta updates) or a signed version-check + self-replace; model weights update on a separate, content-addressed cadence.

### 2.6 Privacy: engineer the claim before marketing it

The split keeps raw pixels on-device, but the transmitted features are **not** opaque:

- **CapRecover** (ACM MM'25, [arXiv:2507.22828](https://www.alphaxiv.org/abs/2507.22828)) attacks exactly our topology and recovers image labels (up to 92.71% top-1) and fluent captions (ROUGE-L ≈ 0.52) from encoder outputs *without reconstructing pixels*.
- The mitigating evidence ([arXiv:2606.14783](https://www.alphaxiv.org/abs/2606.14783)): for encoder-based VLMs, exact-string recovery from features was 0/48, and the governing variable is **spatial sampling density** — our 3×3 spatial pooling (2376→264 tokens) coarsens the grid in the right direction. Token-level noise/quantization do not help.
- Wire integrity matters as much as confidentiality: tampering with 10% of vision tokens collapsed VLM accuracy in VTM-Attack ([arXiv:2607.02819](https://www.alphaxiv.org/abs/2607.02819)). TLS + request authentication are mandatory.

**Defensible wording** (see §5): "Raw images are processed entirely on your device and are never transmitted. To answer, your device sends a compact numerical representation (visual features) of the image to our servers." Never claim "data never leaves your device," "features can't be reconstructed," or "private by design" as a bare slogan. FTC Operation AI Comply ("no AI exemption"), the Zoom ToS backlash, and the Friend/CNIL scrutiny are the standing warnings.

**Trust levers, ranked by community evidence:** open/source-available client; published data-flow table (Screenpipe); network-inspection guide ("verify it" beats "trust us"); fail-closed behavior; user-exportable request log (Apple Intelligence Report pattern); BYOK/custom endpoint; and the biggest one — **NVIDIA confidential-computing mode on the H200 with client-verified remote attestation**, which converts "trust our policy" into "verify the hardware" (Apple PCC and Screenpipe/Tinfoil have pre-educated the market on this vocabulary).

### 2.7 Browser storage: sufficient, with one sharp edge

- Cache API/OPFS/IndexedDB quotas are generous on desktop (Chrome ~60% of disk per origin; Safari ~60%; OPFS mature everywhere, no prompts). 58–98 MB is trivially within budget.
- **Eviction is the real constraint and is unchanged in 2026:** best-effort storage is LRU-evicted under disk pressure (origin-wide, atomic), and **Safari deletes script-written storage after 7 days without user interaction — installed PWAs ("Add to Dock") are exempt.** Treat cache as opportunistic; handle miss→re-download silently; nudge Safari users to install; call `navigator.storage.persist()` (Chrome auto-grants by engagement; Firefox prompts).
- Per-origin partitioning means every site re-downloads its own copy; no cross-site dedup exists. This is a structural cost of browser-only distribution.

---

## 3. The plan

### Phase 0 — Server-first product foundation (now; ~1–2 weeks)

The fallback tier we need forever; also the value-preview funnel.

- Productize the existing gateway path: HTTPS termination, auth, rate limits, request-size/shape/dtype validation (exists), idempotency IDs, `AbortController` cancellation, streaming `fetch`/SSE responses.
- UX: image upload + streaming answer; labeled mode indicator ("Cloud"); first-run consent naming exactly what leaves the device (the image, in this mode).
- Publish the **data-flow page** on day one: what is sent, where, retention, training posture — one unambiguous sentence a journalist can quote, plus the verification guide.
- Exit criteria: production deployment serving real users with p50/p90/p99 latency and cost instrumentation.

### Phase 1 — INT4 browser spike (days; bounded pass/fail)

The cheapest step that can kill or unlock the entire browser-local tier.

1. Quantize the exported FP16 ONNX (`optimized_v2/onnx_gemma4_e4b_vision.py` output) per §2.4: RTN, block_size 32, symmetric; verify MatMul→MatMulNBits coverage; measure artifact size (target ≤100 MB raw, ≤70 MB over the wire).
2. **Feature gates:** percentile cosine / KL-DS / token-norm KS vs native BF16 on a representative image set; include black/overexposed edge inputs.
3. **Decision gates:** correctness agreement + next-token logit KL + ChartQA on the composed browser-features → H200 pipeline.
4. **Performance probe:** minimal Chromium page, ORT Web in a DedicatedWorker, `executionProviders: ['webgpu']` only (no silent WASM fallback), `shader-f16` check, `enableGraphCapture`, warm/cold latency, memory behavior, `device.lost` recovery.

**Pass criteria:** ≥99% correctness agreement vs native pipeline (no ChartQA regression beyond noise), artifact ≤100 MB raw, warm encode <1 s on the base M4, single WebGPU partition (no WASM-fallback nodes mid-graph).
**If it fails:** retry once with asymmetric/GPTQ + block-0 exclusion; then INT8 (~143 MB, evidence-backed ~zero risk); if that fails, browser-local is dead and native becomes the only local tier. **Timebox: do not iterate more than twice.**

### Phase 2 — Hybrid web product (~1–2 weeks after Phase 1 passes)

- Sharded model delivery: 8–32 MB shards behind a versioned `manifest.json` (`model_id`, content hash, per-shard sha256, `required_features: ["shader-f16"]`, min runtime version); immutable CDN URLs; parallel resumable downloads; verify-then-commit per shard; Cache API (or OPFS) storage, IndexedDB for manifest metadata.
- Capability routing with a cached, measured qualification probe (browser version + OS + GPU adapter + model revision), not just `navigator.gpu` existence: **Auto** (local only after probe passes) / **Local preferred** (explicit consent before any cloud fallback) / **Cloud**.
- Dedicated Web Worker inference; transferable ArrayBuffers; f32 output → single-rounding RNE BF16 wire encoding; existing binary protocol unchanged (`dtype` metadata addition only if needed); streamed answers to the UI.
- Consent UX: per-request mode indicator, first-run wording per §5, fail-closed local-only mode, exportable request log.
- Safari/Firefox: server mode (no ORT Web WebGPU); Safari durability via PWA-install nudge.
- Exit criteria: local route live for Chrome/Edge on Apple Silicon with quality parity, sub-second warm encode, and eviction/device-loss recovery tested.

### Phase 3 — Native macOS helper (trigger: repeat-user evidence; ~2–4 weeks)

- Signed + notarized pkg/dmg (Developer ID, hardened runtime, `notarytool` in CI); weights bundled (~350 MB installer) or first-run download into `~/Library/Application Support/<app>/models` (content-addressed, sha256, resumable).
- Reuses the qualified MLX/BF16 encoder unchanged; localhost HTTP/SSE bridge with the §2.5 security checklist; onboarding copy planned around the Chrome LNA prompt; Sparkle or signed self-update.
- Exit criteria: notarized install → first answer in <5 min on a clean Mac; bridge passes the localhost security checklist review.

### Phase 4 — Optional leverage (any time after Phase 0)

- **H200 confidential computing + client-verified attestation** (the strongest trust lever; converts policy claims into hardware-verified ones).
- **Feature-leakage evaluation**: run CapRecover-style attacks against our transmitted features and publish the results — first-mover credibility no competitor currently offers.
- **Wire-level token pruning** (LAST-style, query-agnostic) if bandwidth or H200 prefill becomes a bottleneck; does not affect the download.
- **Encoder swap** (SigLIP2-B/16 + projector retrain → ~45 MB) only if sub-80 MB becomes a hard requirement.
- **WebNN re-evaluation** when the Chrome/Edge origin trial actually ships (currently 156–160).

---

## 4. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| INT4 tower fails decision-level quality gates | medium | kills browser-local tier | INT8 fallback (evidence-backed ~zero risk); native tier remains |
| Browser encode >1 s warm on M4 | medium | local tier feels slow vs native 320 ms | `enableGraphCapture`, Metal SubGroupMatrix kernels landing in ORT; gate at <1 s; position local as privacy tier, not speed tier |
| Safari/Firefox never get ORT Web WebGPU | high (near-term) | local tier is Chrome/Edge-only | server fallback is the product foundation (Phase 0) |
| Safari 7-day eviction wipes the model | high for Safari users | re-download annoyance | opportunistic cache + PWA nudge; Safari is server-mode anyway |
| FP16 accumulation in browser `MatMulNBits` kernel | low–medium | subtle feature drift | FP16-reference validation; edge-input tests; decision-level gates |
| Quantizer under-coverage (Gemm/fused attention) | low | artifact barely shrinks | node-count coverage check before benchmarking |
| Feature-inversion press/research moment | medium | privacy claim credibility | preemptive security page + own leakage evaluation + precise wording (§5); consider H200 attestation |
| Chrome LNA prompt confuses users (native tier) | medium | onboarding drop-off | one-time, per-profile; plan copy; Safari/Firefox unaffected today |
| Localhost bridge CVE-class mistake (native tier) | low if checklist followed | critical | §2.5 checklist + security review before launch |
| Download bounce on slow connections (browser tier) | medium | first-visit abandonment | server-first first impression with consent; real progress UI; 58 MB target |

## 5. Positioning and consent language

**Approved core claim:**

> Raw images are processed entirely on your device and are never transmitted. To answer, your device sends a compact numerical representation (visual features — about 400 KB, not an image) to our servers, where the language model runs. We never receive your photos. Verify it: [network-inspection guide].

**Do:** name the boundary precisely; publish the data-flow table; state what the server *can* see and its retention/training posture in one quotable sentence; label the cloud path as cloud (superwhisper pattern); provide a per-request mode indicator and an exportable request log; fail closed; keep telemetry off by default or loudly disclosed.

**Don't:** "your data never leaves your device" (features are data); "private by design" as a slogan; "images can't be reconstructed from what we send" (CapRecover recovers semantics without pixels); "anonymous features" (GDPR treats derived features as plausibly personal data — plan EU posture deliberately); silent default changes to routing or retention (Zoom/Google lesson); absolute safety framing.

## 6. Open questions

1. Exact warm/cold WebGPU latency distribution for the INT4 tower on base M4 vs M-series Pro/Max (Phase 1 probe).
2. Whether INT8 (143 MB) converts more net users than INT4 (58 MB) if INT4 shows any quality regression — an A/B worth doing only after Phase 1.
3. H200 NVIDIA CC mode + attestation: compatibility with our vLLM serving stack and its performance overhead.
4. EU posture for feature transmission (GDPR personal-data treatment) — needs a deliberate decision before marketing there.
5. Whether bundling vs first-run download wins for the native helper (bundle is simpler at our size; revisit if the model grows).

## 7. Sources (primary)

Runtime: [ORT Web support matrix](https://onnxruntime.ai/docs/get-started/with-javascript/web.html) · [WebGPU op list](https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md) · [ORT 1.17 release](https://github.com/microsoft/onnxruntime/releases/tag/v1.17.0) · [WebNN W3C CR](https://www.w3.org/TR/2026/CR-webnn-20260122/) · [WebNN OT thread](https://groups.google.com/a/chromium.org/g/blink-dev/c/5CWKSChYo98) · [shader-f16 support](https://web3dsurvey.com/webgpu/features/shader-f16) · [WebLLM](https://arxiv.org/abs/2412.15803) · [LlamaWeb](https://arxiv.org/html/2605.20706)

Quantization: [ORT MatMulNBits quantizer](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/matmul_nbits_quantizer.py) · [MatMulNBits WebGPU kernel](https://github.com/microsoft/onnxruntime/blob/main/js/web/lib/wasm/jsep/webgpu/ops/matmulnbits.ts) · [transformers.js quantize script](https://github.com/huggingface/transformers.js/blob/main/scripts/quantize.py) · [Gemma-3 ONNX artifacts](https://huggingface.co/onnx-community/gemma-3-4b-it-ONNX) · [sVLM component study](https://www.alphaxiv.org/abs/2607.08029) · [VLM quantization best practices](https://www.alphaxiv.org/abs/2601.15287) · [MixFrag](https://www.alphaxiv.org/abs/2607.28589) · [APHQ-ViT](https://arxiv.org/abs/2504.02508) · [RepQ-ViT](https://arxiv.org/abs/2212.08254) · [ZipNN](https://arxiv.org/abs/2411.05239)

Split architectures: [LAST](https://www.alphaxiv.org/abs/2607.27952) · [Distributed VLMs](https://www.ee.columbia.edu/~jghaderi/perconai25.pdf) · [edgeVLM](https://www.alphaxiv.org/abs/2508.12638) · [SplitCompute](https://github.com/tanmaysachan/splitcompute)

Privacy/security: [CapRecover](https://www.alphaxiv.org/abs/2507.22828) · [Vision encoder as privacy boundary](https://www.alphaxiv.org/abs/2606.14783) · [VTM-Attack](https://www.alphaxiv.org/abs/2607.02819) · [Image prompt reconstruction](https://www.alphaxiv.org/abs/2606.18710) · [FTC Operation AI Comply](https://www.ftc.gov/news-events/news/press-releases/2024/09/ftc-announces-crackdown-deceptive-ai-claims-schemes) · [Screenpipe architecture](https://screenpipe.com/security/architecture) · [Apple PCC](https://security.apple.com/blog/private-cloud-compute/)

Native/macOS: [Apple notarization](https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution) · [Chrome Local Network Access](https://developer.chrome.com/blog/local-network-access) · [Sparkle](https://sparkle-project.org/documentation/) · [Ollama FAQ](https://docs.ollama.com/faq) · [Homebrew Gatekeeper policy](https://github.com/Homebrew/brew/issues/20755) · [Zoom CVE-2019-13450](https://nvd.nist.gov/vuln/detail/cve-2019-13450) · [Ollama CVE-2024-37032](https://www.wiz.io/blog/probllama-ollama-vulnerability-cve-2024-37032) · [0.0.0.0 Day](https://www.oligo.security/blog/0-0-0-0-day-exploiting-localhost-apis-from-the-browser)

Storage/product patterns: [MDN storage quotas](https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria) · [WebKit storage policy](https://webkit.org/blog/14403/updates-to-storage-policy/) · [Chrome built-in AI model management](https://developer.chrome.com/docs/ai/understand-built-in-model-management) · [Granola security](https://www.granola.ai/security) · [superwhisper](https://superwhisper.com) · [Draw Things engineering](https://engineering.drawthings.ai/)
