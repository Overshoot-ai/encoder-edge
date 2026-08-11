# H200 Vision Projector Shift

## Configuration

- Client: Apple M4, MLX 0.32.0, mlx-vlm 0.6.9
- Server: NVIDIA H200, BF16 vLLM
- Baseline transport: post-projector `[N, 2560]` BF16
- Candidate transport: pooled pre-projector `[N, 768]` BF16
- Candidate server artifact: `gemma-4-e4b-vllm-projector`
- The server accepts both widths; width 2560 bypasses the projector.

## 480p TTFT

Five sequential measured requests followed one warmup request for each path.

| Metric | Width 2560 | Width 768 | Change |
|---|---:|---:|---:|
| Tensor bytes | 1,351,680 | 405,504 | -70.0% |
| Request bytes | 1,351,865 | 405,688 | -70.0% |
| Client encode mean | 659.7 ms | 648.1 ms | -1.8% |
| Client serialization mean | 1.84 ms | 0.82 ms | -55.5% |
| Gateway preparation mean | 8.78 ms | 2.89 ms | -67.1% |
| vLLM TTFT mean | 22.86 ms | 20.39 ms | -10.8% |
| Remote TTFT mean | 654.5 ms | 388.8 ms | -40.6% |
| Pipeline TTFT mean | 1,336.7 ms | 1,057.5 ms | -20.9% |

Artifacts:

- `../mlx-roofline/projector-shift-width2560-5r.json`
- `../mlx-roofline/projector-shift-width768-5r.json`

## ChartQA Check

The first 10 ChartQA test samples were run at a processor soft-token budget of
273. The parsed answer matched between paths for all 10 samples.

| Metric | Width 2560 | Width 768 |
|---|---:|---:|
| Exact match | 0.60 | 0.60 |
| Relaxed accuracy | 1.00 | 1.00 |
| Anywhere accuracy | 1.00 | 1.00 |
| Pipeline TTFT p50 | 1,320.3 ms | 916.2 ms |
| Pipeline TTFT p90 | 1,648.1 ms | 1,147.2 ms |
| Pipeline E2E p50 | 3,198.0 ms | 2,738.5 ms |

Full generated explanations are not byte-identical because the projector now
runs through CUDA BF16 kernels instead of MLX. The task answers and scores were
unchanged in this sample. These sequential timings remain sensitive to Mac and
network pressure; the payload reduction and quality result are the primary
qualification signals.

## B1 and B8 Concurrency

Each batch size had one warmup followed by three measured rounds. B8 performs
one eight-image MLX encoder call, then sends eight requests concurrently.

| Metric (p50 unless noted) | B1 | B8 |
|---|---:|---:|
| Batch wall time | 872.5 ms | 5,325.3 ms |
| Throughput | 1.146 images/s | 1.502 images/s |
| Batch encode time | 668.1 ms | 4,679.7 ms |
| Pipeline TTFT per request | 871.1 ms | 5,155.9 ms |
| Remote TTFT per request | 164.3 ms | 425.7 ms |
| Request bytes per image | 405,705 | 405,705 |

Compared with the earlier width-2560 run, projector-split B8 p50 throughput
increased from 1.294 to 1.502 images/s and p50 pipeline TTFT decreased from
6,085.4 to 5,155.9 ms. Those runs were not interleaved, so this comparison also
contains system and network variation.

Artifact: `e2e-split-b1-b8-3r.json`

## 4-bit Encoder Experiment

The original MLX checkpoint keeps the vision encoder in BF16 and quantizes only
the local projector. The experiment applied 4-bit affine, group-size-64,
weight-only quantization to every encoder Linear while retaining BF16
activations and transport.

| Metric (p50) | Local 4-bit projector | H200 BF16 projector |
|---|---:|---:|
| B1 throughput | 0.948 images/s | 0.892 images/s |
| B1 pipeline TTFT | 1,036.1 ms | 1,029.5 ms |
| B1 remote TTFT | 292.2 ms | 171.0 ms |
| B8 throughput | 1.207 images/s | 1.400 images/s |
| B8 pipeline TTFT | 6,323.9 ms | 5,698.4 ms |
| B8 remote TTFT | 849.9 ms | 475.6 ms |
| Request bytes per image | 1,351,882 | 405,705 |

Encoder parameter storage fell from 334,730,112 to 116,827,008 bytes, a 65.1%
reduction. A 10-round same-process interleaved encoder-only A/B found that Q4
regressed p50 from 626.5 to 659.5 ms at B1 (5.3%) and from 4,651.3 to 5,026.7
ms at B8 (8.1%). This removes network and H200 variation from the comparison.
The Q4 pre-projector output had 54.2% relative L2 error and only 0.191% of BF16
values were bit-equal to the BF16 encoder output.

A fresh end-to-end BF16 baseline in the same VM session was noisy: Q4 was 4.1%
slower in B1 pipeline TTFT but 3.8% faster in B8 pipeline TTFT because remote
time moved in the opposite direction. The interleaved encoder-only result is the
valid measure of quantization speed; the apparent B8 end-to-end gain is not an
encoder gain.

On the first 10 ChartQA samples, both Q4 encoder projector placements scored
0.60 exact match, 0.80 relaxed accuracy, and 0.90 anywhere accuracy. The BF16
encoder scored 0.60, 1.00, and 1.00 respectively. Encoder quantization lost two
previously correct cases.

With an identical captured `[1, 266, 768]` BF16 input, the local Q4 projector
and H200 BF16 projector outputs had 6,252 of 680,960 values bit-equal (0.918%).
The outputs had cosine similarity 0.98334, relative L2 error 0.18346, mean
absolute error 0.09139, and maximum absolute error 0.52539.

Artifacts:

- `e2e-q4-encoder-local-q4-projector-b1-b8-3r-warm.json`
- `e2e-q4-encoder-h200-bf16-projector-b1-b8-3r.json`
- `chartqa-q4-encoder-local-q4-projector/`
- `chartqa-q4-encoder-h200-bf16-projector/`
- `vision-encoder-bf16-vs-q4-interleaved-10r.json`
- `e2e-bf16-encoder-h200-bf16-projector-b1-b8-5r-current.json`

## Blockwise FP4 Follow-up

The Baseten NVFP4 method uses 16-element E2M1 blocks, FP8 two-level scales,
quantized activations, and native Blackwell FP4 tensor-core GEMMs. The earlier
affine experiment used 64-element integer groups, per-group biases, and BF16
activations, so it was not equivalent.

MLX 0.32 exposes the matching `nvfp4` representation, but activation-plus-weight
`QQMatmul` is unimplemented on Metal for this workload, including after
flattening Gemma's 3-D token tensors to 2-D. Weight-only NVFP4 is supported:

| Variant | Parameter reduction | B1 p50 change | B8 p50 change | Relative L2 error |
|---|---:|---:|---:|---:|
| All transformer linears, block 16 | 64.8% | 9.6% slower | 4.7% slower | 13.5% |
| MLP linears only, block 16 | 48.6% | 7.4% slower | 9.7% slower | 8.8% |

The M4 has no native FP4 matrix hardware. The H200 is Hopper and also lacks the
Blackwell NVFP4 tensor-core path used in the article. Consequently, the
available implementation provides storage compression but not a compute
speedup. The BF16 encoder remains the qualified path for this hardware.

Artifacts:

- `vision-encoder-bf16-vs-nvfp4-weight-interleaved-10r.json`
- `vision-encoder-bf16-vs-nvfp4-mlp-interleaved-10r.json`

## mlx-vlm Policy Audit

The upstream `mlx-vlm` converter does not apply its standard Q4 recipe to vision
encoders. Its base predicate calls `skip_multimodal_module`, which excludes
paths containing `vision_tower`, `vision_model`, connectors, projectors, and
audio towers. The downloaded Gemma checkpoint confirms this policy: text
linears and `embed_vision.embedding_projection` are affine Q4/group-64, while
vision attention and MLP linears are BF16.

The upstream mixed-bit recipes are language-model recipes. They preserve more
bits for sensitive `v_proj` and `down_proj` modules in early, late, and periodic
blocks. A vision-adapted `mixed_4_8` experiment, with the patch embedder retained
in BF16, produced:

| Parameter reduction | B1 p50 change | B8 p50 change | Relative L2 error |
|---:|---:|---:|---:|
| 61.3% | 6.2% slower | 10.1% slower | 11.0% |

The ViT PTQ literature explains why a generic recipe is insufficient. Effective
methods calibrate quantization intervals, preserve attention ranking, assign
mixed precision from layer sensitivity, and apply transformer-specific handling
for LayerNorm and softmax distributions. FQ-ViT's near-lossless result uses
8-bit weights/activations plus specialized Power-of-Two Factor and Log-Int-
Softmax quantizers, not naive weight-only W4.

Artifact: `vision-encoder-bf16-vs-mixed-4-8-interleaved-10r.json`

## Cider Integer-GEMM Follow-up

The Cider M5 INT8 guide was mapped against the M4 affine-Q4 experiment. Its
largest optimization, direct device-to-register reads, was already present in
the custom kernel. Small-M dispatch does not apply to the encoder's `M=264` and
`M=2112` projection shapes. M5 `int8 x int8 -> int32` TensorOps are unavailable
on the M4, and the existing kernel uses classic `simdgroup_matrix` FP32
accumulation after Q4 dequantization.

Three chip-agnostic adaptations were tested under the same interleaved harness:

- Cache each output fragment's four Q4 scale/bias pairs across its complete
  32- or 64-element quantization group.
- Add Cider's volatile compiler scheduling fence after each load/MMA step.
- Swizzle groups of four M tiles across adjacent N tiles.

All cached, fenced, and swizzled outputs were bit-identical to the original
custom kernel on all 12 B1/B8 projection and group-size cases. Paired per-round
medians showed explicit parameter caching improved 9 of 12 cases, usually by
about 4-10%, while the fence had no consistent M4 advantage and swizzling was
neutral or harmful for the largest B8/long-K cases. Even the best candidate
remained slower than MLX's native affine-Q4 implementation in every case, so
none qualifies for integration. Production remains BF16.

This run used the current project environment's MLX 0.31.1 on Apple GPU G16G;
the artifact records raw samples because host load produced large outliers.

Artifact: `../mlx-roofline/affine-q4-cider-agnostic-candidates.json`

## Core ML Tower-Only Follow-up

The initial Core ML graph included the H200-bound RMSNorm/projector. The saved
ML Program was truncated immediately before projector RMSNorm and re-exported
with a fixed `[1,264,768]` output, avoiding any need for the deleted source
checkpoint. A Core ML compute-plan audit reports all 2,490 executable tower
operations as Neural Engine preferred and supported; the remaining 2,171
entries are constants or have no independent device assignment. CPU fallback
therefore does not explain the latency.

| Tower-only variant | Package bytes | p50 latency | Relative L2 vs Core ML FP16 | Cosine |
|---|---:|---:|---:|---:|
| FP16 | 312,245,729 | 747.2 ms | 0 | 1.00000 |
| K-means INT4 | 77,771,129 | 748.0 ms | 22.38% | 0.97551 |

Projector removal reduces the earlier full-graph Q4 error from 35.96% to 22.38%,
showing that the projector amplified tower drift. It does not reveal a latency
gain: tower-only INT4 is effectively tied and slightly slower at p50. Core ML
FP16 is also about 2.1x slower than the qualified 352 ms optimized MLX encoder.
Both Core ML variants are rejected.

The temporary Core ML packages were not retained in the repository; the table
above records their measured size, latency, and numerical results.

## MLX CPU-Pressure Sweep

A fresh isolated environment used MLX 0.32.0, `mlx-vlm` 0.6.9, and the
`mlx-community/gemma-4-e4b-it-4bit` checkpoint. BF16 and affine-Q4/group-64
tower calls were interleaved under an ascending/descending schedule of
`0,2,4,8,4,2,0` CPU workers. Each worker ran `/usr/bin/yes > /dev/null`; each
pass used one warmup and five measured rounds for B1 and B8.

| Pass | CPU workers | B1 BF16 p50 | B1 Q4 p50 | B8 BF16 p50 | B8 Q4 p50 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 375.3 ms | 407.2 ms | 3,096.1 ms | 3,266.7 ms |
| 1 | 2 | 389.3 ms | 426.0 ms | 3,053.6 ms | 3,251.4 ms |
| 2 | 4 | 385.0 ms | 423.3 ms | 3,137.2 ms | 3,403.8 ms |
| 3 | 8 | 399.3 ms | 426.7 ms | 3,087.9 ms | 3,236.4 ms |
| 4 | 4 | 398.7 ms | 415.3 ms | 3,137.6 ms | 3,355.2 ms |
| 5 | 2 | 402.0 ms | 429.8 ms | 3,117.8 ms | 3,334.9 ms |
| 6 | 0 | 400.4 ms | 434.8 ms | 3,029.5 ms | 3,236.7 ms |

CPU load is not the dominant source of variance. B1's closing idle control was
6.7% slower than its opening idle control, and every pressured B1 observation
fell within that idle bracket. B8 showed no monotonic relationship with worker
count and varied only 3.6% across the complete schedule. The MLX process used
only about 14-15 ms of process CPU time during a 375-402 ms B1 wall-time call,
confirming this segmented encoder is primarily waiting on GPU work.

Q4 remained slower in every pass: 4.2-9.9% at B1 and 4.8-8.5% at B8. Its
pre-projector relative L2 error was 35.48% in this current MLX/checkpoint run.
Background CPU work can move individual samples, but it does not reverse the
BF16/Q4 decision. Thermal, run-order, and unrelated system pressure need
bracketed controls for absolute comparisons.

Artifact: `mlx-bf16-q4-cpu-pressure-sweep-5r.json`

## Clean BF16 Production Confirmation

The qualified pre-projector BF16 graph was rerun in isolation with no Q4 tower
resident and no interleaved comparison arm. The environment used MLX 0.32.0,
`mlx-vlm` 0.6.9, gathered positions, fused RoPE/layout, segment size 3 with
evaluated boundaries, and the 2 GiB wired limit. Each input received five
warmups followed by 20 measured rounds.

| Input | Visual tokens | Mean | p50 | p90 | Peak MLX memory |
|---|---:|---:|---:|---:|---:|
| Current 480p sample | 264 | 383.4 ms | 382.2 ms | 396.1 ms | 738.9 MB |
| Historical ChartQA control | 247 | 340.0 ms | 337.6 ms | 347.5 ms | 713.1 MB |

The 247-token result is directly comparable to the historical 345.7 ms
qualification and is 2.3% faster. The recent 375-400 ms observations were not a
lost optimization: the CPU-pressure sweep used the larger 264-token sample and
kept a second Q4 tower resident. Removing Q4 residency gives 382.2 ms for that
same larger input. Production remains the optimized BF16 MLX encoder.

Artifacts:

- `mlx-bf16-production-clean-20r.json`
- `mlx-bf16-production-chartqa-clean-20r.json`

## Current BF16 End-to-End Run

The H200 projector artifact was restarted with embeddings enabled, prefix
caching disabled, a 4,096-token model limit, and 50% GPU memory allocation. The
Mac used MLX 0.32.0 and `mlx-vlm` 0.6.9 to send 264 pooled `[N,768]` BF16 states
through the binary gateway over an SSH tunnel. One complete warmup preceded five
measured B1 requests; generation was limited to one token.

| Metric | p50 | p90 |
|---|---:|---:|
| Client preprocessing | 18.9 ms | 27.4 ms |
| M4 BF16 encoder | 384.5 ms | 402.3 ms |
| Serialization | 0.61 ms | 2.54 ms |
| Remote TTFT | 196.0 ms | 368.7 ms |
| Pipeline TTFT | 600.3 ms | 769.1 ms |
| One-token pipeline E2E | 690.4 ms | 769.1 ms |
| Batch wall time | 690.6 ms | 769.5 ms |

The request is 405,705 bytes and B1 throughput is 1.448 images/s at p50. Gateway
preparation is about 3.45 ms p50 and vLLM TTFT is about 20.84 ms p50. The M4
encoder agrees with the clean 382.2 ms standalone result; the larger remote TTFT
variance occurs outside H200 model execution, primarily in public-network/SSH
tunnel and streamed-event delivery.

Artifact: `e2e-bf16-production-clean-b1-5r.json`

### B8 Confirmation

The identical configuration was run with one batched M4 encoder call followed
by eight concurrent gateway requests, again with one warmup and five measured
rounds.

| Metric | B1 p50 | B8 p50 |
|---|---:|---:|
| M4 BF16 encoder | 384.5 ms | 2,937.2 ms |
| Encoder per image | 384.5 ms | 367.2 ms |
| Remote TTFT | 196.0 ms | 410.1 ms |
| Pipeline TTFT | 600.3 ms | 3,380.5 ms |
| One-token pipeline E2E | 690.4 ms | 3,380.6 ms |
| Batch wall time | 690.6 ms | 3,395.1 ms |
| Throughput | 1.448 images/s | 2.356 images/s |

B8 improves throughput by 62.7% over B1 and reduces amortized encoder time by
4.5%. Pipeline TTFT remains batch-synchronous: all eight requests are sent only
after the complete 2.94-second encoder call. Concurrent H200/gateway service
also raises remote TTFT, but M4 encoding remains the dominant B8 cost.

Artifact: `e2e-bf16-production-clean-b8-5r.json`

### Pipelined B8

Because native B8 encoding provides little amortized benefit, a second B8 run
encoded each image independently and submitted its request immediately while
the M4 encoded the next image. This overlaps H200/network work with M4 work
without changing model precision, token count, or payload format.

| Metric | Batch-synchronous B8 | Pipelined B8 | Change |
|---|---:|---:|---:|
| Pipeline TTFT p50 | 3,380.5 ms | 605.0 ms | -82.1% |
| Pipeline TTFT p90 | 3,511.6 ms | 666.0 ms | -81.0% |
| Batch wall p50 | 3,395.1 ms | 3,387.5 ms | -0.2% |
| Throughput p50 | 2.356 images/s | 2.362 images/s | +0.2% |
| Remote TTFT p50 | 410.1 ms | 197.2 ms | -51.9% |

Pipelining preserves total throughput while removing the batch barrier from
request latency. It is the preferred scheduling policy for independent
multi-request workloads. Native B8 remains appropriate only when a downstream
consumer requires one synchronized tensor batch.

Artifact: `e2e-bf16-production-pipeline-b8-5r.json`

## FP16 Vision-Tower Gate

The complete optimized vision tower was converted from BF16 weights and
activations to FP16, with its `[1,264,768]` output cast back to BF16 for the
existing transport contract. BF16 and FP16 calls were interleaved after five
warmups for 20 measured rounds.

| Metric | BF16 | FP16 to BF16 | Change |
|---|---:|---:|---:|
| Tower p50 | 377.8 ms | 509.8 ms | +35.0% |
| Tower p90 | 389.9 ms | 531.6 ms | +36.3% |
| Parameter bytes | 334,730,112 | 334,730,112 | 0% |

The FP16 output has 0.694% relative L2 error versus BF16, 0.999977 cosine
similarity, and 18.0% bit-equal values. Despite the relatively small numerical
drift, FP16 is materially slower on this M4 and is rejected. Production remains
BF16.

Artifact: `mlx-vision-bf16-vs-fp16-20r.json`

## Cider Private-ANE Gate

Cider's M5 TensorOps path does not apply to the M4, but its experimental private
ANE bridge can split one projection by output channel between ANE and GPU. The
65% ANE partition failed inference with `com.apple.appleneuralengine Code=8`
and `ANEProgramProcessRequestDirect() ... Program Inference error`. Reducing the
ANE partition to 25% produced a runnable `[2376,768] -> [2376,3072]` gate
projection, measured for 10 interleaved rounds after three warmups:

| Variant | p50 | p90 | Relative L2 vs BF16 |
|---|---:|---:|---:|
| MLX BF16 GPU | 4.63 ms | 5.13 ms | 0 |
| MLX FP16 GPU | 4.76 ms | 5.45 ms | 0.169% |
| 25% ANE + 75% GPU | 11.38 ms | 21.43 ms | 71.12% |

The runnable split is 2.46x slower than BF16 GPU and numerically invalid. The
bridge also eagerly synchronizes MLX and crosses through NumPy, so it cannot be
composed into the qualified segmented lazy encoder without additional stalls.
Both private-ANE configurations are rejected; no production code uses private
Apple frameworks.

Artifact: `mlx-ane-gpu-gate-projection-25pct-10r.json`

## Persistent Split Transport

A 20-round interleaved transport-only A/B sent the same 405,682-byte,
264-token payload through either a new HTTP connection per request or one
persistent connection. Gateway and vLLM execution were effectively unchanged,
isolating connection setup and tunnel transport.

| Metric (p50) | Fresh connection | Persistent connection | Change |
|---|---:|---:|---:|
| Remote TTFT | 212.9 ms | 154.0 ms | -27.7% |
| Remote one-token E2E | 267.7 ms | 168.1 ms | -37.2% |
| Gateway TTFT | 24.50 ms | 24.47 ms | -0.1% |
| vLLM TTFT | 21.02 ms | 20.73 ms | -1.4% |

Persistent connections remove 58.9 ms from p50 remote TTFT without changing
the request, model, precision, or generated-token limit. The production client
now maintains a bounded pool keyed by its configured concurrency. Connections
that fail are discarded rather than transparently replayed, because replaying a
partially sent generation request could duplicate remote work.

Artifact: `split-transport-fresh-vs-persistent-20r.json`

## Production Persistent Pipeline

The independent-request pipeline was integrated into
`MLXBinaryStreamingImageClient.complete_many()`. It retains sequential B1 M4
encoding, submits each completed tensor immediately through a bounded persistent
connection pool, preserves input result order, and cancels outstanding work on
failure. The per-request soft-token override is restored after encoding so one
call cannot alter later calls.

The B8 production run used `max_in_flight=4`, five measured rounds, the same
264-token BF16 encoder output, and the same H200 BF16 projector/model as the
earlier pipeline harness.

| Metric (p50) | Fresh-connection pipeline | Persistent production pipeline | Change |
|---|---:|---:|---:|
| Pipeline TTFT | 605.0 ms | 520.3 ms | -14.0% |
| Remote TTFT | 197.2 ms | 116.8 ms | -40.8% |
| Batch wall | 3,387.5 ms | 3,397.8 ms | +0.3% |
| Throughput | 2.362 images/s | 2.354 images/s | -0.3% |

The production scheduler preserves throughput while lowering per-request TTFT.
A five-round B1 control through the same API measured 479.1 ms p50 pipeline
TTFT, 97.2 ms remote TTFT, and 371.6 ms M4 encoding, compared with the previous
600.3 ms split B1 TTFT. This is a scheduling and transport optimization only:
the qualified BF16 graph, 264 visual tokens, 405,504-byte tensor, server
projector, and model are unchanged, so the existing quality qualification still
applies. The roughly 372 ms M4 encoder is now the dominant irreducible B1 cost.

Artifacts:

- `e2e-bf16-production-persistent-pipeline-b1-5r.json`
- `e2e-bf16-production-persistent-pipeline-b8-5r.json`

## Aligned ANE Projection Follow-up

Standalone private-ANE probes localized the earlier 71% hybrid error to a
sequence-layout constraint. Real Gemma gate weights and random, row-ramp,
channel-ramp, and sparse-impulse inputs were tested at five fixed lengths:

| Sequence | Sequence mod 64 | Max relative L2 | ANE p50 | MLX FP16 p50 |
|---:|---:|---:|---:|---:|
| 512 | 0 | 0.0572% | 0.642 ms | 1.447 ms |
| 2048 | 0 | 0.0574% | 1.792 ms | 2.268 ms |
| 2368 | 0 | 0.0574% | 2.287 ms | 2.857 ms |
| 2376 | 8 | 152.996% | 2.087 ms | 4.856 ms |
| 2432 | 0 | 0.0574% | 2.391 ms | 2.779 ms |

Every 64-aligned length is correct; only the production length 2,376 is
globally corrupted. Padding 2,376 rows to 2,432 before ANE execution and
trimming afterward fixes correctness. A 25% padded-ANE plus 75% GPU gate
projection then measured 0.004-0.097% relative L2 across the four patterns, but
failed performance after including conversion, synchronization, padding,
transpose, merge, and final synchronization:

| Mode | p50 | p90 |
|---|---:|---:|
| Full BF16 GPU, 3,072 channels | 8.712 ms | 11.868 ms |
| Padded ANE only, 768 channels | 10.344 ms | 14.337 ms |
| GPU only, 2,304 channels | 6.135 ms | 9.831 ms |
| Concurrent padded hybrid | 14.240 ms | 21.211 ms |

The padded hybrid is correct but 63.4% slower. Without true shared-buffer MLX
and ANE interoperability, private ANE remains rejected.

Artifacts:

- `ane-projection-probe/summary.json`
- `ane-projection-probe/padded-2376-hybrid-v1/summary.json`
- `ane-projection-probe/padded-2376-hybrid-v1/result.json`

## Experimental Relaxed-QKV Pipeline

The reassociated fused-QKV epilogue is the default accuracy-qualified client
path after a 100-case paired ChartQA gate improved exact match `36% -> 40%`,
relaxed accuracy `67% -> 73%`, and anywhere accuracy `92% -> 95%`. It produced
seven candidate-only versus one baseline-only relaxed-correct cases. Parsed
answers agreed in 91% of cases and full generations in 24%, so `--strict-qkv`
remains available when bit-exact arithmetic is required.

| Metric (p50) | Exact BF16 pipeline | Experimental QKV | Change |
|---|---:|---:|---:|
| B1 encoder | 371.6 ms | 332.0 ms | -10.6% |
| B8 summed encoder | 3,109.7 ms | 2,815.3 ms | -9.5% |
| B8 pipeline TTFT | 520.3 ms | 474.3 ms | -8.8% |
| B8 throughput | 2.354 images/s | 2.545 images/s | +8.1% |

The B1 experimental TTFT was 484.0 ms versus the earlier exact-path 479.1 ms,
because remote TTFT moved from 97.2 to 147.3 ms between non-interleaved runs.
The encoder and balanced quality-gate measurements, rather than that network
variation, establish the local speedup.

Artifacts:

- `e2e-qkv-epilogue-persistent-pipeline-b1-5r.json`
- `e2e-qkv-epilogue-persistent-pipeline-b8-5r.json`
- `../mlx-roofline/qkv-quality-gate/summary.json`
