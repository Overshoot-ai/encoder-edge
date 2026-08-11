# Gemma 4 In-Tower Token Reduction

## Decision

Do not promote any training-free in-tower token-reduction candidate. Fixed 3x3
pooling provides large M4 latency wins but destroys ChartQA accuracy. Progressive
cell-local ToMe preserves substantially more quality, but even the most
conservative schedule regresses on the binding 100-case gate.

Production remains the QKV-default segmented encoder with stock terminal pooling.

## Fixed 3x3 Pooling

Each candidate replaces the full patch sequence with one unscaled spatial mean per
existing 3x3 terminal-pool cell, then runs the remaining blocks on those tokens.
All candidates preserve finite BF16 `[cells, 768]` output and the H200 projector
contract.

| Pool after block | Encoder p50 | Speedup vs 584.19 ms control | 30-case relaxed accuracy | Paired relaxed baseline-only / candidate-only | Decision |
|---:|---:|---:|---:|---:|---|
| 12 | 465.41 ms | 20.3% | 73.3% -> 3.3% | 21 / 0 | Reject |
| 10 | 398.44 ms | 31.8% | 73.3% -> 10.0% | 19 / 0 | Reject |
| 8 | 334.82 ms | 42.7% | 73.3% -> 0.0% | 22 / 0 | Reject |
| 6 | 272.90 ms | 53.3% | 73.3% -> 3.3% | 21 / 0 | Reject |

The earliest fixed candidate already has mean relative feature L2 drift of about
`0.704`. Moving the same irreversible average earlier only increases the damage.

## Progressive Cell-Local ToMe

The benchmark-only ToMe path restricts matching to each terminal 3x3 cell. It uses
hidden-state cosine similarity, bipartite soft matching, size-weighted averages,
and destination integer coordinates for subsequent 2D RoPE. It does not use global
merges or proportional-attention changes. Root-hidden-size scaling and
standardization occur exactly once after the final merge.

| Schedule, tokens per cell | Encoder p50 A/B | Paired wins | Quality cases | Relaxed accuracy A/B | Baseline-only / candidate-only | Decision |
|---|---:|---:|---:|---:|---:|---|
| after 8: 6; after 12: 3; after 16: 1 | 538.56 / 405.85 ms | 40 / 40 | 30 | 73.3% / 50.0% | 8 / 1 | Reject |
| after 12: 6; after 14: 3; after 16: 1 | 566.70 / 529.29 ms | 34 / 40 | 30 | 73.3% / 66.7% | 3 / 1 | Reject |
| after 14: 6; after 15: 3; after 16: 1 | 633.17 / 600.10 ms | 24 / 40 | 30 | 73.3% / 73.3% | 0 / 0 | Extend |
| after 14: 6; after 15: 3; after 16: 1 | same performance run | same run | 100 | 73.0% / 64.0% | 12 / 3 | Reject |
| same, proportional attention + centroid member coordinates | 327.19 / 321.45 ms | 26 / 40 | 100 | 73.0% / 65.0% | 11 / 3 | Reject |
| after 15: 6; after 16: 3 then 1, same corrections | 333.78 / 336.80 ms | 17 / 40 | Not run | Not run | Not run | Reject on latency |

The last-two schedule's 30-case result was a false positive. Its 100-case result is
the promotion decision. Its modest and noisy latency signal would not compensate
for the nine-point relaxed-accuracy regression even if the quality gate allowed a
small loss.

Adding ToMe proportional attention (`log(token_size)` as a key bias) and retaining
the cluster member nearest the weighted spatial centroid recovered only one relaxed
accuracy point at 100 cases. It also reduced the p50 latency signal to 1.8%, with
only 26 of 40 paired wins. Deferring the first merge until immediately before block
16 reduced feature drift substantially but was 1.4% slower by mean and 0.9% slower
by p50 because matching and masked-attention overhead exceeded the saved work.

Absolute timing controls vary across runs because these experiments were executed
in separate thermal sessions. Decisions use within-run interleaved paired results,
not cross-run absolute controls.

## Artifacts

- `performance.json`: fixed-pooling numerical and 40-round performance sweep.
- `quality/pool-after-*/summary.json`: fixed-pooling 30-case quality summaries.
- `../progressive-tome/performance.json`: late/safe ToMe performance and numerical evidence.
- `../progressive-tome/very-late-performance.json`: very-late ToMe evidence.
- `../progressive-tome/last-two-performance.json`: last-two ToMe evidence.
- `../progressive-tome/last-two-proportional-centroid-performance.json`: corrected last-two evidence.
- `../progressive-tome/final-block-proportional-centroid-performance.json`: safest schedule evidence.
- `../progressive-tome/quality*/summary.json`: staged ToMe quality summaries.

## Next Step

Further in-tower reduction requires adaptation rather than another training-free
schedule. Follow the research ladder: tune the H200 projector and post-merge
normalization first, then consider LoRA for tower blocks after the first merge.
Keep the training-free production encoder unchanged until an adapted checkpoint
passes the same 100-case gate and a repeated latency run.
