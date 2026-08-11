# Encoder-Side Token Reduction for the Gemma 4 Vision Tower

Research checked against primary papers and official repositories on 2026-08-07.

## Bottom line

For this Gemma tower, start with **fixed-shape, weighted token merging constrained to each existing final 3x3 pooling cell**. Use ToMe's key similarity and size-weighted averaging first; compare PiToMe's energy-preserving matcher second. Do not begin with unrestricted global ToMe: Gemma applies 2D RoPE in every block, while published ToMe image results mainly use ViTs whose absolute position is added before the blocks. A globally merged Gemma token has no unambiguous rotary position. Cell-local merging bounds that error, preserves the 264-token spatial/output contract, and turns the existing terminal average into the final merge.

The simplest control is to move Gemma's existing 3x3 average from after block 16 to after block 12, 8, or 4. It has no matching overhead and static shapes, but is likely less accurate than gradual similarity-based merging. For low fine-tuning, train the balanced cell-local schedule with frozen or low-LR tower weights and tune the multimodal projector.

The local 480p path is especially worth optimizing: it uses a 36x66 patch grid, processes **2,376 tokens in every one of 16 blocks**, then averages to 12x22 = **264 tokens**. Measured on the Apple M4, the compiled tower/projector takes 369.36 ms p50; isolated block p50s sum to 385.06 ms, versus 1.07 ms for pooling and 0.61 ms for projection. Token work inside the encoder is therefore the dominant target ([local benchmark](../benchmark-results/mlx-token-budget/local-target-tiers-b1-b8.json), [stage profile](../benchmark-results/mlx-metal-profile/stage-profile-5r.json)).

## Scope classification

“Visual token reduction” is used for three materially different locations:

1. **Inside the ViT:** saves subsequent attention, QKV/output projections, and MLP work. This is the target here.
2. **After the complete vision encoder / in a resampler or projector:** saves projector or LLM work, but not ViT work.
3. **Inside the language model:** saves later LLM layers, but not the ViT or early LLM layers.

Only category 1 reduces Gemma's 16-layer tower compute. Reported speed and accuracy values below are paper-specific and are not directly transferable across hardware, image resolution, or VLM benchmarks.

## Methods that reduce encoder compute

| Method | Reduction location and operation | Training | Reported tradeoff | Supported/evaluated towers | Implementation |
|---|---|---|---|---|---|
| **ToMe / Token Merging** | Between attention and MLP in each block; bipartite soft matching on mean keys, size-weighted merge; constant or decreasing `r` schedule | None required; fine-tuning improves quality | Off-the-shelf ViT-L/512 and ViT-H/518: about 2x throughput for 0.2-0.3 top-1 point loss. Official timm table: ViT-L/512 12.8 to 26.3 img/s, 88.06 to 87.80; ViT-H/518 4.7 to 9.8, 88.55 to 88.25. Smaller 224px models lose more. | Standard dense ViTs; official patches for timm, SWAG, MAE; image, video, audio. Not validated for 2D-RoPE Gemma. | [Paper](https://arxiv.org/abs/2210.09461), [official repo](https://github.com/facebookresearch/ToMe), [block patch](https://github.com/facebookresearch/ToMe/blob/main/tome/patch/timm.py) |
| **PiToMe** | After successive transformer layers; energy score preserves isolated/informative tokens and merges large similar clusters | Off-the-shelf; training also supported | Saves 40-60% FLOPs. At comparable compression, paper reports mean ViT-MAE-H classification loss of 0.5 point versus 2.6 for baselines, and CLIP Flickr30k retrieval loss of 0.3 versus 4.5. | MAE ViTs, CLIP, LAVIS BLIP/BLIP2/ALBEF, BERT; paper includes LLaVA VQA. No Gemma/2D-RoPE implementation. | [Paper](https://arxiv.org/abs/2405.16148), [official repo](https://github.com/hchautran/PiToMe) |
| **EViT** | At selected blocks between MHSA and FFN; class-token attention keeps attentive patches and fuses all inattentive patches into one surrogate | Fine-tunes/trains the ViT | DeiT-S: 50% faster with 0.3-point top-1 loss. Official 0.7 keep model: 4.6G to 3.0G MACs, 79.8 to 79.5, 4,385 img/s. | DeiT-S; requires a CLS token and classification attention signal. | [Paper](https://arxiv.org/abs/2202.07800), [official repo](https://github.com/youweiliang/evit) |
| **DynamicViT** | Lightweight predictors at several layers progressively prune tokens; differentiable attention masks during training, physical removal at inference | Yes, typically 15-30 epoch fine-tune | Pruning 66% of tokens reduces 31-37% FLOPs, improves throughput over 40%, and stays within 0.5 point. Official DeiT-S `rho=.7`: 4.6G to 2.9G, 79.8 to 79.32. | DeiT, LV-ViT; journal extension covers Swin and ConvNeXt. | [Paper](https://arxiv.org/abs/2106.02034), [official repo](https://github.com/raoyongming/DynamicViT) |
| **A-ViT** | Per-token Adaptive Computation Time; tokens halt at learned depths, using an existing embedding channel as the halting score | Yes, paper/repo use 100-epoch ImageNet fine-tune | DeiT-T throughput +62%, DeiT-S +38%, each around 0.3-point loss. Official A-ViT-S: 78.8 top-1, 3.6G FLOPs versus DeiT-S near 79.8/4.6G. | DeiT-T/S classification. Dynamic halting is awkward for fixed-shape accelerators; repo notes runtime token zipping was not yet released. | [Paper](https://arxiv.org/abs/2112.07658), [official repo](https://github.com/NVlabs/A-ViT) |
| **TokenLearner** | One learned spatial-attention module at an intermediate depth converts hundreds of grid tokens to 8-16 learned tokens for all later blocks | Yes; changes architecture | ViT-B/16 example: 55.6 to 28.7 GFLOPs, 84.73 to 83.65; variants with extra layers reached 85.21-85.45 at 47 GFLOPs. ViT-L 16-token module at layer 12: 363.1 to 184.6 GFLOPs and 87.35 to 87.68. | ViT and ViViT; image/video; expects image-shaped features and learned conv/MLP attention maps. | [Paper](https://arxiv.org/abs/2106.11297), [official Scenic code](https://github.com/google-research/scenic/tree/main/scenic/projects/token_learner) |
| **Token Pooling** | Usually after blocks; weighted K-medoids/K-means clusters tokens and emits cluster centers to minimize reconstruction error | Trained with pooling; not presented as an off-the-shelf patch | Same DeiT-Ti top-1 with 42% fewer computations; strongest reported tradeoff uses weighted K-medoids. Clustering overhead is included. | Standard ViT/DeiT classification. No clear maintained official code repository was found in the primary paper. | [Paper](https://arxiv.org/abs/2110.03860), [WACV paper](https://openaccess.thecvf.com/content/WACV2023/html/Marin_Token_Pooling_in_Vision_Transformers_for_Image_Classification_WACV_2023_paper.html) |
| **PPT** | Per-layer adaptive choice between significance pruning and similarity pooling; variance of significance scores selects the policy | Training-free or trainable | Official off-the-shelf DeiT-S example reports 2.944G MACs and 79.498 top-1, close to full DeiT-S accuracy at roughly 64% of baseline MACs. | DeiT/timm-style ViTs. | [Paper](https://arxiv.org/abs/2310.01812), [official repo](https://github.com/xjwu1024/PPT) |
| **FiCoCo-V** | Training-free filtering and correlation-based recycling inside selected vision-encoder layers; FiCoCo-L is the separate LLM variant | None | FiCoCo family reports up to 14.7x FLOP reduction with 93.6% performance retention; repo's common encoder setting starts compression at zero-based layer 11 and removes 42 tokens/layer. Figures combine model/task settings, so do not treat 14.7x as tower-only speed. | Implemented around LLaVA/CLIP-style encoders; paper discusses a no-CLS alternative. | [Paper](https://arxiv.org/abs/2411.17686), [official repo](https://github.com/kawhiiiileo/FiCoCo) |

### Practical ranking for Gemma

1. **Cell-local ToMe-style weighted merging:** best training-free starting point. Minimal parameters, fixed token counts, and the final output can remain exactly 264 spatially ordered vectors.
2. **Cell-local PiToMe matching:** likely better at preserving uncommon details, but its advantage was measured on much larger global token sets; energy ranking among only nine cell members may add overhead without helping.
3. **Move the existing fixed 3x3 average earlier:** essential low-overhead control and possibly the production winner on Apple hardware.
4. **FiCoCo-V/PPT:** promising research comparisons, but substantially more control logic and less direct support for Gemma's grid, RoPE, and static compilation.
5. **DynamicViT/EViT/A-ViT/TokenLearner/Token Pooling:** require task training or classification-specific signals and are not first choices for preserving a pretrained VLM tower.

## VLM methods that do not reduce encoder compute

| Method | Actual reduction point | Training | Reported result | Encoder support / implementation |
|---|---|---|---|---|
| **VisionZip** | Selects dominant tokens and merges contextual tokens from a selected completed encoder layer, before the LLM | Training-free; optional 30-minute projector tuning on 8 A800s | LLaVA-1.5 576 to 192 tokens: 98.5% mean normalized performance training-free; 64 tokens: 94.2%, or 95.2% with projector tuning. LLaVA-NeXT 2,880 to 160: 7.8x prefill and 3x total-time improvement. | CLIP and no-CLS SigLIP logic; [paper](https://arxiv.org/abs/2412.04467), [repo](https://github.com/dvlab-research/VisionZip). It still runs the full encoder to obtain final attention/features. |
| **PruMerge / PruMerge+** | Uses penultimate/final CLIP attention and keys after vision encoding, then sends fewer tokens to projector/LLM | Training-free or LoRA fine-tune | PruMerge averages about 32/576 tokens but has noticeable benchmark losses; PruMerge+ uses about 25% and is closer to baseline. Paper reports about 4x reduction with comparable/better aggregate results and 4-10x LLM-prefill FLOP savings. | CLIP with CLS; [paper](https://arxiv.org/abs/2403.15388), [repo](https://github.com/42Shawn/LLaVA-PruMerge). No tower FLOPs saved. |
| **FastV** | Ranks visual tokens from LLM attention at language layer `K` (typically after layer 2), then removes them from later LLM layers | None | 45% theoretical LLaVA-1.5-13B FLOP reduction without aggregate loss; repo reports only about 8% latency benefit with KV cache for normal images, up to 25% for video. | LLaVA, Qwen-VL-Chat, Video-LLaVA; [paper](https://arxiv.org/abs/2403.06764), [repo](https://github.com/pkunlp-icler/FastV). Full encoder and first LLM layers still run. |
| **SparseVLM** | Text-guided progressive pruning/recycling inside LLM self-attention | None | LLaVA: 54% FLOP reduction, 37% CUDA-time reduction, 97% retained accuracy. | LLaVA-family implementation; [paper](https://proceedings.mlr.press/v267/zhang25s.html), [repo](https://github.com/Gumpest/SparseVLMs). No encoder saving. |
| **MQT** | A learned query transformer compresses completed encoder embeddings to the first `m` of `M` latent queries | Full multimodal pretraining/fine-tuning | Matches LLaVA-1.5 across 11 benchmarks at 256 versus 576 tokens; 16 tokens gives 8x lower reported TFLOPs and -2.4 MMBench points; some tasks tolerate 2 tokens. | LLaVA + learned Q-former; [paper](https://arxiv.org/abs/2405.19315), [repo](https://github.com/gordonhu608/MQT-LLaVA). The pretrained vision tower remains dense. |
| **M3 / Matryoshka Multimodal Models** | Learns nested coarse-to-fine subsets/pools from completed visual features before the LLM | Visual instruction fine-tuning | COCO-style tasks reportedly need about 9 of 576 tokens for similar accuracy; one checkpoint serves multiple budgets. | LLaVA-1.5/NeXT; [paper](https://arxiv.org/abs/2405.17430), [repo](https://github.com/mu-cai/matryoshka-mm). Saves downstream, not tower, compute. |

These methods can still be combined with encoder-side merging if downstream LLM prefill, transport size, or KV cache is also a goal. VisionZip is the strongest training-free post-encoder baseline in this set, but it does not solve the measured local tower bottleneck.

## Gemma-specific design

### Why stock global merging is risky

The current implementation adds 2D position embeddings before block 1 and passes 2D rotary positions to every self-attention block ([fixed wrapper](../optimized_v2/coreml_gemma4_e4b_vision.py#L81-L105)). ToMe's official implementation merges arbitrary tokens and retains target rows after attention, an operation that is natural when position was already embedded once. For Gemma, subsequent blocks need a rotary coordinate for each merged row. Choosing one target coordinate biases a potentially nonlocal merged feature; averaging coordinates is also not equivalent to rotating each source feature.

Restrict source/target matching to the nine patches belonging to one terminal pooling cell. Retain the target patch's integer coordinate (prefer the member nearest the size-weighted coordinate centroid), carry a `token_size`, and use size-weighted feature averaging. If proportional attention is enabled, add `log(token_size)` to attention logits as in ToMe. At the final stage, merge every cell to one token in row-major order and bypass the old pooler; the output remains `[1, 264, 768]` before projection and `[1, 264, 2560]` after it.

For the first implementation, use hidden-state cosine similarity or mean normalized K vectors. Reusing K is algorithmically faithful to ToMe but the current MLX attention API does not return K, so returning it may inhibit fusion. Benchmark hidden-state matching and K matching separately rather than assuming the latter is faster.

### Schedules to test

Layer numbers below are one-based and a merge “after block N” affects block `N+1`. Counts are for the 480p 36x66 grid. The cost ratio is the average token ratio across blocks, a rough proxy for the dominant linear projections/MLPs; actual attention is partly quadratic and merge/gather overhead is excluded.

| Test | Merge events per 3x3 cell | Global tokens entering block ranges | Approx. linear block-cost ratio | Ideal block-only upper speedup |
|---|---|---|---:|---:|
| Baseline | Terminal average only | blocks 1-16: 2,376 | 1.000 | 1.00x |
| **Late/safe** | after 8: 9→6; after 12: 6→3; after 16: 3→1 | 1-8: 2,376; 9-12: 1,584; 13-16: 792 | 0.750 | 1.33x |
| **Balanced, recommended** | after 4: 9→6; after 8: 6→3; after 12: 3→1 | 1-4: 2,376; 5-8: 1,584; 9-12: 792; 13-16: 264 | 0.528 | 1.89x |
| **Aggressive** | after 2: 9→6; after 6: 6→3; after 10: 3→1 | 1-2: 2,376; 3-6: 1,584; 7-10: 792; 11-16: 264 | 0.417 | 2.40x |

Also run these fixed-pooling controls:

| Control | Token count after pool | Approx. linear block-cost ratio | Ideal block-only upper speedup |
|---|---:|---:|---:|
| Pool after block 12 | 264 | 0.778 | 1.29x |
| **Pool after block 8** | 264 | 0.556 | 1.80x |
| Pool after block 4 | 264 | 0.333 | 3.00x |

The balanced schedule is the primary quality/speed candidate. The block-8 fixed pool is the primary implementation-overhead control. Avoid content-dependent output counts initially: they complicate batching, MLX compilation/cache reuse, Core ML static export, and downstream framing.

### Fine-tuning ladder

1. Evaluate all schedules training-free.
2. Tune only the multimodal projector and any normalization immediately after pooling/merging.
3. If needed, LoRA-tune attention/MLP projections in blocks after the first merge while keeping the patch embedder and early blocks frozen.
4. Only then fine-tune the whole tower with schedule sampling among late, balanced, aggressive, and baseline. This is the closest useful analogue to Matryoshka training while reducing encoder work.

Use a weighted mixture of OCR/document, chart/UI, counting/localization, general VQA, and captioning data. Classification top-1 evidence is not enough: pruning methods favor dominant objects, while Gemma must preserve small text and spatial detail.

## Evaluation gates

Record each result against the unchanged model with the same image preprocessing:

- Tower and end-to-end p50/p90 latency, peak memory, and compile time at batch 1 and target batch sizes.
- Actual token shapes per segment and merge overhead as its own synchronized stage.
- ChartQA, TextVQA, DocVQA/OCRBench, MMMU/MMBench, POPE, counting, and spatial/grounding scores.
- Output cosine/relative-L2 only as a debugging signal, not a quality substitute.
- Per-cell source maps and the rate at which small text/high-frequency patches are merged early.
- Static versus content-adaptive schedules on the same hardware; theoretical FLOPs alone are not acceptance evidence.

Promote a schedule only if it gives a real compiled latency reduction. The existing profile shows roughly 24 ms per full-token block, but gather/scatter, sorting, dynamic shape recompilation, or exposing K can erase theoretical savings on Apple GPU/ANE.

## Experimental outcome

Training-free evaluation rejected every proposed schedule. Fixed 3x3 pooling after
blocks 12, 10, 8, and 6 improved representative M4 encoder p50 by 20-53%, but
reduced 30-case ChartQA relaxed accuracy from 73.3% to 0-10%. Cell-local ToMe was
less destructive: the late/safe schedule improved p50 by 24.6% but scored 50.0%,
and a very-late schedule improved p50 by 6.6% but scored 66.7%. The most
conservative merges after blocks 14, 15, and 16 matched 73.3% on the 30-case gate,
then failed the required 100-case extension at 64% versus the 73% baseline.
Adding proportional attention and centroid-member coordinates raised the 100-case
candidate to only 65% and reduced its p50 speed signal to 1.8%. A safer schedule
that reduced only block 16 input was slower than baseline because merge and
masked-attention overhead exceeded the compute saving.

These results rule out production promotion without adaptation. The next justified
experiment is projector/post-merge-normalization tuning, followed by LoRA on tower
blocks after the first merge. See
[`benchmark-results/mlx-roofline/early-pool/REPORT.md`](../benchmark-results/mlx-roofline/early-pool/REPORT.md)
for the complete measured frontier and artifact paths.

## Source notes

- “Token Merging” and “ToMe” refer to the same primary method, not two independent papers.
- Speed figures above preserve the papers' units and baselines. “Accuracy loss” means absolute points unless a paper reports normalized retained performance.
- VisionZip, PruMerge, MQT, and M3 inspect or consume completed vision features. Their excellent visual-token compression results should not be cited as evidence of vision-encoder acceleration.
- FastV and SparseVLM operate in the language backbone. Their names and diagrams can make this easy to miss.
- The newer FiCoCo-V is the clearest VLM-specific method in this review that actually reduces tokens within a pretrained vision encoder without training, making it the most relevant external comparison after ToMe/PiToMe.
