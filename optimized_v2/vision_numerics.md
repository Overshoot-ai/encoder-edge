# MPS and CUDA Vision Numerics

The split deployment runs the vision embedder with Apple MPS, while the full
deployment runs it with CUDA. They use the same model weights and preprocessing,
but their visual embeddings are not bit-identical.

## Measurement

[`compare_vision_stages.py`](compare_vision_stages.py) executes the vision
embedder one stage at a time with deterministic PyTorch algorithms enabled. The
MPS tensors are saved and then compared with tensors produced from the same
image on CUDA.

The input hashes matched exactly:

```text
pixel_values SHA256:       72d355b7f7a9e091ebb5454b5273ddccb00e42e63d41e7818bfc21a01ab35136
image_position_ids SHA256: 5912b92d86062919ebc8d10df7ef9b96935727eac8dacd364ef880406e306a8d
```

The first difference appeared in `patch_ln1`, before the first dense matrix
multiplication:

| Stage | Differing values | Total values | Mean absolute difference | Maximum absolute difference |
| --- | ---: | ---: | ---: | ---: |
| Input patches | 0 | 1,935,360 | 0 | 0 |
| `patch_ln1` | 352 | 1,935,360 | 0.00000288 | 0.25 |
| `patch_dense` | 34,701 | 1,075,200 | 0.2135 | 512 |
| `patch_ln2` | 8,700 | 1,075,200 | 0.000529 | 2 |
| Positional embedding | 0 | 1,075,200 | 0 | 0 |
| Final projection | 159,776 | 1,075,200 | 0.000213 | 0.125 |

The end-to-end comparison retained 260 visual tokens. Of its 998,400 BF16
values, 158,836 differed, with a mean absolute difference of 0.000227833 and a
maximum difference of 0.125.

## Explanation

The first divergence is caused by backend-specific LayerNorm implementations,
not by different inputs or by the following matrix multiplication.

The version-matched follow-up used the same PyTorch 2.11.0 commit, Torchvision
0.26.0, and Transformers 5.12.1 on both machines. PyTorch still implements the
device backends separately:

- [MPS operation dispatch](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/mps/operations/Normalization.mm)
- [MPS Metal kernel](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/mps/kernels/LayerNorm.metal)
- [CUDA kernel](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/layer_norm_kernel.cu)

For this 6,912-element normalized row, the MPS dispatch code selects
`layer_norm_looped_bfloat` because `6912 > 1024 * 4`. It launches one
1,024-thread Metal threadgroup per row. Each thread reads groups of four BF16
values, converts them to FP32, and accumulates `sum(x)` and `sum(x^2)`. Metal
SIMD-group reductions combine those partials, after which the kernel computes
`variance = E[x^2] - E[x]^2`, applies `metal::precise::rsqrt`, and casts the
normalized result back to BF16.

The CUDA 2.11 implementation selects between a vectorized kernel and a rowwise
fallback. The vectorized path updates and merges Welford statistics with CUDA
warp shuffles and shared memory; the fallback uses `RowwiseMomentsCUDAKernel`
and `BlockReduce`. Which CUDA branch executed was not captured by this test, so
no narrower branch claim is made here. Neither branch is the Metal SIMD-group
raw-moment reduction above.

Floating-point addition is not associative. Different statistical algorithms
and reduction groupings can therefore produce different final FP32 bits for the
mean and variance even when all operands are identical. Casting the normalized
results to BF16 changed 352 values. The dense projection then spread and
amplified those initial differences.

`torch.use_deterministic_algorithms(True)` makes each backend repeatable. It
does not require independent MPS and CUDA kernels to produce identical bits.

## Consequence

The binary transport itself is bit-identical: the gateway reconstructs the
same BF16 tensor emitted by the Mac. Full-H200 and split inference nevertheless
begin language generation from slightly different visual embeddings because
the vision encoder ran on different compute backends.

On the first 200 ChartQA test samples, both deployments achieved 36.5% relaxed
accuracy, but only 133 parsed answers were identical and 24 samples changed
correctness outcome. See [`chartqa_benchmark.txt`](chartqa_benchmark.txt) for the
complete aggregate result.

## Version-Matched Follow-Up

After aligning both machines to PyTorch commit `70d99e998b4955e0049d13a98d77ae1b14db1f45`,
the first divergence remained in `patch_ln1`. On `sample.png`, 59 of 1,935,360
first-LayerNorm values differed. The final retained embedding had 89,282
differing BF16 values out of 1,013,760 (8.81%), so matching framework versions
did not produce bit identity. The original stage check used a different image,
so its differing-value counts are not a controlled before/after measurement.

A version-matched split run on the first 100 ChartQA samples scored 39%,
compared with 36% for the prior split run and 37% for the unchanged full-H200
records. See
[`pytorch_upgrade_benchmark.txt`](pytorch_upgrade_benchmark.txt) for the complete
stage, agreement, and latency results.
