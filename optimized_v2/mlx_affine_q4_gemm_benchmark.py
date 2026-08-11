"""Isolated fused affine-Q4 Gemma 4 vision GEMM experiment.

This benchmark intentionally does not patch MLX or the model production path.
The custom Metal kernel consumes MLX's native affine-Q4 packing directly.
"""

import argparse
import importlib.metadata
import json
import math
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


METAL_HEADER = r"""
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

template <typename U, int GS>
METAL_FUNC U affine_q4_value(
    const device uint* packed,
    const device bfloat* scales,
    const device bfloat* biases,
    uint n,
    uint k,
    uint packed_k,
    uint groups_k) {
  uint word = packed[n * packed_k + (k >> 3)];
  uint q = (word >> ((k & 7u) * 4u)) & 15u;
  uint group = n * groups_k + k / GS;
  return static_cast<U>(q) * static_cast<U>(scales[group])
      + static_cast<U>(biases[group]);
}
"""


METAL_SOURCE = r"""
uint lane = thread_index_in_simdgroup;
uint simd = simdgroup_index_in_threadgroup;
uint block_m = threadgroup_position_in_grid.y * 32u;
uint block_n = threadgroup_position_in_grid.x * 32u;
uint m0 = block_m + (simd >> 1) * 16u;
uint n0 = block_n + (simd & 1u) * 16u;

uint qid = lane >> 2;
uint row = (qid & 4u) + ((lane >> 1) & 3u);
uint col = (qid & 2u) * 2u + (lane & 1u) * 2u;
uint packed_k = K / 8u;
uint groups_k = K / GroupSize;

simdgroup_matrix<float, 8, 8> c00;
simdgroup_matrix<float, 8, 8> c01;
simdgroup_matrix<float, 8, 8> c10;
simdgroup_matrix<float, 8, 8> c11;
c00.thread_elements()[0] = 0.0f;
c00.thread_elements()[1] = 0.0f;
c01.thread_elements()[0] = 0.0f;
c01.thread_elements()[1] = 0.0f;
c10.thread_elements()[0] = 0.0f;
c10.thread_elements()[1] = 0.0f;
c11.thread_elements()[0] = 0.0f;
c11.thread_elements()[1] = 0.0f;

for (uint kb = 0; kb < K; kb += 8u) {
  simdgroup_matrix<float, 8, 8> a0;
  simdgroup_matrix<float, 8, 8> a1;
  simdgroup_matrix<float, 8, 8> b0;
  simdgroup_matrix<float, 8, 8> b1;

  uint am0 = m0 + row;
  uint am1 = am0 + 8u;
  uint ak0 = kb + col;
  uint ak1 = ak0 + 1u;
  a0.thread_elements()[0] = am0 < M ? static_cast<float>(x[am0 * K + ak0]) : 0.0f;
  a0.thread_elements()[1] = am0 < M ? static_cast<float>(x[am0 * K + ak1]) : 0.0f;
  a1.thread_elements()[0] = am1 < M ? static_cast<float>(x[am1 * K + ak0]) : 0.0f;
  a1.thread_elements()[1] = am1 < M ? static_cast<float>(x[am1 * K + ak1]) : 0.0f;

  uint bk = kb + row;
  uint bn00 = n0 + col;
  uint bn01 = bn00 + 1u;
  uint bn10 = bn00 + 8u;
  uint bn11 = bn10 + 1u;
  b0.thread_elements()[0] = bn00 < N
      ? affine_q4_value<float, GroupSize>(w, scales, biases, bn00, bk, packed_k, groups_k)
      : 0.0f;
  b0.thread_elements()[1] = bn01 < N
      ? affine_q4_value<float, GroupSize>(w, scales, biases, bn01, bk, packed_k, groups_k)
      : 0.0f;
  b1.thread_elements()[0] = bn10 < N
      ? affine_q4_value<float, GroupSize>(w, scales, biases, bn10, bk, packed_k, groups_k)
      : 0.0f;
  b1.thread_elements()[1] = bn11 < N
      ? affine_q4_value<float, GroupSize>(w, scales, biases, bn11, bk, packed_k, groups_k)
      : 0.0f;

  simdgroup_matrix<float, 8, 8> d00;
  simdgroup_matrix<float, 8, 8> d01;
  simdgroup_matrix<float, 8, 8> d10;
  simdgroup_matrix<float, 8, 8> d11;
  simdgroup_multiply_accumulate(d00, a0, b0, c00);
  simdgroup_multiply_accumulate(d01, a0, b1, c01);
  simdgroup_multiply_accumulate(d10, a1, b0, c10);
  simdgroup_multiply_accumulate(d11, a1, b1, c11);
  c00 = d00;
  c01 = d01;
  c10 = d10;
  c11 = d11;
}

uint out_m0 = m0 + row;
uint out_m1 = out_m0 + 8u;
uint out_n0 = n0 + col;
uint out_n1 = out_n0 + 8u;
if (out_m0 < M && out_n0 < N) out[out_m0 * N + out_n0] = static_cast<bfloat>(c00.thread_elements()[0]);
if (out_m0 < M && out_n0 + 1u < N) out[out_m0 * N + out_n0 + 1u] = static_cast<bfloat>(c00.thread_elements()[1]);
if (out_m0 < M && out_n1 < N) out[out_m0 * N + out_n1] = static_cast<bfloat>(c01.thread_elements()[0]);
if (out_m0 < M && out_n1 + 1u < N) out[out_m0 * N + out_n1 + 1u] = static_cast<bfloat>(c01.thread_elements()[1]);
if (out_m1 < M && out_n0 < N) out[out_m1 * N + out_n0] = static_cast<bfloat>(c10.thread_elements()[0]);
if (out_m1 < M && out_n0 + 1u < N) out[out_m1 * N + out_n0 + 1u] = static_cast<bfloat>(c10.thread_elements()[1]);
if (out_m1 < M && out_n1 < N) out[out_m1 * N + out_n1] = static_cast<bfloat>(c11.thread_elements()[0]);
if (out_m1 < M && out_n1 + 1u < N) out[out_m1 * N + out_n1 + 1u] = static_cast<bfloat>(c11.thread_elements()[1]);
"""


_BASELINE_K_LOOP = r"""for (uint kb = 0; kb < K; kb += 8u) {
  simdgroup_matrix<float, 8, 8> a0;
  simdgroup_matrix<float, 8, 8> a1;
  simdgroup_matrix<float, 8, 8> b0;
  simdgroup_matrix<float, 8, 8> b1;

  uint am0 = m0 + row;
  uint am1 = am0 + 8u;
  uint ak0 = kb + col;
  uint ak1 = ak0 + 1u;
  a0.thread_elements()[0] = am0 < M ? static_cast<float>(x[am0 * K + ak0]) : 0.0f;
  a0.thread_elements()[1] = am0 < M ? static_cast<float>(x[am0 * K + ak1]) : 0.0f;
  a1.thread_elements()[0] = am1 < M ? static_cast<float>(x[am1 * K + ak0]) : 0.0f;
  a1.thread_elements()[1] = am1 < M ? static_cast<float>(x[am1 * K + ak1]) : 0.0f;

  uint bk = kb + row;
  uint bn00 = n0 + col;
  uint bn01 = bn00 + 1u;
  uint bn10 = bn00 + 8u;
  uint bn11 = bn10 + 1u;
  b0.thread_elements()[0] = bn00 < N
      ? affine_q4_value<float, GroupSize>(w, scales, biases, bn00, bk, packed_k, groups_k)
      : 0.0f;
  b0.thread_elements()[1] = bn01 < N
      ? affine_q4_value<float, GroupSize>(w, scales, biases, bn01, bk, packed_k, groups_k)
      : 0.0f;
  b1.thread_elements()[0] = bn10 < N
      ? affine_q4_value<float, GroupSize>(w, scales, biases, bn10, bk, packed_k, groups_k)
      : 0.0f;
  b1.thread_elements()[1] = bn11 < N
      ? affine_q4_value<float, GroupSize>(w, scales, biases, bn11, bk, packed_k, groups_k)
      : 0.0f;

  simdgroup_matrix<float, 8, 8> d00;
  simdgroup_matrix<float, 8, 8> d01;
  simdgroup_matrix<float, 8, 8> d10;
  simdgroup_matrix<float, 8, 8> d11;
  simdgroup_multiply_accumulate(d00, a0, b0, c00);
  simdgroup_multiply_accumulate(d01, a0, b1, c01);
  simdgroup_multiply_accumulate(d10, a1, b0, c10);
  simdgroup_multiply_accumulate(d11, a1, b1, c11);
  c00 = d00;
  c01 = d01;
  c10 = d10;
  c11 = d11;
}"""

_CACHED_QPARAM_K_LOOP = r"""uint bn00 = n0 + col;
uint bn01 = bn00 + 1u;
uint bn10 = bn00 + 8u;
uint bn11 = bn10 + 1u;
for (uint group_k = 0; group_k < groups_k; group_k++) {
  uint group_offset = group_k * GroupSize;
  float scale00 = bn00 < N ? static_cast<float>(scales[bn00 * groups_k + group_k]) : 0.0f;
  float scale01 = bn01 < N ? static_cast<float>(scales[bn01 * groups_k + group_k]) : 0.0f;
  float scale10 = bn10 < N ? static_cast<float>(scales[bn10 * groups_k + group_k]) : 0.0f;
  float scale11 = bn11 < N ? static_cast<float>(scales[bn11 * groups_k + group_k]) : 0.0f;
  float bias00 = bn00 < N ? static_cast<float>(biases[bn00 * groups_k + group_k]) : 0.0f;
  float bias01 = bn01 < N ? static_cast<float>(biases[bn01 * groups_k + group_k]) : 0.0f;
  float bias10 = bn10 < N ? static_cast<float>(biases[bn10 * groups_k + group_k]) : 0.0f;
  float bias11 = bn11 < N ? static_cast<float>(biases[bn11 * groups_k + group_k]) : 0.0f;

  for (uint group_kk = 0; group_kk < GroupSize; group_kk += 8u) {
    uint kb = group_offset + group_kk;
    simdgroup_matrix<float, 8, 8> a0;
    simdgroup_matrix<float, 8, 8> a1;
    simdgroup_matrix<float, 8, 8> b0;
    simdgroup_matrix<float, 8, 8> b1;
    volatile int compiler_barrier;

    uint am0 = m0 + row;
    uint am1 = am0 + 8u;
    uint ak0 = kb + col;
    uint ak1 = ak0 + 1u;
    a0.thread_elements()[0] = am0 < M ? static_cast<float>(x[am0 * K + ak0]) : 0.0f;
    a0.thread_elements()[1] = am0 < M ? static_cast<float>(x[am0 * K + ak1]) : 0.0f;
    a1.thread_elements()[0] = am1 < M ? static_cast<float>(x[am1 * K + ak0]) : 0.0f;
    a1.thread_elements()[1] = am1 < M ? static_cast<float>(x[am1 * K + ak1]) : 0.0f;

    uint bk = kb + row;
    uint shift = (bk & 7u) * 4u;
    uint q00 = bn00 < N ? (w[bn00 * packed_k + (bk >> 3)] >> shift) & 15u : 0u;
    uint q01 = bn01 < N ? (w[bn01 * packed_k + (bk >> 3)] >> shift) & 15u : 0u;
    uint q10 = bn10 < N ? (w[bn10 * packed_k + (bk >> 3)] >> shift) & 15u : 0u;
    uint q11 = bn11 < N ? (w[bn11 * packed_k + (bk >> 3)] >> shift) & 15u : 0u;
    b0.thread_elements()[0] = static_cast<float>(q00) * scale00 + bias00;
    b0.thread_elements()[1] = static_cast<float>(q01) * scale01 + bias01;
    b1.thread_elements()[0] = static_cast<float>(q10) * scale10 + bias10;
    b1.thread_elements()[1] = static_cast<float>(q11) * scale11 + bias11;

    simdgroup_matrix<float, 8, 8> d00;
    simdgroup_matrix<float, 8, 8> d01;
    simdgroup_matrix<float, 8, 8> d10;
    simdgroup_matrix<float, 8, 8> d11;
    simdgroup_multiply_accumulate(d00, a0, b0, c00);
    simdgroup_multiply_accumulate(d01, a0, b1, c01);
    simdgroup_multiply_accumulate(d10, a1, b0, c10);
    simdgroup_multiply_accumulate(d11, a1, b1, c11);
    c00 = d00;
    c01 = d01;
    c10 = d10;
    c11 = d11;
    (void)compiler_barrier;
  }
}"""

if METAL_SOURCE.count(_BASELINE_K_LOOP) != 1:
    raise RuntimeError("Failed to locate the baseline Q4 K loop")
METAL_SOURCE_CACHED = METAL_SOURCE.replace(_BASELINE_K_LOOP, _CACHED_QPARAM_K_LOOP)
METAL_SOURCE_CACHED_NO_FENCE = METAL_SOURCE_CACHED.replace(
    "    volatile int compiler_barrier;\n", ""
).replace("    (void)compiler_barrier;\n", "")
METAL_SOURCE_CACHED_SWIZZLED = METAL_SOURCE_CACHED.replace(
    "uint block_m = threadgroup_position_in_grid.y * 32u;\n"
    "uint block_n = threadgroup_position_in_grid.x * 32u;",
    "uint block_m = (threadgroup_position_in_grid.y * Swizzle + "
    "threadgroup_position_in_grid.x % Swizzle) * 32u;\n"
    "uint block_n = (threadgroup_position_in_grid.x / Swizzle) * 32u;\n"
    "if (block_m >= M) return;",
)


AFFINE_Q4_GEMM = mx.fast.metal_kernel(
    name="gemma4_vision_affine_q4_gemm_32x32",
    input_names=["x", "w", "scales", "biases"],
    output_names=["out"],
    header=METAL_HEADER,
    source=METAL_SOURCE,
)

AFFINE_Q4_GEMM_CACHED = mx.fast.metal_kernel(
    name="gemma4_vision_affine_q4_gemm_cached_32x32",
    input_names=["x", "w", "scales", "biases"],
    output_names=["out"],
    header=METAL_HEADER,
    source=METAL_SOURCE_CACHED,
)

AFFINE_Q4_GEMM_CACHED_NO_FENCE = mx.fast.metal_kernel(
    name="gemma4_vision_affine_q4_gemm_cached_no_fence_32x32",
    input_names=["x", "w", "scales", "biases"],
    output_names=["out"],
    header=METAL_HEADER,
    source=METAL_SOURCE_CACHED_NO_FENCE,
)

AFFINE_Q4_GEMM_CACHED_SWIZZLED = mx.fast.metal_kernel(
    name="gemma4_vision_affine_q4_gemm_cached_swizzled_32x32",
    input_names=["x", "w", "scales", "biases"],
    output_names=["out"],
    header=METAL_HEADER,
    source=METAL_SOURCE_CACHED_SWIZZLED,
)


def fused_affine_q4_gemm(x, packed, scales, biases, group_size):
    m, k = x.shape
    n = packed.shape[0]
    return AFFINE_Q4_GEMM(
        inputs=[x, packed, scales, biases],
        output_shapes=[(m, n)],
        output_dtypes=[mx.bfloat16],
        grid=(math.ceil(n / 32) * 128, math.ceil(m / 32), 1),
        threadgroup=(128, 1, 1),
        template=[
            ("M", m),
            ("K", k),
            ("N", n),
            ("GroupSize", group_size),
        ],
    )[0]


def fused_affine_q4_gemm_cached(
    x, packed, scales, biases, group_size, swizzle=1, compiler_fence=True
):
    m, k = x.shape
    n = packed.shape[0]
    if swizzle > 1:
        kernel = AFFINE_Q4_GEMM_CACHED_SWIZZLED
    elif compiler_fence:
        kernel = AFFINE_Q4_GEMM_CACHED
    else:
        kernel = AFFINE_Q4_GEMM_CACHED_NO_FENCE
    return kernel(
        inputs=[x, packed, scales, biases],
        output_shapes=[(m, n)],
        output_dtypes=[mx.bfloat16],
        grid=(math.ceil(n / 32) * swizzle * 128, math.ceil(math.ceil(m / 32) / swizzle), 1),
        threadgroup=(128, 1, 1),
        template=[
            ("M", m),
            ("K", k),
            ("N", n),
            ("GroupSize", group_size),
            ("Swizzle", swizzle),
        ],
    )[0]


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize(values):
    return {
        "mean_ms": statistics.mean(values),
        "p50_ms": percentile(values, 0.5),
        "p90_ms": percentile(values, 0.9),
        "min_ms": min(values),
        "max_ms": max(values),
        "raw_ms": values,
    }


def measure_interleaved(functions, warmups, rounds):
    compiled = {name: mx.compile(function) for name, function in functions.items()}
    for function in compiled.values():
        for _ in range(warmups):
            mx.eval(function())
    mx.synchronize()

    samples = {name: [] for name in compiled}
    names = list(compiled)
    for round_index in range(rounds):
        order = names[round_index % len(names) :] + names[: round_index % len(names)]
        if round_index & 1:
            order.reverse()
        for name in order:
            start = time.perf_counter_ns()
            mx.eval(compiled[name]())
            mx.synchronize()
            samples[name].append((time.perf_counter_ns() - start) / 1e6)
    return {name: summarize(values) for name, values in samples.items()}


def error_metrics(reference, candidate):
    mx.eval(reference, candidate)
    reference_f32 = np.asarray(reference.astype(mx.float32))
    candidate_f32 = np.asarray(candidate.astype(mx.float32))
    absolute = np.abs(reference_f32 - candidate_f32)
    reference_bits = np.asarray(reference.view(mx.uint16))
    candidate_bits = np.asarray(candidate.view(mx.uint16))
    denominator = np.linalg.norm(reference_f32.ravel())
    return {
        "bit_identical": bool(np.array_equal(reference_bits, candidate_bits)),
        "bit_identical_percent": float(np.mean(reference_bits == candidate_bits) * 100),
        "differing_values": int(np.count_nonzero(reference_bits != candidate_bits)),
        "mean_absolute_error": float(absolute.mean()),
        "max_absolute_error": float(absolute.max()),
        "relative_l2_error": float(np.linalg.norm(absolute.ravel()) / denominator),
    }


def run_case(m, k, n, group_size, warmups, rounds):
    mx.random.seed(17 + m + k + n + group_size)
    x = (mx.random.normal((m, k)) / math.sqrt(k)).astype(mx.bfloat16)
    weight = mx.random.normal((n, k)).astype(mx.bfloat16)
    packed, scales, biases = mx.quantize(
        weight, group_size=group_size, bits=4, mode="affine"
    )
    mx.eval(x, weight, packed, scales, biases)

    def custom():
        return fused_affine_q4_gemm(x, packed, scales, biases, group_size)

    def custom_cached():
        return fused_affine_q4_gemm_cached(
            x, packed, scales, biases, group_size
        )

    def custom_cached_no_fence():
        return fused_affine_q4_gemm_cached(
            x, packed, scales, biases, group_size, compiler_fence=False
        )

    def custom_cached_swizzled():
        return fused_affine_q4_gemm_cached(
            x, packed, scales, biases, group_size, swizzle=4
        )

    def mlx_q4():
        return mx.quantized_matmul(
            x,
            packed,
            scales,
            biases,
            transpose=True,
            group_size=group_size,
            bits=4,
            mode="affine",
        )

    def bf16():
        return x @ weight.T

    custom_output = custom()
    cached_output = custom_cached()
    cached_no_fence_output = custom_cached_no_fence()
    swizzled_output = custom_cached_swizzled()
    q4_output = mlx_q4()
    bf16_output = bf16()
    mx.eval(
        custom_output,
        cached_output,
        cached_no_fence_output,
        swizzled_output,
        q4_output,
        bf16_output,
    )
    errors = {
        "custom_vs_mlx_q4": error_metrics(q4_output, custom_output),
        "cached_vs_mlx_q4": error_metrics(q4_output, cached_output),
        "cached_no_fence_vs_mlx_q4": error_metrics(q4_output, cached_no_fence_output),
        "cached_swizzled_vs_mlx_q4": error_metrics(q4_output, swizzled_output),
        "cached_vs_custom": error_metrics(custom_output, cached_output),
        "cached_no_fence_vs_custom": error_metrics(custom_output, cached_no_fence_output),
        "cached_swizzled_vs_custom": error_metrics(custom_output, swizzled_output),
        "mlx_q4_vs_bf16": error_metrics(bf16_output, q4_output),
        "custom_vs_bf16": error_metrics(bf16_output, custom_output),
    }
    for candidate_name in (
        "custom_vs_mlx_q4",
        "cached_vs_mlx_q4",
        "cached_no_fence_vs_mlx_q4",
        "cached_swizzled_vs_mlx_q4",
    ):
        candidate_error = errors[candidate_name]
        if (
            not math.isfinite(candidate_error["relative_l2_error"])
            or candidate_error["relative_l2_error"] > 0.005
            or candidate_error["max_absolute_error"] > 0.05
        ):
            raise RuntimeError(
                f"{candidate_name} failed Q4 correctness for "
                f"{m}x{k}x{n} g{group_size}: {candidate_error}"
            )

    timings = measure_interleaved(
        {
            "custom_fused_q4": custom,
            "custom_cached_q4": custom_cached,
            "custom_cached_no_fence_q4": custom_cached_no_fence,
            "custom_cached_swizzled_q4": custom_cached_swizzled,
            "mlx_quantized_matmul": mlx_q4,
            "bf16_matmul": bf16,
        },
        warmups,
        rounds,
    )
    return {
        "shape": {"m": m, "k": k, "n": n},
        "group_size": group_size,
        "packed_shape": list(packed.shape),
        "timings": timings,
        "speedup_vs_mlx_q4": (
            timings["mlx_quantized_matmul"]["p50_ms"]
            / timings["custom_fused_q4"]["p50_ms"]
        ),
        "cached_speedup_vs_mlx_q4": (
            timings["mlx_quantized_matmul"]["p50_ms"]
            / timings["custom_cached_q4"]["p50_ms"]
        ),
        "cached_no_fence_speedup_vs_mlx_q4": (
            timings["mlx_quantized_matmul"]["p50_ms"]
            / timings["custom_cached_no_fence_q4"]["p50_ms"]
        ),
        "cached_swizzled_speedup_vs_mlx_q4": (
            timings["mlx_quantized_matmul"]["p50_ms"]
            / timings["custom_cached_swizzled_q4"]["p50_ms"]
        ),
        "errors": errors,
    }


def parse_shape(value):
    try:
        shape = tuple(int(part) for part in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must be MxKxN") from error
    if len(shape) != 3 or any(dimension <= 0 for dimension in shape):
        raise argparse.ArgumentTypeError("shape must be MxKxN with positive dimensions")
    if shape[1] % 64:
        raise argparse.ArgumentTypeError("K must be divisible by 64")
    return shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        nargs="+",
        type=parse_shape,
        default=[(264, 768, 768), (264, 768, 3072), (264, 3072, 768)],
        help="B1 MxKxN shapes (default: the three Gemma 4 vision projection shapes)",
    )
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8])
    parser.add_argument(
        "--group-sizes", nargs="+", type=int, choices=[32, 64], default=[32, 64]
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if any(batch <= 0 for batch in args.batches):
        parser.error("batches must be positive")

    results = []
    for base_m, k, n in args.shapes:
        for batch in args.batches:
            for group_size in args.group_sizes:
                case = run_case(
                    base_m * batch, k, n, group_size, args.warmups, args.rounds
                )
                case["batch"] = batch
                case["base_tokens"] = base_m
                results.append(case)
                timing = case["timings"]
                error = case["errors"]["custom_vs_mlx_q4"]
                print(
                    f"B{batch} {base_m * batch}x{k}x{n} g{group_size}: "
                    f"custom={timing['custom_fused_q4']['p50_ms']:.3f} ms, "
                    f"cached={timing['custom_cached_q4']['p50_ms']:.3f} ms, "
                    f"no-fence={timing['custom_cached_no_fence_q4']['p50_ms']:.3f} ms, "
                    f"swizzled={timing['custom_cached_swizzled_q4']['p50_ms']:.3f} ms, "
                    f"mlx_q4={timing['mlx_quantized_matmul']['p50_ms']:.3f} ms, "
                    f"bf16={timing['bf16_matmul']['p50_ms']:.3f} ms, "
                    f"q4 speedup={case['speedup_vs_mlx_q4']:.2f}x, "
                    f"bit equal={error['bit_identical_percent']:.2f}%, "
                    f"rel_l2={error['relative_l2_error']:.3e}",
                    flush=True,
                )

    report = {
        "metadata": {
            "mlx_version": importlib.metadata.version("mlx"),
            "device_info": mx.device_info(),
            "dtype": "bfloat16",
            "quantization": "affine_q4",
            "warmups": args.warmups,
            "rounds": args.rounds,
            "interleaved": True,
            "kernel_tile": "32x32 threadgroup, 16x16 per SIMD group, 8x8 MMA fragments",
        },
        "results": results,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
