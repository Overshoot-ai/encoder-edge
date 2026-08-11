"""Isolated exact native RMSNorm-plus-residual experiment for Gemma 4 vision."""

import argparse
import copy
import gc
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm import load

from .mlx_vision_optimizations import (
    make_segmented_gemma4_encoder,
    optimize_gemma4_positions,
)


class RMSResidualVisionBlock(nn.Module):
    """Benchmark-only block wrapper that changes only the final norm/add pair."""

    def __init__(self, block):
        super().__init__()
        self.self_attn = block.self_attn
        self.mlp = block.mlp
        self.input_layernorm = block.input_layernorm
        self.post_attention_layernorm = block.post_attention_layernorm
        self.pre_feedforward_layernorm = block.pre_feedforward_layernorm
        self.post_feedforward_layernorm = block.post_feedforward_layernorm

    def __call__(self, x, positions, mask=None):
        normed = self.input_layernorm(x)
        attn_out = self.self_attn(normed, positions, mask)
        attn_out = self.post_attention_layernorm(attn_out)
        hidden = x + attn_out
        normed_hidden = self.pre_feedforward_layernorm(hidden)
        ffw_out = self.mlp(normed_hidden)
        norm = self.post_feedforward_layernorm
        return mx.fast.rms_norm_residual(
            ffw_out,
            hidden,
            norm.weight,
            norm.eps,
        )


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize(values: list[float]) -> dict:
    return {
        "mean_ms": statistics.mean(values),
        "p50_ms": statistics.median(values),
        "p90_ms": percentile(values, 0.9),
        "minimum_ms": min(values),
        "maximum_ms": max(values),
        "raw_ms": values,
    }


def difference(reference: mx.array, candidate: mx.array) -> dict:
    mx.eval(reference, candidate)
    reference_bits = np.array(reference.view(mx.uint16), copy=True)
    candidate_bits = np.array(candidate.view(mx.uint16), copy=True)
    differing = reference_bits != candidate_bits
    return {
        "bit_identical": bool(np.array_equal(reference_bits, candidate_bits)),
        "differing_values": int(np.count_nonzero(differing)),
        "max_ulp": int(
            np.max(
                np.abs(
                    reference_bits.astype(np.int32)
                    - candidate_bits.astype(np.int32)
                )
            )
        ),
    }


def gate_exactness() -> dict:
    mx.random.seed(20260806)
    cases = {
        "gemma_aligned": (1, 256, 1152),
        "row_and_axis_tail": (3, 257, 1153),
    }
    results = {}
    for name, shape in cases.items():
        x = mx.random.normal(shape).astype(mx.bfloat16)
        residual = mx.random.normal(shape).astype(mx.bfloat16)
        weight = mx.random.normal((shape[-1],)).astype(mx.bfloat16)
        reference = mx.fast.rms_norm(x, weight, 1e-6) + residual
        candidate = mx.fast.rms_norm_residual(x, residual, weight, 1e-6)
        result = difference(reference, candidate)
        result["shape"] = list(shape)
        results[name] = result
    passed = all(result["max_ulp"] == 0 for result in results.values())
    return {"passed": passed, "required_max_ulp": 0, "cases": results}


def balanced_benchmark(functions: dict, argument, rounds: int, warmups: int) -> tuple:
    outputs = {}
    for name, function in functions.items():
        for _ in range(warmups):
            outputs[name] = function(argument)
            mx.eval(outputs[name])
    mx.synchronize()

    timings = {name: [] for name in functions}
    names = list(functions)
    for round_index in range(rounds):
        order = names if round_index % 2 == 0 else list(reversed(names))
        for name in order:
            started = time.perf_counter()
            outputs[name] = functions[name](argument)
            mx.eval(outputs[name])
            mx.synchronize()
            timings[name].append((time.perf_counter() - started) * 1000)
    return {name: summarize(values) for name, values in timings.items()}, outputs


def measure_peak_memory(function, argument) -> int:
    mx.clear_cache()
    mx.reset_peak_memory()
    output = function(argument)
    mx.eval(output)
    mx.synchronize()
    return mx.get_peak_memory()


def isolated_benchmark(rounds: int, warmups: int) -> dict:
    mx.random.seed(20260806)
    shape = (1, 256, 1152)
    x = mx.random.normal(shape).astype(mx.bfloat16)
    residual = mx.random.normal(shape).astype(mx.bfloat16)
    weight = mx.random.normal((shape[-1],)).astype(mx.bfloat16)
    inputs = (x, residual, weight)

    def baseline(values):
        value, skip, scale = values
        return mx.fast.rms_norm(value, scale, 1e-6) + skip

    def candidate(values):
        value, skip, scale = values
        return mx.fast.rms_norm_residual(value, skip, scale, 1e-6)

    functions = {
        "stock_rms_then_residual": mx.compile(baseline),
        "native_rms_residual": mx.compile(candidate),
    }
    summaries, outputs = balanced_benchmark(functions, inputs, rounds, warmups)
    exactness = difference(
        outputs["stock_rms_then_residual"],
        outputs["native_rms_residual"],
    )
    baseline_p50 = summaries["stock_rms_then_residual"]["p50_ms"]
    candidate_p50 = summaries["native_rms_residual"]["p50_ms"]
    return {
        "shape": list(shape),
        "rounds": rounds,
        "warmups": warmups,
        "results": summaries,
        "exactness": exactness,
        "candidate_speedup": baseline_p50 / candidate_p50,
        "candidate_p50_delta_percent":
            (candidate_p50 / baseline_p50 - 1.0) * 100.0,
        "faster_and_exact": candidate_p50 < baseline_p50
        and exactness["max_ulp"] == 0,
        "peak_memory_bytes": {
            name: measure_peak_memory(function, inputs)
            for name, function in functions.items()
        },
    }


def install_wrappers(tower) -> int:
    layers = tower.encoder.layers
    if len(layers) != 16:
        raise RuntimeError(f"Expected 16 Gemma vision blocks, found {len(layers)}")
    for index, layer in enumerate(layers):
        layers[index] = RMSResidualVisionBlock(layer)
    return len(layers)


def encoder_benchmark(args, pixels) -> dict:
    model, _ = load(args.model)
    baseline_tower = model.vision_tower
    projector = model.embed_vision
    optimize_gemma4_positions(baseline_tower)
    candidate_tower = copy.deepcopy(baseline_tower)
    wrapped_blocks = install_wrappers(candidate_tower)
    del model
    gc.collect()
    mx.clear_cache()

    functions = {
        "stock_segment3": make_segmented_gemma4_encoder(
            baseline_tower,
            projector,
            3,
            evaluate_segments=True,
        ),
        "rms_residual_segment3": make_segmented_gemma4_encoder(
            candidate_tower,
            projector,
            3,
            evaluate_segments=True,
        ),
    }
    summaries, outputs = balanced_benchmark(
        functions,
        pixels,
        args.encoder_rounds,
        args.warmups,
    )
    exactness = difference(
        outputs["stock_segment3"],
        outputs["rms_residual_segment3"],
    )
    baseline_p50 = summaries["stock_segment3"]["p50_ms"]
    candidate_p50 = summaries["rms_residual_segment3"]["p50_ms"]
    return {
        "wrapped_blocks": wrapped_blocks,
        "segment_size": 3,
        "evaluate_segments": True,
        "pixel_shape": list(pixels.shape),
        "rounds": args.encoder_rounds,
        "warmups": args.warmups,
        "method": "Adjacent interleaved A/B with order reversed every round",
        "results": summaries,
        "exactness": exactness,
        "candidate_speedup": baseline_p50 / candidate_p50,
        "candidate_p50_delta_percent":
            (candidate_p50 / baseline_p50 - 1.0) * 100.0,
        "peak_memory_bytes": {
            name: measure_peak_memory(function, pixels)
            for name, function in functions.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-4-e4b-it-4bit",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--isolated-rounds", type=int, default=100)
    parser.add_argument("--encoder-rounds", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--wired-limit", type=int, default=2 * 1024**3)
    parser.add_argument("--expected-mlx-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    loaded_mlx = Path(mx.__file__).resolve()
    expected_root = args.expected_mlx_root.resolve()
    if expected_root not in loaded_mlx.parents:
        raise RuntimeError(
            f"Refusing non-isolated MLX at {loaded_mlx}; expected {expected_root}"
        )
    if not hasattr(mx.fast, "rms_norm_residual"):
        raise RuntimeError("Isolated MLX lacks mx.fast.rms_norm_residual")

    mx.set_wired_limit(args.wired_limit)
    gate = gate_exactness()
    if not gate["passed"]:
        raise RuntimeError(f"Exactness gate failed: {gate}")
    isolated = isolated_benchmark(args.isolated_rounds, args.warmups)

    result = {
        "metadata": {
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "mlx_version": mx.__version__,
            "mlx_path": str(loaded_mlx),
            "device": str(mx.default_device()),
            "device_info": mx.device_info(),
            "model": args.model,
            "input": str(args.input),
            "wired_limit_bytes": args.wired_limit,
        },
        "exactness_gate": gate,
        "isolated": isolated,
        "encoder": None,
        "decision": "reject_before_encoder",
    }
    if isolated["faster_and_exact"]:
        pixels = mx.load(str(args.input))["pixels"]
        result["encoder"] = encoder_benchmark(args, pixels)
        encoder_exact = result["encoder"]["exactness"]["max_ulp"] == 0
        encoder_faster = (
            result["encoder"]["candidate_p50_delta_percent"] < 0
        )
        result["decision"] = (
            "promote_benchmark_candidate"
            if encoder_exact and encoder_faster
            else "reject_after_encoder"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
