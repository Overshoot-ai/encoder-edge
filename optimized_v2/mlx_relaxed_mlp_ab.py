"""Isolated native relaxed fused-MLP gate for Gemma 4 E4B vision."""

import argparse
import copy
import gc
import importlib.metadata
import json
import math
import os
import resource
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm import load
from mlx_vlm.models.base import ensure_fused_sdpa
from mlx_vlm.models.gemma4.vision import apply_multidimensional_rope

from .mlx_mixed_shape_benchmark import output_difference
from .mlx_vision_optimizations import (
    fuse_gemma4_qkv_epilogue,
    make_segmented_gemma4_encoder,
    optimize_gemma4_positions,
    prepare_gemma4_rope_constants,
)


MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
DEFAULT_INPUT = Path(
    "benchmark-results/mlx-roofline/mixed-shape-corpus/cases/"
    "chartqa-0000/input.safetensors"
)
DEFAULT_OUTPUT = Path(
    "benchmark-results/mlx-roofline/relaxed-mlp/composition-3arm-40r.json"
)
SOURCE_PATH = Path(os.environ.get("MLX_RELAXED_MLP_SOURCE", "mlx-relaxed-mlp-src"))
PACKAGE_PATH = Path(os.environ.get("MLX_RELAXED_MLP_PACKAGE", "mlx-relaxed-mlp-site"))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize(values: list[float]) -> dict:
    return {
        "rounds": len(values),
        "mean_ms": statistics.mean(values),
        "p50_ms": percentile(values, 0.5),
        "p90_ms": percentile(values, 0.9),
        "minimum_ms": min(values),
        "maximum_ms": max(values),
        "raw_ms": values,
    }


def memory_snapshot() -> dict:
    return {
        "active_bytes": mx.get_active_memory(),
        "cache_bytes": mx.get_cache_memory(),
        "peak_bytes": mx.get_peak_memory(),
        "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def measure_interleaved(
    functions: dict,
    warmups: int,
    rounds: int,
    cooldown: float = 0.0,
) -> tuple[dict, dict]:
    outputs = {}
    for name, function in functions.items():
        for _ in range(warmups):
            outputs[name] = function()
            mx.eval(outputs[name])
            mx.synchronize()
    if cooldown:
        time.sleep(cooldown)
    samples = {name: [] for name in functions}
    peaks = {name: [] for name in functions}
    names = list(functions)
    for round_index in range(rounds):
        shift = (round_index // 2) % len(names)
        order = names[shift:] + names[:shift]
        if round_index % 2:
            order.reverse()
        for name in order:
            mx.reset_peak_memory()
            started = time.perf_counter()
            outputs[name] = functions[name]()
            mx.eval(outputs[name])
            mx.synchronize()
            samples[name].append((time.perf_counter() - started) * 1000)
            peaks[name].append(mx.get_peak_memory())
    return (
        {
            name: {**summarize(values), "maximum_peak_bytes": max(peaks[name])}
            for name, values in samples.items()
        },
        outputs,
    )


def speedup(reference: dict, candidate: dict) -> dict:
    ratio = reference["p50_ms"] / candidate["p50_ms"]
    return {
        "ratio": ratio,
        "improvement_percent": (ratio - 1.0) * 100.0,
        "latency_change_percent": (1.0 / ratio - 1.0) * 100.0,
    }


def paired_summary(reference: dict, candidate: dict) -> dict:
    reference_values = reference["raw_ms"]
    candidate_values = candidate["raw_ms"]
    deltas = [
        candidate_value - reference_value
        for reference_value, candidate_value in zip(
            reference_values, candidate_values
        )
    ]
    wins = sum(delta < 0 for delta in deltas)
    ties = sum(delta == 0 for delta in deltas)
    half = max(5, len(reference_values) // 2)
    first_p50 = percentile(reference_values[:half], 0.5)
    last_p50 = percentile(reference_values[-half:], 0.5)
    return {
        "candidate_minus_stock_ms": {
            "mean": statistics.mean(deltas),
            "median": statistics.median(deltas),
            "p10": percentile(deltas, 0.1),
            "p90": percentile(deltas, 0.9),
            "raw": deltas,
        },
        "candidate_wins": wins,
        "ties": ties,
        "stock_wins": len(deltas) - wins - ties,
        "paired_median_improvement_percent": (
            -statistics.median(deltas) / reference["p50_ms"] * 100.0
        ),
        "stock_control_drift": {
            "first_half_p50_ms": first_p50,
            "last_half_p50_ms": last_p50,
            "last_vs_first_percent": (last_p50 / first_p50 - 1.0) * 100.0,
        },
    }


class RelaxedVisionMLP(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.gate_proj = mlp.gate_proj
        self.up_proj = mlp.up_proj
        self.down_proj = mlp.down_proj

    def product(self, value):
        gate = self.gate_proj(value)
        up_input = mx.clip(
            value,
            self.up_proj.input_min,
            self.up_proj.input_max,
        )
        return mx.fast.gemma_relaxed_gated_up(
            up_input,
            self.up_proj.linear.weight,
            gate,
            self.up_proj.output_min,
            self.up_proj.output_max,
        )

    def __call__(self, value):
        return self.down_proj(self.product(value))


class RelaxedPairedVisionMLP(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        if not mx.array_equal(
            mlp.gate_proj.input_min, mlp.up_proj.input_min
        ).item() or not mx.array_equal(
            mlp.gate_proj.input_max, mlp.up_proj.input_max
        ).item():
            raise ValueError("paired MLP requires identical gate/up input clips")
        self.gate_proj = mlp.gate_proj
        self.up_proj = mlp.up_proj
        self.down_proj = mlp.down_proj

    def product(self, value):
        value = mx.clip(
            value,
            self.gate_proj.input_min,
            self.gate_proj.input_max,
        )
        return mx.fast.gemma_relaxed_paired_mlp(
            value,
            self.gate_proj.linear.weight,
            self.up_proj.linear.weight,
            self.gate_proj.output_min,
            self.gate_proj.output_max,
            self.up_proj.output_min,
            self.up_proj.output_max,
        )

    def __call__(self, value):
        return self.down_proj(self.product(value))


def stock_product(mlp, value):
    return nn.gelu_approx(mlp.gate_proj(value)) * mlp.up_proj(value)


def representative_mlp_input(tower, pixels):
    _, _, height, width = pixels.shape
    length = (height // tower.patch_size) * (width // tower.patch_size)
    positions_np, padding_np, _ = tower._patch_positions_single(
        height,
        width,
        max_patches=length,
    )
    positions = mx.array(positions_np[None])
    padding = mx.array(padding_np[None])
    hidden = tower.patch_embedder(pixels, positions, padding)
    layer = tower.encoder.layers[0]
    normalized = layer.input_layernorm(hidden)
    attention = layer.self_attn
    q = attention.q_proj(normalized).reshape(1, length, 12, 64)
    k = attention.k_proj(normalized).reshape(1, length, 12, 64)
    v = attention.v_proj(normalized).reshape(1, length, 12, 64)
    q = apply_multidimensional_rope(
        attention.q_norm(q), positions, attention.rope_base_frequency
    ).transpose(0, 2, 1, 3)
    k = apply_multidimensional_rope(
        attention.k_norm(k), positions, attention.rope_base_frequency
    ).transpose(0, 2, 1, 3)
    v = attention._v_norm(v).transpose(0, 2, 1, 3)
    attention_input = ensure_fused_sdpa(q, k, v, scale=1.0, mask=None)
    attention_input = attention_input.transpose(0, 2, 1, 3).reshape(
        1, length, -1
    )
    attention_output = attention.o_proj(attention_input)
    residual = hidden + layer.post_attention_layernorm(attention_output)
    value = layer.pre_feedforward_layernorm(residual)
    mx.eval(value)
    return value


def wrap_relaxed_mlps(tower, wrapper) -> None:
    for layer in tower.encoder.layers:
        layer.mlp = wrapper(layer.mlp)


def layer_health(tower, pixels) -> list[dict]:
    _, _, height, width = pixels.shape
    patch_count = (height // tower.patch_size) * (width // tower.patch_size)
    positions_np, padding_np, _ = tower._patch_positions_single(
        height, width, max_patches=patch_count
    )
    positions = mx.array(positions_np[None])
    padding = mx.array(padding_np[None])
    rope = prepare_gemma4_rope_constants(tower, positions)
    hidden = tower.patch_embedder(pixels, positions, padding)
    records = []
    for index, layer in enumerate(tower.encoder.layers):
        hidden = layer(hidden, rope, None)
        mx.eval(hidden)
        mx.synchronize()
        finite = bool(mx.all(mx.isfinite(hidden)).item())
        records.append(
            {
                "layer": index,
                "finite": finite,
                "nan_count": int(mx.sum(mx.isnan(hidden)).item()),
                "inf_count": int(mx.sum(mx.isinf(hidden)).item()),
                "maximum_absolute_value": (
                    float(mx.max(mx.abs(hidden)).item()) if finite else None
                ),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--direct-rounds", type=int, default=40)
    parser.add_argument("--encoder-rounds", type=int, default=40)
    parser.add_argument("--cooldown", type=float, default=10.0)
    parser.add_argument("--wired-limit", type=int, default=2 * 1024**3)
    args = parser.parse_args()
    if args.warmups < 5 or args.direct_rounds < 40 or args.encoder_rounds < 40:
        raise ValueError("gates require >=5 warmups and >=40 direct/encoder rounds")
    required_primitives = (
        "gemma_relaxed_gated_up",
        "gemma_relaxed_paired_mlp",
    )
    if not all(hasattr(mx.fast, name) for name in required_primitives):
        raise RuntimeError("isolated MLX package with relaxed MLP primitives is required")

    mx.set_wired_limit(args.wired_limit)
    model, _ = load(args.model)
    tower = model.vision_tower
    optimize_gemma4_positions(tower)
    pixels = mx.load(str(args.input))["pixels"]
    mlp_input = representative_mlp_input(tower, pixels)
    stock_mlp = tower.encoder.layers[0].mlp
    candidate_mlp = RelaxedVisionMLP(stock_mlp)
    paired_mlp = RelaxedPairedVisionMLP(stock_mlp)
    if list(mlp_input.shape) != [1, 2223, 768]:
        raise RuntimeError(f"representative MLP shape mismatch: {mlp_input.shape}")
    del model
    gc.collect()
    mx.clear_cache()
    mx.synchronize()

    direct_product, product_outputs = measure_interleaved(
        {
            "stock": mx.compile(lambda: stock_product(stock_mlp, mlp_input)),
            "relaxed_up_epilogue": mx.compile(lambda: candidate_mlp.product(mlp_input)),
            "relaxed_paired_mlp": mx.compile(lambda: paired_mlp.product(mlp_input)),
        },
        args.warmups,
        args.direct_rounds,
    )
    direct_complete, complete_outputs = measure_interleaved(
        {
            "stock": mx.compile(lambda: stock_mlp(mlp_input)),
            "relaxed_up_epilogue": mx.compile(lambda: candidate_mlp(mlp_input)),
            "relaxed_paired_mlp": mx.compile(lambda: paired_mlp(mlp_input)),
        },
        args.warmups,
        args.direct_rounds,
    )
    product_speedups = {
        name: speedup(direct_product["stock"], direct_product[name])
        for name in ("relaxed_up_epilogue", "relaxed_paired_mlp")
    }
    passing_variants = [
        name
        for name, value in product_speedups.items()
        if value["improvement_percent"] >= 5.0
    ]
    selected_variant = min(
        passing_variants,
        key=lambda name: direct_product[name]["p50_ms"],
        default=None,
    )
    direct_gate_pass = selected_variant is not None
    result = {
        "metadata": {
            "benchmark": "gemma4_e4b_relaxed_fused_mlp_ab",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "input": str(args.input),
            "pixel_shape": list(pixels.shape),
            "mlp_input_shape": list(mlp_input.shape),
            "dtype": str(mlp_input.dtype),
            "warmups_per_arm": args.warmups,
            "direct_rounds_per_arm": args.direct_rounds,
            "encoder_rounds_per_arm": args.encoder_rounds,
            "cooldown_seconds_after_warmups": args.cooldown,
            "schedule": "rotating forward/reverse six-round cycle",
            "segment_size": 3,
            "evaluate_segments": True,
            "attention": "reassociated fused QKV epilogue",
            "mlx_version": importlib.metadata.version("mlx"),
            "mlx_vlm_version": importlib.metadata.version("mlx-vlm"),
            "device": str(mx.default_device()),
            "device_info": mx.device_info(),
            "source_path": str(SOURCE_PATH),
            "package_path": str(PACKAGE_PATH),
        },
        "implementation": {
            "variants": [
                "materialized clipped gate + fused up GEMM/clip/fast FP32 GELU/product",
                "shared-input paired gate/up GEMM + independent clips + fast FP32 GELU/product",
            ],
            "input_clipping": "stock gate and up bounds independently",
            "output_clipping": "stock gate projection and up projection bounds",
            "output_dtype": "bfloat16 after one FP32 epilogue conversion",
        },
        "direct": {
            "projection_product": direct_product,
            "complete_mlp_including_down": direct_complete,
            "projection_product_speedups": product_speedups,
            "complete_mlp_speedups": {
                name: speedup(direct_complete["stock"], direct_complete[name])
                for name in ("relaxed_up_epilogue", "relaxed_paired_mlp")
            },
            "projection_product_differences": {
                name: output_difference(product_outputs["stock"], product_outputs[name])
                for name in ("relaxed_up_epilogue", "relaxed_paired_mlp")
            },
            "complete_mlp_differences": {
                name: output_difference(complete_outputs["stock"], complete_outputs[name])
                for name in ("relaxed_up_epilogue", "relaxed_paired_mlp")
            },
            "selected_variant": selected_variant,
            "gate_definition": ">=5% projection-product p50 improvement",
            "gate_pass": direct_gate_pass,
        },
        "whole_encoder": {"status": "pending_composition_override"},
        "quality": {"status": "not_run_whole_encoder_gate_not_passed"},
        "promotion_decision": "do_not_promote",
        "memory_final": memory_snapshot(),
    }

    baseline_tower = tower
    materialized_tower = copy.deepcopy(tower)
    paired_tower = copy.deepcopy(tower)
    for current_tower in (baseline_tower, materialized_tower, paired_tower):
        fuse_gemma4_qkv_epilogue(current_tower)
    wrap_relaxed_mlps(materialized_tower, RelaxedVisionMLP)
    wrap_relaxed_mlps(paired_tower, RelaxedPairedVisionMLP)
    baseline_encode = make_segmented_gemma4_encoder(
        baseline_tower, None, 3, evaluate_segments=True
    )
    materialized_encode = make_segmented_gemma4_encoder(
        materialized_tower, None, 3, evaluate_segments=True
    )
    paired_encode = make_segmented_gemma4_encoder(
        paired_tower, None, 3, evaluate_segments=True
    )
    encoder_timing, encoder_outputs = measure_interleaved(
        {
            "qkv_default_stock_mlp": lambda: baseline_encode(pixels),
            "qkv_plus_materialized_gate": lambda: materialized_encode(pixels),
            "qkv_plus_paired_mlp": lambda: paired_encode(pixels),
        },
        args.warmups,
        args.encoder_rounds,
        args.cooldown,
    )
    baseline_name = "qkv_default_stock_mlp"
    candidate_names = (
        "qkv_plus_materialized_gate",
        "qkv_plus_paired_mlp",
    )
    finite_outputs = {
        name: bool(mx.all(mx.isfinite(value)).item())
        for name, value in encoder_outputs.items()
    }
    comparisons = {}
    advancing_candidates = []
    for name in candidate_names:
        candidate_speedup = speedup(encoder_timing[baseline_name], encoder_timing[name])
        paired = paired_summary(encoder_timing[baseline_name], encoder_timing[name])
        difference = output_difference(
            encoder_outputs[baseline_name], encoder_outputs[name]
        )
        healthy = bool(
            finite_outputs[baseline_name]
            and finite_outputs[name]
            and difference["nan_count"] == 0
            and difference["inf_count"] == 0
        )
        advances = bool(
            healthy
            and candidate_speedup["improvement_percent"] > 0.0
            and paired["candidate_minus_stock_ms"]["median"] < 0.0
        )
        comparisons[name] = {
            "speedup": candidate_speedup,
            "paired": paired,
            "feature_difference": difference,
            "healthy_features": healthy,
            "advances_to_quality": advances,
        }
        if advances:
            advancing_candidates.append(name)
    result["whole_encoder"] = {
        "status": "complete",
        "timing": encoder_timing,
        "finite_outputs": finite_outputs,
        "comparisons": comparisons,
        "gate_definition": "positive p50 and paired-median signal with healthy finite features",
        "advancing_candidates": advancing_candidates,
        "gate_pass": bool(advancing_candidates),
    }
    if not all(finite_outputs.values()):
        result["whole_encoder"]["layer_health"] = {
            "qkv_default_stock_mlp": layer_health(baseline_tower, pixels),
            "qkv_plus_materialized_gate": layer_health(materialized_tower, pixels),
            "qkv_plus_paired_mlp": layer_health(paired_tower, pixels),
        }
    result["quality"]["status"] = (
        "required_30_case_gate" if advancing_candidates
        else "not_run_whole_encoder_no_winner"
    )
    del baseline_encode, materialized_encode, paired_encode
    del materialized_tower, paired_tower
    gc.collect()
    mx.clear_cache()
    result["memory_final"] = memory_snapshot()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
