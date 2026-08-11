import argparse
import importlib.metadata
import json
import math
import statistics
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm import load
from mlx_vlm.models.base import ensure_fused_sdpa
from mlx_vlm.models.gemma4.vision import apply_multidimensional_rope

from .mlx_vision_optimizations import optimize_gemma4_positions


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize(values: list[float]) -> dict:
    return {
        "mean_ms": statistics.mean(values),
        "p50_ms": percentile(values, 0.5),
        "p90_ms": percentile(values, 0.9),
        "min_ms": min(values),
        "max_ms": max(values),
        "raw_ms": values,
    }


def difference(reference: mx.array, candidate: mx.array) -> dict:
    mx.eval(reference, candidate)
    reference_bits = np.array(reference.view(mx.uint16), copy=True)
    candidate_bits = np.array(candidate.view(mx.uint16), copy=True)
    absolute = np.abs(
        np.array(reference.astype(mx.float32), copy=True)
        - np.array(candidate.astype(mx.float32), copy=True)
    )
    return {
        "bit_identical": bool(np.array_equal(reference_bits, candidate_bits)),
        "differing_values": int(np.count_nonzero(reference_bits != candidate_bits)),
        "mean_absolute_difference": float(absolute.mean()),
        "maximum_absolute_difference": float(absolute.max(initial=0)),
    }


def steel(projection, value):
    value = mx.clip(value, projection.input_min, projection.input_max)
    value = value @ projection.linear.weight.T
    return mx.clip(value, projection.output_min, projection.output_max)


def selective_nax(projection, value):
    return mx.fast.selective_nax_linear(
        value,
        projection.linear.weight,
        projection.input_min,
        projection.input_max,
        projection.output_min,
        projection.output_max,
    )


def measure_interleaved(functions: dict, rounds: int) -> dict:
    compiled = {name: mx.compile(function) for name, function in functions.items()}
    for function in compiled.values():
        for _ in range(3):
            mx.eval(function())
    mx.synchronize()
    timings = {name: [] for name in compiled}
    names = list(compiled)
    for round_index in range(rounds):
        order = names[round_index % len(names) :] + names[: round_index % len(names)]
        if round_index % 2:
            order.reverse()
        for name in order:
            started = time.perf_counter()
            mx.eval(compiled[name]())
            mx.synchronize()
            timings[name].append((time.perf_counter() - started) * 1000)
    return {name: summarize(values) for name, values in timings.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/gemma-4-e4b-it-4bit")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not hasattr(mx.fast, "selective_nax_linear"):
        raise RuntimeError("The layered selective-NAX MLX package is not active")

    mx.set_wired_limit(2 * 1024**3)
    model, _ = load(args.model)
    tower = model.vision_tower
    optimize_gemma4_positions(tower)
    pixels = mx.load(str(args.input))["pixels"]
    layer = tower.encoder.layers[0]

    _, _, height, width = pixels.shape
    length = (height // tower.patch_size) * (width // tower.patch_size)
    positions_np, padding_np, _ = tower._patch_positions_single(
        height, width, max_patches=length
    )
    positions = mx.array(positions_np[None])
    padding = mx.array(padding_np[None])
    hidden = tower.patch_embedder(pixels, positions, padding)
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
    attention_input = attention_input.transpose(0, 2, 1, 3).reshape(1, length, -1)
    attention_output = attention.o_proj(attention_input)
    residual = hidden + layer.post_attention_layernorm(attention_output)
    mlp_input = layer.pre_feedforward_layernorm(residual)
    gate = layer.mlp.gate_proj(mlp_input)
    up = layer.mlp.up_proj(mlp_input)
    down_input = nn.gelu_approx(gate) * up
    mx.eval(normalized, down_input)

    operations = {
        "q_proj": (attention.q_proj, normalized),
        "down_proj": (layer.mlp.down_proj, down_input),
    }
    direct = {}
    direct_gate_passed = True
    for name, (projection, value) in operations.items():
        functions = {
            "steel": lambda p=projection, x=value: steel(p, x),
            "selective_nax": lambda p=projection, x=value: selective_nax(p, x),
        }
        reference = functions["steel"]()
        candidate = functions["selective_nax"]()
        numerical = difference(reference, candidate)
        timings = measure_interleaved(functions, args.rounds)
        speedup = (
            timings["steel"]["p50_ms"] - timings["selective_nax"]["p50_ms"]
        ) / timings["steel"]["p50_ms"] * 100
        correctness_passed = (
            numerical["maximum_absolute_difference"] <= 0.125
            and numerical["mean_absolute_difference"] <= 0.005
        )
        speed_passed = speedup > 0
        direct_gate_passed &= correctness_passed and speed_passed
        direct[name] = {
            "shape_mkn": [value.shape[-2], value.shape[-1], projection.linear.weight.shape[0]],
            "numerical": numerical,
            "timings": timings,
            "selective_nax_p50_speedup_percent": speedup,
            "correctness_passed": correctness_passed,
            "speed_passed": speed_passed,
        }
        print(
            f"{name}: steel={timings['steel']['p50_ms']:.3f} ms "
            f"nax={timings['selective_nax']['p50_ms']:.3f} ms "
            f"max_abs={numerical['maximum_absolute_difference']:.6g}",
            flush=True,
        )

    result = {
        "metadata": {
            "model": args.model,
            "input": str(args.input),
            "rounds": args.rounds,
            "sequence_length": length,
            "mlx_version": importlib.metadata.version("mlx"),
            "device_info": mx.device_info(),
            "method": "Rotating and reversing interleaved order",
        },
        "direct_gate": {
            "passed": direct_gate_passed,
            "requirements": {
                "maximum_absolute_difference": 0.125,
                "mean_absolute_difference": 0.005,
                "p50_speedup_percent": "> 0 for both projections",
            },
            "results": direct,
        },
        "attribution_gate": {
            "executed": False,
            "reason": (
                "Direct Q/down gate failed; Q-only/down-only/Q+down attribution skipped."
                if not direct_gate_passed
                else "Not implemented because this harness is scoped to the direct gate."
            ),
        },
        "whole_encoder_gate": {
            "executed": False,
            "reason": "Direct speed and correctness must survive before encoder timing.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
