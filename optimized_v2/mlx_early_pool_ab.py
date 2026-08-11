"""Performance and numerical A/B for fixed 3x3 Gemma 4 early pooling."""

import argparse
import gc
import importlib.metadata
import json
import math
import resource
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

from .mlx_mixed_shape_benchmark import output_difference
from .mlx_vision_optimizations import (
    fuse_gemma4_qkv_epilogue,
    make_segmented_gemma4_encoder,
    optimize_gemma4_positions,
    prepare_gemma4_rope_constants,
)


MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
DEFAULT_CORPUS = Path("benchmark-results/mlx-roofline/mixed-shape-corpus")
DEFAULT_OUTPUT = Path("benchmark-results/mlx-roofline/early-pool/performance.json")
DEFAULT_REPRESENTATIVE = Path(
    "benchmark-results/mlx-roofline/early-pool/representative-480p.safetensors"
)
POOL_POINTS = (12, 10, 8, 6)


@dataclass(frozen=True)
class SpatialPoolPlan:
    patch_height: int
    patch_width: int
    pooled_height: int
    pooled_width: int
    positions: mx.array
    padding: mx.array
    valid: mx.array

    @property
    def output_length(self) -> int:
        return self.pooled_height * self.pooled_width


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def timing_summary(values: list[float]) -> dict:
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


def thermal_state() -> str:
    result = subprocess.run(
        ["pmset", "-g", "therm"], check=False, capture_output=True, text=True
    )
    return (result.stdout + result.stderr).strip()


def make_spatial_pool_plan(
    patch_height: int,
    patch_width: int,
    padding: np.ndarray | None = None,
    kernel: int = 3,
) -> SpatialPoolPlan:
    if patch_height <= 0 or patch_width <= 0:
        raise ValueError("Patch geometry must be positive")
    if patch_height % kernel or patch_width % kernel:
        raise ValueError(
            f"Unsupported {patch_height}x{patch_width} patch grid: both dimensions "
            f"must be divisible by the {kernel}x{kernel} spatial pool; refusing "
            "a layout that could cross row boundaries"
        )
    patch_count = patch_height * patch_width
    if padding is None:
        padding = np.zeros((1, patch_count), dtype=bool)
    padding = np.asarray(padding, dtype=bool)
    if padding.ndim == 1:
        padding = padding[None]
    if padding.ndim != 2 or padding.shape[0] != 1 or padding.shape[1] < patch_count:
        raise ValueError("Padding must be [1, sequence] and cover the full patch grid")
    if np.any(~padding[0, patch_count:]):
        raise ValueError("Non-padding tokens after the declared spatial grid are unsupported")

    valid = (~padding[:, :patch_count]).reshape(
        1, patch_height // kernel, kernel, patch_width // kernel, kernel
    )
    counts = valid.sum(axis=(2, 4))
    if np.any(counts == 0):
        raise ValueError("A pooled cell has no valid patches")
    y, x = np.meshgrid(
        np.arange(kernel // 2, patch_height, kernel, dtype=np.int32),
        np.arange(kernel // 2, patch_width, kernel, dtype=np.int32),
        indexing="ij",
    )
    positions = np.stack((x.reshape(-1), y.reshape(-1)), axis=-1)[None]
    pooled_padding = counts.reshape(1, -1) == 0
    return SpatialPoolPlan(
        patch_height=patch_height,
        patch_width=patch_width,
        pooled_height=patch_height // kernel,
        pooled_width=patch_width // kernel,
        positions=mx.array(positions),
        padding=mx.array(pooled_padding),
        valid=mx.array(valid),
    )


def fixed_spatial_mean(hidden: mx.array, plan: SpatialPoolPlan) -> mx.array:
    """Unscaled spatial mean in row-major cell order, excluding invalid patches."""
    patch_count = plan.patch_height * plan.patch_width
    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] < patch_count:
        raise ValueError(
            f"Expected hidden [1, >= {patch_count}, width], got {hidden.shape}"
        )
    width = hidden.shape[-1]
    cells = hidden[:, :patch_count].reshape(
        1,
        plan.pooled_height,
        3,
        plan.pooled_width,
        3,
        width,
    )
    valid = mx.expand_dims(plan.valid, -1)
    summed = mx.sum(mx.where(valid, cells.astype(mx.float32), 0.0), axis=(2, 4))
    counts = mx.sum(plan.valid, axis=(2, 4)).astype(mx.float32)
    pooled = summed / mx.expand_dims(counts, -1)
    return pooled.reshape(1, plan.output_length, width).astype(hidden.dtype)


def _segments(layers, start: int, stop: int, segment_size: int):
    compiled = []
    labels = []
    for offset in range(start, stop, segment_size):
        current_stop = min(offset + segment_size, stop)
        current_layers = tuple(layers[offset:current_stop])

        def run(hidden, cosine, sine, selected=current_layers):
            for layer in selected:
                hidden = layer(hidden, (cosine, sine), None)
            return hidden

        compiled.append(mx.compile(run))
        labels.append(f"blocks_{offset + 1}_{current_stop}")
    return compiled, labels


def make_early_pool_encoder(
    tower,
    projector,
    pool_after: int | None,
    *,
    segment_size: int = 3,
    evaluate_segments: bool = True,
):
    """Build the production-equivalent stock path or one fixed early-pool path."""
    layer_count = len(tower.encoder.layers)
    if pool_after is not None and pool_after not in POOL_POINTS:
        raise ValueError(f"pool_after must be one of {POOL_POINTS}")
    if pool_after is not None and pool_after >= layer_count:
        raise ValueError("Early pool must occur before the final transformer block")

    patch = mx.compile(
        lambda pixels, positions, padding: tower.patch_embedder(
            pixels, positions, padding
        )
    )
    split = layer_count if pool_after is None else pool_after
    pre_segments, pre_labels = _segments(
        tower.encoder.layers, 0, split, segment_size
    )
    post_segments, post_labels = _segments(
        tower.encoder.layers, split, layer_count, segment_size
    )

    def stock_finish(hidden, positions, padding):
        output_length = hidden.shape[1] // (tower.pooling_kernel_size**2)
        hidden, _ = tower.pooler(
            hidden, positions, padding, output_length=output_length
        )
        if tower.config.standardize:
            hidden = (hidden - tower.std_bias) * tower.std_scale
        return hidden if projector is None else projector(hidden)

    def early_finish(hidden):
        hidden = hidden * tower.pooler.root_hidden_size
        if tower.config.standardize:
            hidden = (hidden - tower.std_bias) * tower.std_scale
        return hidden if projector is None else projector(hidden)

    stock_finish = mx.compile(stock_finish)
    early_finish = mx.compile(early_finish)
    rope_cache = {}
    last_trace = []

    def encode(pixels, trace: bool = False):
        if pixels.ndim != 4 or pixels.shape[0] != 1:
            raise ValueError("Fixed early pooling currently requires NCHW batch size 1")
        _, _, height, width = pixels.shape
        if height % tower.patch_size or width % tower.patch_size:
            raise ValueError("Pixel dimensions must be divisible by the patch size")
        patch_height = height // tower.patch_size
        patch_width = width // tower.patch_size
        patch_count = patch_height * patch_width
        positions_np, padding_np, _ = tower._patch_positions_single(
            height, width, max_patches=patch_count
        )
        positions = mx.array(positions_np[None])
        padding = mx.array(padding_np[None])
        key = (height, width)
        if key not in rope_cache:
            original_rope = prepare_gemma4_rope_constants(tower, positions)
            if original_rope is None:
                raise RuntimeError("Early-pool encoder requires fused QKV RoPE constants")
            plan = make_spatial_pool_plan(patch_height, patch_width, padding_np)
            reduced_rope = prepare_gemma4_rope_constants(tower, plan.positions)
            pool_function = mx.compile(
                lambda hidden, current_plan=plan: fixed_spatial_mean(
                    hidden, current_plan
                )
            )
            rope_cache[key] = (original_rope, plan, reduced_rope, pool_function)
        original_rope, plan, reduced_rope, pool_function = rope_cache[key]

        trace_values = []

        def measured(label, function, *values):
            started = time.perf_counter()
            result = function(*values)
            if evaluate_segments or trace:
                mx.eval(result)
                mx.synchronize()
            if trace:
                trace_values.append(
                    {"stage": label, "elapsed_ms": (time.perf_counter() - started) * 1000}
                )
            return result

        hidden = measured("patch_embed", patch, pixels, positions, padding)
        for label, segment in zip(pre_labels, pre_segments):
            hidden = measured(label, segment, hidden, *original_rope)
        if pool_after is None:
            output = measured("stock_pool_finish", stock_finish, hidden, positions, padding)
        else:
            hidden = measured(
                f"spatial_pool_after_{pool_after}", pool_function, hidden
            )
            for label, segment in zip(post_labels, post_segments):
                hidden = measured(label, segment, hidden, *reduced_rope)
            output = measured("scale_standardize_finish", early_finish, hidden)
        if trace:
            last_trace[:] = trace_values
        return output

    encode.last_trace = last_trace
    encode.variant = "baseline" if pool_after is None else f"pool_after_{pool_after}"
    return encode


def _run_interleaved(functions: dict, pixels, warmups: int, rounds: int) -> dict:
    for _ in range(warmups):
        for function in functions.values():
            mx.eval(function(pixels))
            mx.synchronize()
    timings = {name: [] for name in functions}
    peaks = {name: [] for name in functions}
    names = list(functions)
    for round_index in range(rounds):
        rotation = round_index % len(names)
        order = names[rotation:] + names[:rotation]
        if round_index % 2:
            order.reverse()
        for name in order:
            mx.reset_peak_memory()
            started = time.perf_counter()
            value = functions[name](pixels)
            mx.eval(value)
            mx.synchronize()
            timings[name].append((time.perf_counter() - started) * 1000)
            peaks[name].append(mx.get_peak_memory())
    return {"timings": timings, "peak_memory_bytes": peaks}


def _paired_summary(baseline: list[float], candidate: list[float]) -> dict:
    deltas = [candidate_ms - baseline_ms for baseline_ms, candidate_ms in zip(baseline, candidate)]
    return {
        "mean_delta_ms": statistics.mean(deltas),
        "p50_delta_ms": percentile(deltas, 0.5),
        "candidate_wins": sum(delta < 0 for delta in deltas),
        "baseline_wins": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "raw_delta_ms": deltas,
    }


def _real_win(baseline: dict, candidate: dict, paired: dict) -> bool:
    return bool(
        candidate["p50_ms"] < baseline["p50_ms"]
        and candidate["mean_ms"] < baseline["mean_ms"]
        and paired["candidate_wins"] > paired["baseline_wins"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--representative-input", type=Path, default=DEFAULT_REPRESENTATIVE
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--cooldown", type=float, default=20.0)
    parser.add_argument("--drift-threshold-percent", type=float, default=3.0)
    args = parser.parse_args()
    if args.warmups < 5 or args.rounds < 40:
        raise ValueError("Performance gate requires at least 5 warmups and 40 rounds")

    mx.set_wired_limit(2 * 1024**3)
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    model, processor = load(args.model)
    if not args.representative_input.exists():
        source_case = manifest["cases"][0]
        source = Image.open(
            args.corpus / "cases" / source_case["case_id"] / "source.png"
        ).convert("RGB")
        image = ImageOps.fit(
            source, (854, 480), method=Image.Resampling.LANCZOS
        )
        prompt = apply_chat_template(
            processor, model.config, "Describe this image.", num_images=1
        )
        pixels = prepare_inputs(
            processor,
            images=[image],
            prompts=prompt,
            add_special_tokens=False,
        )["pixel_values"]
        args.representative_input.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(args.representative_input), {"pixels": pixels})
    else:
        pixels = mx.load(str(args.representative_input))["pixels"]
    mx.eval(pixels)
    if list(pixels.shape) != [1, 3, 576, 1056]:
        raise RuntimeError(
            f"Representative input must produce the 36x66 grid, got {pixels.shape}"
        )
    representative = {
        "case_id": "cached-480p",
        "pixel_shape": list(pixels.shape),
        "patch_grid": [36, 66],
    }

    tower = model.vision_tower
    optimize_gemma4_positions(tower)
    fuse_gemma4_qkv_epilogue(tower)
    production = make_segmented_gemma4_encoder(
        tower, projector=None, segment_size=3, evaluate_segments=True
    )
    functions = {
        "baseline": make_early_pool_encoder(tower, None, None),
        **{
            f"pool_after_{point}": make_early_pool_encoder(tower, None, point)
            for point in POOL_POINTS
        },
    }
    del model
    gc.collect()
    mx.clear_cache()
    mx.synchronize()

    production_output = production(pixels)
    baseline_output = functions["baseline"](pixels)
    mx.eval(production_output, baseline_output)
    mx.synchronize()
    production_difference = output_difference(production_output, baseline_output)
    production_match = production_difference["differing_values"] == 0
    if not production_match:
        raise RuntimeError("Experimental baseline does not match current production encoder")

    outputs = {"baseline": baseline_output}
    numerical = {}
    for name, function in functions.items():
        value = baseline_output if name == "baseline" else function(pixels)
        mx.eval(value)
        mx.synchronize()
        outputs[name] = value
        if value.shape != baseline_output.shape or value.dtype != mx.bfloat16:
            raise RuntimeError(f"Invalid {name} output contract: {value.shape} {value.dtype}")
        metrics = output_difference(baseline_output, value)
        metrics["finite"] = metrics["nan_count"] == 0 and metrics["inf_count"] == 0
        numerical[name] = metrics
        if not metrics["finite"]:
            raise RuntimeError(f"Non-finite output from {name}")

    geometry_checks = []
    for patch_height, patch_width in sorted(
        {tuple(case["patch_grid"]) for case in manifest["cases"]}
    ):
        plan = make_spatial_pool_plan(patch_height, patch_width)
        expected = patch_height * patch_width // 9
        if plan.output_length != expected:
            raise RuntimeError("Early-pool token count differs from stock pool token count")
        geometry_checks.append(
            {
                "patch_grid": [patch_height, patch_width],
                "pooled_grid": [plan.pooled_height, plan.pooled_width],
                "tokens": plan.output_length,
            }
        )

    thermal_before_cooldown = thermal_state()
    time.sleep(args.cooldown)
    thermal_before = thermal_state()
    interleaved = _run_interleaved(functions, pixels, args.warmups, args.rounds)
    thermal_after = thermal_state()
    summaries = {
        name: {
            **timing_summary(values),
            "maximum_peak_memory_bytes": max(interleaved["peak_memory_bytes"][name]),
            "output_shape": list(outputs[name].shape),
            "output_dtype": str(outputs[name].dtype),
        }
        for name, values in interleaved["timings"].items()
    }
    paired = {}
    real_winners = []
    baseline_raw = interleaved["timings"]["baseline"]
    half = len(baseline_raw) // 2
    first_half = statistics.median(baseline_raw[:half])
    second_half = statistics.median(baseline_raw[half:])
    drift_percent = abs(second_half - first_half) / first_half * 100
    for point in POOL_POINTS:
        name = f"pool_after_{point}"
        paired[name] = _paired_summary(baseline_raw, interleaved["timings"][name])
        if _real_win(summaries["baseline"], summaries[name], paired[name]):
            real_winners.append(name)

    repeats = {}
    if drift_percent > args.drift_threshold_percent and real_winners:
        time.sleep(args.cooldown)
        for name in real_winners:
            repeated = _run_interleaved(
                {"baseline": functions["baseline"], name: functions[name]},
                pixels,
                args.warmups,
                args.rounds,
            )
            repeated_summaries = {
                arm: timing_summary(values)
                for arm, values in repeated["timings"].items()
            }
            repeated_paired = _paired_summary(
                repeated["timings"]["baseline"], repeated["timings"][name]
            )
            repeats[name] = {
                "timing": repeated_summaries,
                "paired": repeated_paired,
                "real_win": _real_win(
                    repeated_summaries["baseline"],
                    repeated_summaries[name],
                    repeated_paired,
                ),
            }
        real_winners = [name for name in real_winners if repeats[name]["real_win"]]

    segment_timings = {}
    for name, function in functions.items():
        mx.eval(function(pixels, trace=True))
        mx.synchronize()
        segment_timings[name] = list(function.last_trace)

    result = {
        "metadata": {
            "benchmark": "gemma4_fixed_early_pool_performance_ab",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "mlx_version": importlib.metadata.version("mlx"),
            "mlx_vlm_version": importlib.metadata.version("mlx-vlm"),
            "device_info": mx.device_info(),
            "representative_case": representative["case_id"],
            "pixel_shape": representative["pixel_shape"],
            "patch_grid": representative["patch_grid"],
            "warmups": args.warmups,
            "rounds": args.rounds,
            "method": "rotating/reversing interleaved arms with synchronization",
            "cooldown_seconds": args.cooldown,
            "wired_limit_bytes": 2 * 1024**3,
            "segment_size": 3,
            "evaluate_segments": True,
            "projector": None,
            "thermal_before_cooldown": thermal_before_cooldown,
            "thermal_before": thermal_before,
            "thermal_after": thermal_after,
        },
        "production_baseline_proof": {
            "bit_identical": production_match,
            "difference": production_difference,
            "production_graph": "optimized positions + reassociated QKV + segment3 + stock pool",
            "experimental_graph": "same harness with stock finish",
        },
        "geometry_checks": geometry_checks,
        "timing": summaries,
        "paired": paired,
        "thermal_drift": {
            "baseline_first_half_p50_ms": first_half,
            "baseline_second_half_p50_ms": second_half,
            "absolute_drift_percent": drift_percent,
            "material_threshold_percent": args.drift_threshold_percent,
            "material": drift_percent > args.drift_threshold_percent,
        },
        "thermal_repeats": repeats,
        "segment_timing": segment_timings,
        "numerical_features": numerical,
        "real_performance_winners_safest_first": real_winners,
        "memory_final": memory_snapshot(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
