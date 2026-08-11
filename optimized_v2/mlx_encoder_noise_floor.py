import argparse
import gc
import json
import math
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from mlx_vlm import load

from .mlx_vision_optimizations import (
    encode_gemma4_unpadded_batch1,
    optimize_gemma4_positions,
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize(values: list[float]) -> dict:
    return {
        "mean_ms": statistics.mean(values),
        "p50_ms": percentile(values, 0.5),
        "p90_ms": percentile(values, 0.9),
        "p99_ms": percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
        "raw_ms": values,
    }


def thermal_state() -> str:
    result = subprocess.run(
        ["pmset", "-g", "therm"],
        check=False,
        capture_output=True,
        text=True,
    )
    return (result.stdout + result.stderr).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--cooldown", type=int, default=90)
    parser.add_argument("--wired-limit", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 30:
        raise ValueError("Noise floor requires at least 30 rounds")

    if args.wired_limit:
        mx.set_wired_limit(args.wired_limit)
    model, _ = load("mlx-community/gemma-4-e4b-it-4bit")
    tower = model.vision_tower
    projector = model.embed_vision
    optimize_gemma4_positions(tower)
    del model
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    pixels = mx.load(str(args.input))["pixels"]
    encode = mx.compile(
        lambda value: encode_gemma4_unpadded_batch1(tower, projector, value)
    )
    for _ in range(5):
        mx.eval(encode(pixels))
    mx.synchronize()

    functions = {"identical_a": encode, "identical_b": encode}
    timings = {name: [] for name in functions}
    names = list(functions)
    thermal_before_cooldown = thermal_state()
    time.sleep(args.cooldown)
    thermal_before_timing = thermal_state()
    mx.reset_peak_memory()
    for round_index in range(args.rounds):
        offset = round_index % len(names)
        order = names[offset:] + names[:offset]
        if (round_index // len(names)) % 2:
            order.reverse()
        for name in order:
            started = time.perf_counter()
            output = functions[name](pixels)
            mx.eval(output)
            mx.synchronize()
            timings[name].append((time.perf_counter() - started) * 1000)
    thermal_after_timing = thermal_state()
    summaries = {name: summarize(values) for name, values in timings.items()}
    delta_ms = abs(
        summaries["identical_a"]["p50_ms"]
        - summaries["identical_b"]["p50_ms"]
    )
    center_ms = statistics.mean(
        [
            summaries["identical_a"]["p50_ms"],
            summaries["identical_b"]["p50_ms"],
        ]
    )
    result = {
        "metadata": {
            "model": "mlx-community/gemma-4-e4b-it-4bit",
            "device": str(mx.default_device()),
            "device_info": mx.device_info(),
            "pixel_shape": list(pixels.shape),
            "rounds": args.rounds,
            "cooldown_seconds": args.cooldown,
            "wired_limit_bytes": args.wired_limit,
            "method": "Same compiled callable under two names; rotating and block-reversing interleaved order",
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "thermal_before_cooldown": thermal_before_cooldown,
            "thermal_before_timing": thermal_before_timing,
            "thermal_after_timing": thermal_after_timing,
        },
        "results": summaries,
        "noise_floor": {
            "absolute_p50_delta_ms": delta_ms,
            "relative_p50_delta_percent": delta_ms / center_ms * 100,
            "promotion_threshold_ms": 2 * delta_ms,
        },
        "peak_memory_bytes": mx.get_peak_memory(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
