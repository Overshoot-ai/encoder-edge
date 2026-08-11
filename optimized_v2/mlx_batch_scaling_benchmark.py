import argparse
import copy
import gc
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm import load

from .mlx_vision_optimizations import (
    fuse_gemma4_rope_layout,
    make_segmented_gemma4_encoder,
    optimize_gemma4_positions,
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize(values: list[float], batch_size: int, peak: int) -> dict:
    p50 = percentile(values, 0.5)
    return {
        "mean_ms": statistics.mean(values),
        "p50_ms": p50,
        "p90_ms": percentile(values, 0.9),
        "minimum_ms": min(values),
        "maximum_ms": max(values),
        "amortized_p50_ms_per_image": p50 / batch_size,
        "images_per_second_at_p50": batch_size / (p50 / 1000),
        "peak_memory_bytes": peak,
        "raw_ms": values,
    }


def bits_by_image(output: mx.array, batch_size: int) -> np.ndarray:
    if output.shape[-1] != 2560:
        raise RuntimeError(f"Unexpected hidden size in output {output.shape}")
    values = np.array(output.view(mx.uint16), copy=True)
    if values.size % (batch_size * 2560):
        raise RuntimeError(f"Cannot split output {output.shape} into {batch_size} images")
    return values.reshape(batch_size, -1, 2560)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-4-e4b-it-4bit",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--wired-limit", type=int, default=2 * 1024**3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    if not batch_sizes or any(value < 1 for value in batch_sizes):
        raise ValueError("Batch sizes must be positive")
    if args.rounds < 1 or args.warmups < 1:
        raise ValueError("Rounds and warmups must be positive")

    mx.set_wired_limit(args.wired_limit)
    model, _ = load(args.model)
    baseline_tower = model.vision_tower
    projector = model.embed_vision
    optimized_tower = copy.deepcopy(baseline_tower)
    optimize_gemma4_positions(optimized_tower)
    fuse_gemma4_rope_layout(optimized_tower)
    optimized_encode = make_segmented_gemma4_encoder(
        optimized_tower,
        projector,
        segment_size=3,
        evaluate_segments=True,
    )
    pixels = mx.load(str(args.input))["pixels"]
    if pixels.shape[0] != 1:
        raise ValueError("Input tensor must contain exactly one image")
    del model
    gc.collect()
    mx.clear_cache()

    results = {}
    for batch_size in batch_sizes:
        batched_pixels = (
            pixels if batch_size == 1 else mx.repeat(pixels, batch_size, axis=0)
        )
        baseline_encode = mx.compile(
            lambda value: projector(baseline_tower(value, None))
        )
        functions = {
            "stock_baseline": baseline_encode,
            "optimized_segment3": optimized_encode,
        }
        outputs = {}
        peaks = {}
        for name, function in functions.items():
            for _ in range(args.warmups):
                outputs[name] = function(batched_pixels)
                mx.eval(outputs[name])
            mx.synchronize()
            mx.reset_peak_memory()
            outputs[name] = function(batched_pixels)
            mx.eval(outputs[name])
            mx.synchronize()
            peaks[name] = mx.get_peak_memory()

        timings = {name: [] for name in functions}
        names = list(functions)
        for round_index in range(args.rounds):
            order = names if round_index % 2 == 0 else list(reversed(names))
            for name in order:
                started = time.perf_counter()
                outputs[name] = functions[name](batched_pixels)
                mx.eval(outputs[name])
                mx.synchronize()
                timings[name].append((time.perf_counter() - started) * 1000)

        baseline_bits = bits_by_image(outputs["stock_baseline"], batch_size)
        optimized_bits = bits_by_image(outputs["optimized_segment3"], batch_size)
        results[str(batch_size)] = {
            "batch_size": batch_size,
            "pixel_shape": list(batched_pixels.shape),
            "baseline_output_shape": list(outputs["stock_baseline"].shape),
            "optimized_output_shape": list(outputs["optimized_segment3"].shape),
            "visual_tokens_per_image": baseline_bits.shape[1],
            "optimized_bit_identical": bool(
                np.array_equal(baseline_bits, optimized_bits)
            ),
            "optimized_differing_values": int(
                np.count_nonzero(baseline_bits != optimized_bits)
            ),
            "baseline_repeated_outputs_equal": bool(
                all(
                    np.array_equal(baseline_bits[0], baseline_bits[index])
                    for index in range(1, batch_size)
                )
            ),
            "optimized_repeated_outputs_equal": bool(
                all(
                    np.array_equal(optimized_bits[0], optimized_bits[index])
                    for index in range(1, batch_size)
                )
            ),
            "arms": {
                name: summarize(values, batch_size, peaks[name])
                for name, values in timings.items()
            },
        }
        print(json.dumps(results[str(batch_size)], indent=2), flush=True)
        del baseline_encode, batched_pixels
        gc.collect()
        mx.clear_cache()

    report = {
        "metadata": {
            "model": args.model,
            "input": str(args.input),
            "batch_sizes": batch_sizes,
            "rounds": args.rounds,
            "warmups": args.warmups,
            "wired_limit": args.wired_limit,
            "device": str(mx.default_device()),
            "device_info": mx.device_info(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "method": "Alternating interleaved stock and optimized arms",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
