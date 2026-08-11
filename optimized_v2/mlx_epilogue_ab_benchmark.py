import argparse
import copy
import gc
import json
import math
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm import load

from .mlx_vision_optimizations import (
    encode_gemma4_exact_pool_batch1,
    encode_gemma4_unpadded_batch1,
    fuse_gemma4_rope_layout,
    fuse_gemma4_rope_and_output_layout,
    make_segmented_gemma4_encoder,
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
        "minimum_ms": min(values),
        "maximum_ms": max(values),
        "raw_ms": values,
    }


def difference(reference: mx.array, candidate: mx.array) -> dict:
    mx.eval(reference, candidate)
    reference_bits = np.array(reference.view(mx.uint16), copy=True)
    candidate_bits = np.array(candidate.view(mx.uint16), copy=True)
    return {
        "bit_identical": bool(np.array_equal(reference_bits, candidate_bits)),
        "differing_values": int(np.count_nonzero(reference_bits != candidate_bits)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-4-e4b-it-4bit",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--cooldown", type=float, default=0.0)
    parser.add_argument("--wired-limit", type=int, default=2 * 1024**3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segmented-only", action="store_true")
    parser.add_argument("--output-layout", action="store_true")
    args = parser.parse_args()

    mx.set_wired_limit(args.wired_limit)
    model, _ = load(args.model)
    baseline_tower = model.vision_tower
    projector = model.embed_vision
    optimize_gemma4_positions(baseline_tower)
    candidate_tower = copy.deepcopy(baseline_tower)
    if args.output_layout:
        candidate_name = "rope_output_layout"
        fuse_gemma4_rope_and_output_layout(candidate_tower)
    else:
        candidate_name = "rope_layout"
        fuse_gemma4_rope_layout(candidate_tower)
    pixels = mx.load(str(args.input))["pixels"]
    del model
    gc.collect()
    mx.clear_cache()

    functions = {
        "baseline": mx.compile(
            lambda value: encode_gemma4_unpadded_batch1(
                baseline_tower,
                projector,
                value,
            )
        ),
        candidate_name: mx.compile(
            lambda value: encode_gemma4_unpadded_batch1(
                candidate_tower,
                projector,
                value,
            )
        ),
        "exact_pool": mx.compile(
            lambda value: encode_gemma4_exact_pool_batch1(
                baseline_tower,
                projector,
                value,
            )
        ),
        f"{candidate_name}_exact_pool": mx.compile(
            lambda value: encode_gemma4_exact_pool_batch1(
                candidate_tower,
                projector,
                value,
            )
        ),
    }
    if args.segmented_only:
        functions = {
            "baseline": functions["baseline"],
            candidate_name: functions[candidate_name],
            "baseline_segment3_eval": make_segmented_gemma4_encoder(
                baseline_tower,
                projector,
                3,
                evaluate_segments=True,
            ),
            f"{candidate_name}_segment3_eval": make_segmented_gemma4_encoder(
                candidate_tower,
                projector,
                3,
                evaluate_segments=True,
            ),
        }
    outputs = {}
    for name, function in functions.items():
        for _ in range(args.warmups):
            outputs[name] = function(pixels)
            mx.eval(outputs[name])
    mx.synchronize()
    if args.cooldown:
        time.sleep(args.cooldown)

    values = {name: [] for name in functions}
    names = list(functions)
    for round_index in range(args.rounds):
        order = names if round_index % 2 == 0 else list(reversed(names))
        for name in order:
            started = time.perf_counter()
            outputs[name] = functions[name](pixels)
            mx.eval(outputs[name])
            mx.synchronize()
            values[name].append((time.perf_counter() - started) * 1000)

    result = {
        "metadata": {
            "model": args.model,
            "input": str(args.input),
            "pixel_shape": list(pixels.shape),
            "rounds": args.rounds,
            "warmups": args.warmups,
            "cooldown_seconds": args.cooldown,
            "wired_limit": args.wired_limit,
            "device": str(mx.default_device()),
            "device_info": mx.device_info(),
        },
        "results": {name: summarize(samples) for name, samples in values.items()},
        "differences": {
            name: difference(outputs["baseline"], value)
            for name, value in outputs.items()
            if name != "baseline"
        },
        "peak_memory_bytes": mx.get_peak_memory(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
