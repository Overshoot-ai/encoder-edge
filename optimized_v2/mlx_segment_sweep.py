import argparse
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
    fuse_gemma4_rope_layout,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-4-e4b-it-4bit",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--segment-sizes",
        type=int,
        nargs="+",
        default=[3, 4, 5, 8, 16],
    )
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--cooldown", type=float, default=0.0)
    parser.add_argument("--wired-limit", type=int, default=2 * 1024**3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    segment_sizes = list(dict.fromkeys(args.segment_sizes))
    if 3 not in segment_sizes:
        raise ValueError("Segment size 3 is required as the production reference")
    if any(size < 1 for size in segment_sizes):
        raise ValueError("Segment sizes must be positive")

    mx.set_wired_limit(args.wired_limit)
    model, _ = load(args.model)
    tower = model.vision_tower
    projector = model.embed_vision
    optimize_gemma4_positions(tower)
    fuse_gemma4_rope_layout(tower)
    pixels = mx.load(str(args.input))["pixels"]
    del model
    gc.collect()
    mx.clear_cache()

    functions = {
        f"segment{size}_eval": make_segmented_gemma4_encoder(
            tower,
            projector,
            segment_size=size,
            evaluate_segments=True,
        )
        for size in segment_sizes
    }
    outputs = {}
    for name, function in functions.items():
        for _ in range(args.warmups):
            outputs[name] = function(pixels)
            mx.eval(outputs[name])
        mx.synchronize()
    if args.cooldown:
        time.sleep(args.cooldown)

    timings = {name: [] for name in functions}
    names = list(functions)
    for round_index in range(args.rounds):
        offset = round_index % len(names)
        order = names[offset:] + names[:offset]
        if round_index % 2:
            order.reverse()
        for name in order:
            started = time.perf_counter()
            outputs[name] = functions[name](pixels)
            mx.eval(outputs[name])
            mx.synchronize()
            timings[name].append((time.perf_counter() - started) * 1000)

    reference = np.array(outputs["segment3_eval"].view(mx.uint16), copy=True)
    result = {
        "metadata": {
            "model": args.model,
            "input": str(args.input),
            "pixel_shape": list(pixels.shape),
            "segment_sizes": segment_sizes,
            "rounds": args.rounds,
            "warmups": args.warmups,
            "cooldown_seconds": args.cooldown,
            "wired_limit": args.wired_limit,
            "device": str(mx.default_device()),
            "device_info": mx.device_info(),
            "method": "Rotating and reversing interleaved order",
        },
        "results": {name: summarize(values) for name, values in timings.items()},
        "differences_from_segment3": {
            name: {
                "bit_identical": bool(np.array_equal(
                    reference,
                    np.array(output.view(mx.uint16), copy=True),
                )),
                "differing_values": int(np.count_nonzero(
                    reference != np.array(output.view(mx.uint16), copy=True)
                )),
            }
            for name, output in outputs.items()
            if name != "segment3_eval"
        },
        "peak_memory_bytes": mx.get_peak_memory(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
