import argparse
import gc
import json
import math
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

from .mlx_vision_optimizations import fuse_gemma4_qkv, optimize_gemma4_positions


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "p50": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-4-e4b-it-4bit",
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = Image.open(args.image).convert("RGB")
    image = ImageOps.fit(source, (854, 480), method=Image.Resampling.LANCZOS)
    model, processor = load(args.model)
    prompt = apply_chat_template(
        processor,
        model.config,
        "Describe this image.",
        num_images=1,
    )
    inputs = prepare_inputs(
        processor,
        images=[image],
        prompts=prompt,
        add_special_tokens=False,
    )
    pixels = inputs["pixel_values"]
    tower = model.vision_tower
    projector = model.embed_vision
    del model
    gc.collect()
    mx.clear_cache()

    baseline = mx.compile(lambda value: projector(tower(value, None)))
    baseline_output = baseline(pixels)
    mx.eval(baseline_output)
    mx.synchronize()

    optimize_gemma4_positions(tower)
    gathered = mx.compile(lambda value: projector(tower(value, None)))
    gathered_output = gathered(pixels)
    mx.eval(gathered_output)
    mx.synchronize()

    fuse_gemma4_qkv(tower)
    fused_qkv = mx.compile(lambda value: projector(tower(value, None)))
    fused_output = fused_qkv(pixels)
    mx.eval(fused_output)
    mx.synchronize()

    functions = {
        "baseline": baseline,
        "gathered_positions": gathered,
        "gathered_positions_fused_qkv": fused_qkv,
    }
    timings = {name: [] for name in functions}
    names = list(functions)
    for round_index in range(args.rounds):
        order = names if round_index % 2 == 0 else list(reversed(names))
        for name in order:
            started = time.perf_counter()
            output = functions[name](pixels)
            mx.eval(output)
            mx.synchronize()
            timings[name].append((time.perf_counter() - started) * 1000)

    baseline_bits = np.array(baseline_output.view(mx.uint16), copy=True)
    outputs = {
        "gathered_positions": gathered_output,
        "gathered_positions_fused_qkv": fused_output,
    }
    comparisons = {}
    for name, output in outputs.items():
        bits = np.array(output.view(mx.uint16), copy=True)
        comparisons[name] = {
            "bit_identical": bool(np.array_equal(baseline_bits, bits)),
            "differing_values": int(np.count_nonzero(baseline_bits != bits)),
        }

    summaries = {name: summary(values) for name, values in timings.items()}
    baseline_p50 = summaries["baseline"]["p50"]
    for name, metrics in summaries.items():
        metrics["p50_speedup_vs_baseline_percent"] = (
            (baseline_p50 - metrics["p50"]) / baseline_p50 * 100
        )
    result = {
        "model": args.model,
        "device": str(mx.default_device()),
        "rounds": args.rounds,
        "method": "Alternating baseline/gather/QKV order each round",
        "pixel_shape": list(pixels.shape),
        "results": summaries,
        "comparisons_to_baseline": comparisons,
        "raw_ms": timings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
