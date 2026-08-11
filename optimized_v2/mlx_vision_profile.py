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

from .mlx_vision_optimizations import optimize_gemma4_vision


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def measure(function, argument, rounds: int) -> dict[str, float]:
    output = function(argument)
    mx.eval(output)
    mx.synchronize()
    values = []
    for _ in range(rounds):
        started = time.perf_counter()
        output = function(argument)
        mx.eval(output)
        mx.synchronize()
        values.append((time.perf_counter() - started) * 1000)
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
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--output", type=Path)
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
    vision_tower = model.vision_tower
    embed_vision = model.embed_vision
    del model
    gc.collect()
    mx.clear_cache()
    mx.synchronize()

    def tower(value):
        return vision_tower(value, None)

    hidden = tower(pixels)
    mx.eval(hidden)
    mx.synchronize()

    def projector(value):
        return embed_vision(value)

    def encode(value):
        return embed_vision(vision_tower(value, None))

    compiled_encode = mx.compile(encode)
    tower_metrics = measure(tower, pixels, args.rounds)
    projector_metrics = measure(projector, hidden, args.rounds)
    eager_metrics = measure(encode, pixels, args.rounds)
    compiled_metrics = measure(compiled_encode, pixels, args.rounds)
    baseline = encode(pixels)
    mx.eval(baseline)
    mx.synchronize()

    optimize_gemma4_vision(vision_tower)
    optimized = encode(pixels)
    mx.eval(optimized)
    mx.synchronize()
    baseline_bits = np.array(baseline.view(mx.uint16), copy=True)
    optimized_bits = np.array(optimized.view(mx.uint16), copy=True)
    optimized_encode = mx.compile(encode)
    result = {
        "device": str(mx.default_device()),
        "pixel_shape": list(pixels.shape),
        "pixel_dtype": str(pixels.dtype),
        "tower_output_shape": list(hidden.shape),
        "output_shape": list(embed_vision(hidden).shape),
        "rounds": args.rounds,
        "tower_ms": tower_metrics,
        "projector_ms": projector_metrics,
        "eager_encode_ms": eager_metrics,
        "compiled_encode_ms": compiled_metrics,
        "optimized_gathered_positions_fused_qkv": {
            "bit_identical": bool(np.array_equal(baseline_bits, optimized_bits)),
            "differing_values": int(np.count_nonzero(baseline_bits != optimized_bits)),
            "compiled_encode_ms": measure(optimized_encode, pixels, args.rounds),
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
