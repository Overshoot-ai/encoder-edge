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

from .mlx_vision_optimizations import (
    encode_gemma4_unpadded_batch1,
    fuse_gemma4_qkv,
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
        "maximum_absolute_difference": float(absolute.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-4-e4b-it-4bit",
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--final-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    image = ImageOps.fit(
        Image.open(args.image).convert("RGB"),
        (854, 480),
        method=Image.Resampling.LANCZOS,
    )
    model, processor = load(args.model)
    processor.image_processor.max_soft_tokens = 273
    prompt = apply_chat_template(
        processor,
        model.config,
        "Describe this image.",
        num_images=1,
    )
    pixels = prepare_inputs(
        processor,
        images=[image],
        prompts=prompt,
        add_special_tokens=False,
    )["pixel_values"]
    tower = model.vision_tower
    projector = model.embed_vision
    del model
    gc.collect()
    mx.clear_cache()

    batch, _, height, width = pixels.shape
    patch_tokens = (height // tower.patch_size) * (width // tower.patch_size)
    output_tokens = patch_tokens // (tower.pooling_kernel_size**2)
    positions_np, padding_np, _ = tower._patch_positions_single(
        height,
        width,
        max_patches=patch_tokens,
    )
    positions = mx.array(np.tile(positions_np[None], (batch, 1, 1)))
    padding = mx.array(np.tile(padding_np[None], (batch, 1)))
    mx.eval(positions, padding)
    if bool(np.array(padding, copy=True).any()):
        raise ValueError("Unmasked benchmark requires an input with no padded patches")

    def compile_and_warm(function):
        compiled = mx.compile(function)
        for _ in range(5):
            output = compiled(pixels)
            mx.eval(output)
        mx.synchronize()
        return compiled, output

    def unmasked_encode(value):
        return encode_gemma4_unpadded_batch1(tower, projector, value)

    baseline, baseline_output = compile_and_warm(
        lambda value: projector(tower(value, None))
    )

    optimize_gemma4_positions(tower)
    gathered_masked, gathered_masked_output = compile_and_warm(
        lambda value: projector(tower(value, None))
    )
    gathered_unmasked, gathered_unmasked_output = compile_and_warm(unmasked_encode)

    fuse_gemma4_qkv(tower)
    if not args.final_only:
        fused_masked, fused_masked_output = compile_and_warm(
            lambda value: projector(tower(value, None))
        )
    fused_unmasked, fused_unmasked_output = compile_and_warm(unmasked_encode)

    functions = {
        "baseline": baseline,
        "gathered_masked": gathered_masked,
        "gathered_unmasked": gathered_unmasked,
        "fused_qkv_unmasked": fused_unmasked,
    }
    outputs = {
        "baseline": baseline_output,
        "gathered_masked": gathered_masked_output,
        "gathered_unmasked": gathered_unmasked_output,
        "fused_qkv_unmasked": fused_unmasked_output,
    }
    if not args.final_only:
        functions["fused_qkv_masked"] = fused_masked
        outputs["fused_qkv_masked"] = fused_masked_output
    else:
        functions = {
            name: functions[name]
            for name in (
                "gathered_masked",
                "gathered_unmasked",
                "fused_qkv_unmasked",
            )
        }
        del baseline
    names = list(functions)
    timings = {name: [] for name in names}
    for round_index in range(args.rounds):
        offset = round_index % len(names)
        order = names[offset:] + names[:offset]
        if round_index % 2:
            order.reverse()
        for name in order:
            started = time.perf_counter()
            output = functions[name](pixels)
            mx.eval(output)
            mx.synchronize()
            timings[name].append((time.perf_counter() - started) * 1000)

    summaries = {name: summarize(values) for name, values in timings.items()}
    gathered_p50 = summaries["gathered_masked"]["p50_ms"]
    for name, metrics in summaries.items():
        if "baseline" in summaries:
            baseline_p50 = summaries["baseline"]["p50_ms"]
            metrics["speedup_vs_baseline_percent"] = (
                (baseline_p50 - metrics["p50_ms"]) / baseline_p50 * 100
            )
        metrics["speedup_vs_gathered_masked_percent"] = (
            (gathered_p50 - metrics["p50_ms"]) / gathered_p50 * 100
        )

    result = {
        "metadata": {
            "model": args.model,
            "device": str(mx.default_device()),
            "device_info": mx.device_info(),
            "rounds": args.rounds,
            "method": "Rotating and reversing interleaved order",
            "pixel_shape": list(pixels.shape),
            "patch_tokens": patch_tokens,
            "visual_tokens": output_tokens,
        },
        "results": summaries,
        "comparisons_to_baseline": {
            name: difference(baseline_output, output)
            for name, output in outputs.items()
            if name != "baseline"
        },
        "comparisons_to_gathered_masked": {
            name: difference(gathered_masked_output, output)
            for name, output in outputs.items()
            if name not in ("baseline", "gathered_masked")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
