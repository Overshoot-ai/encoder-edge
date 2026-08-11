import argparse
import gc
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

from .mlx_vision_optimizations import optimize_gemma4_positions


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def benchmark(function, pixels, rounds: int) -> tuple[dict[str, float], object]:
    for _ in range(3):
        output = function(pixels)
        mx.eval(output)
    mx.synchronize()
    mx.reset_peak_memory()
    values = []
    for _ in range(rounds):
        started = time.perf_counter()
        output = function(pixels)
        mx.eval(output)
        mx.synchronize()
        values.append((time.perf_counter() - started) * 1000)
    return (
        {
            "mean_ms": statistics.mean(values),
            "p50_ms": percentile(values, 0.5),
            "p90_ms": percentile(values, 0.9),
            "min_ms": min(values),
            "max_ms": max(values),
            "peak_memory_gb": mx.get_peak_memory() / 1e9,
            "raw_ms": values,
        },
        output,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-4-e4b-it-4bit",
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--budgets", default="264,192,128,64")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--batch-rounds", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    budgets = [int(value) for value in args.budgets.split(",")]

    image = ImageOps.fit(
        Image.open(args.image).convert("RGB"),
        (854, 480),
        method=Image.Resampling.LANCZOS,
    )
    model, processor = load(args.model)
    prompt = apply_chat_template(
        processor,
        model.config,
        "Describe this image.",
        num_images=1,
    )
    tower = model.vision_tower
    projector = model.embed_vision
    optimize_gemma4_positions(tower)
    del model
    gc.collect()
    mx.clear_cache()

    results = []
    for budget in budgets:
        processor.image_processor.max_soft_tokens = budget
        inputs = prepare_inputs(
            processor,
            images=[image],
            prompts=prompt,
            add_special_tokens=False,
        )
        pixels = inputs["pixel_values"]
        patch_tokens = (pixels.shape[-2] // tower.patch_size) * (
            pixels.shape[-1] // tower.patch_size
        )
        visual_tokens = patch_tokens // (tower.pooling_kernel_size**2)
        budget_result = {
            "requested_soft_token_budget": budget,
            "actual_visual_tokens_per_image": visual_tokens,
            "patch_tokens_per_image": patch_tokens,
            "pixel_shape_batch_1": list(pixels.shape),
            "payload_bytes_per_image": visual_tokens * 2560 * 2,
            "batches": {},
        }

        for batch_size, rounds in ((1, args.rounds), (args.batch_size, args.batch_rounds)):
            batched_pixels = (
                pixels if batch_size == 1 else mx.repeat(pixels, batch_size, axis=0)
            )
            encode = mx.compile(lambda value: projector(tower(value, None)))
            try:
                metrics, output = benchmark(encode, batched_pixels, rounds)
                expected_tokens = batch_size * visual_tokens
                if output.shape != (1, expected_tokens, 2560):
                    raise RuntimeError(
                        f"Unexpected output shape {output.shape}; expected "
                        f"(1, {expected_tokens}, 2560)"
                    )
                output_bits = np.array(output.view(mx.uint16), copy=True).reshape(
                    batch_size,
                    visual_tokens,
                    2560,
                )
                repeated_outputs_equal = bool(
                    all(
                        np.array_equal(output_bits[0], output_bits[index])
                        for index in range(1, batch_size)
                    )
                )
                metrics.update(
                    {
                        "batch_size": batch_size,
                        "rounds": rounds,
                        "pixel_shape": list(batched_pixels.shape),
                        "output_shape": list(output.shape),
                        "amortized_p50_ms_per_image": metrics["p50_ms"]
                        / batch_size,
                        "images_per_second_at_p50": batch_size
                        / (metrics["p50_ms"] / 1000),
                        "payload_bytes_for_batch": batch_size
                        * visual_tokens
                        * 2560
                        * 2,
                        "repeated_outputs_equal": repeated_outputs_equal,
                    }
                )
                budget_result["batches"][str(batch_size)] = metrics
            except (MemoryError, RuntimeError) as error:
                budget_result["batches"][str(batch_size)] = {
                    "batch_size": batch_size,
                    "rounds": rounds,
                    "error": str(error),
                }
            finally:
                del encode, batched_pixels
                gc.collect()
                mx.clear_cache()
        results.append(budget_result)
        print(json.dumps(budget_result, indent=2), flush=True)

    report = {
        "metadata": {
            "model": args.model,
            "device": str(mx.default_device()),
            "device_info": mx.device_info(),
            "budgets": budgets,
            "batch_size": args.batch_size,
            "rounds": args.rounds,
            "batch_rounds": args.batch_rounds,
            "optimization": "gathered positional embeddings",
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
