"""Interleaved BF16 versus FP16 Gemma 4 vision-tower benchmark."""

import argparse
import importlib.metadata
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs
from PIL import Image, ImageOps

from .mlx_vision_quantization_ab import make_encoder, summarize


def error_metrics(reference: mx.array, candidate: mx.array) -> dict:
    mx.eval(reference, candidate)
    reference_f32 = np.asarray(reference.astype(mx.float32))
    candidate_f32 = np.asarray(candidate.astype(mx.float32))
    difference = candidate_f32 - reference_f32
    return {
        "finite": bool(np.isfinite(candidate_f32).all()),
        "bit_equal_percent": float(
            np.mean(np.asarray(reference.view(mx.uint16)) == np.asarray(candidate.view(mx.uint16)))
            * 100
        ),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "relative_l2_error": float(
            np.linalg.norm(difference.ravel()) / np.linalg.norm(reference_f32.ravel())
        ),
        "cosine_similarity": float(
            np.dot(reference_f32.ravel(), candidate_f32.ravel())
            / (np.linalg.norm(reference_f32) * np.linalg.norm(candidate_f32))
        ),
    }


def timed(function, pixels):
    started = time.perf_counter()
    output = function(pixels)
    mx.eval(output)
    mx.synchronize()
    return (time.perf_counter() - started) * 1000, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default="mlx-community/gemma-4-e4b-it-4bit"
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mx.set_wired_limit(2 * 1024**3)
    placeholder = {
        "mode": "affine",
        "bits": 4,
        "group_size": 64,
        "quantize_input": False,
    }
    _, bf16_encoder, processor, config, bf16_bytes = make_encoder(
        args.checkpoint, False, placeholder
    )
    fp16_tower, fp16_encoder, _, _, _ = make_encoder(
        args.checkpoint, False, placeholder
    )
    fp16_tower.set_dtype(mx.float16)
    fp16_parameters = [value for _, value in tree_flatten(fp16_tower.parameters())]
    mx.eval(*fp16_parameters)
    mx.synchronize()
    fp16_bytes = sum(value.nbytes for value in fp16_parameters)

    image = ImageOps.fit(
        Image.open(args.image).convert("RGB"),
        (854, 480),
        method=Image.Resampling.LANCZOS,
    )
    prompt = apply_chat_template(
        processor, config, "Describe this image.", num_images=1
    )
    pixels = prepare_inputs(
        processor,
        images=[image],
        prompts=prompt,
        add_special_tokens=False,
    )["pixel_values"]

    functions = {
        "bf16": lambda value: bf16_encoder(value),
        "fp16_to_bf16": lambda value: fp16_encoder(
            value.astype(mx.float16)
        ).astype(mx.bfloat16),
    }
    for function in functions.values():
        for _ in range(args.warmups):
            mx.eval(function(pixels))
    mx.synchronize()

    timings = {name: [] for name in functions}
    outputs = {}
    names = list(functions)
    mx.reset_peak_memory()
    for round_index in range(args.rounds):
        order = names if round_index % 2 == 0 else list(reversed(names))
        for name in order:
            elapsed, outputs[name] = timed(functions[name], pixels)
            timings[name].append(elapsed)

    metrics = {name: {**summarize(values), "raw": values} for name, values in timings.items()}
    if not math.isfinite(metrics["fp16_to_bf16"]["p50"]):
        raise RuntimeError("FP16 benchmark produced non-finite latency")
    result = {
        "metadata": {
            "checkpoint": args.checkpoint,
            "mlx_version": importlib.metadata.version("mlx"),
            "mlx_vlm_version": importlib.metadata.version("mlx-vlm"),
            "device_info": mx.device_info(),
            "pixel_shape": list(pixels.shape),
            "output_shape": list(outputs["bf16"].shape),
            "warmups": args.warmups,
            "rounds": args.rounds,
            "projector_included": False,
            "fp16_output_cast_to_bf16": True,
            "parameter_bytes": {"bf16": bf16_bytes, "fp16": fp16_bytes},
        },
        "timings": metrics,
        "fp16_p50_change_percent": (
            metrics["fp16_to_bf16"]["p50"] / metrics["bf16"]["p50"] - 1
        )
        * 100,
        "error": error_metrics(outputs["bf16"], outputs["fp16_to_bf16"]),
        "peak_memory_bytes": mx.get_peak_memory(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
