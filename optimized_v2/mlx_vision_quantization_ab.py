import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs
from PIL import Image, ImageOps

from .mlx_vision_optimizations import (
    fuse_gemma4_rope_layout,
    make_segmented_gemma4_encoder,
    optimize_gemma4_positions,
)


_qqlinear_call = nn.QQLinear.__call__


def flattened_qqlinear_call(self, inputs):
    if inputs.ndim <= 2:
        return _qqlinear_call(self, inputs)
    leading_shape = inputs.shape[:-1]
    output = _qqlinear_call(self, inputs.reshape(-1, inputs.shape[-1]))
    return output.reshape(*leading_shape, output.shape[-1])


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, round((len(ordered) - 1) * fraction))]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "p50": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "minimum": min(values),
        "maximum": max(values),
    }


def parameter_bytes(module) -> int:
    return sum(array.nbytes for _, array in tree_flatten(module.parameters()))


def make_encoder(checkpoint: str, quantized: bool, quantization: dict):
    model, processor = load(checkpoint)
    tower = model.vision_tower
    config = model.config
    if quantized:
        if quantization["quantize_input"]:
            nn.QQLinear.__call__ = flattened_qqlinear_call
        excluded = tuple(quantization.get("excluded", ()))
        included = tuple(quantization.get("included", ()))

        def quantization_predicate(path, module):
            if not isinstance(module, nn.Linear):
                return False
            if any(path.endswith(name) for name in excluded):
                return False
            if included and not any(name in path for name in included):
                return False
            if quantization.get("recipe") != "mixed_4_8":
                return True

            layer_index = next(
                (int(part) for part in path.split(".") if part.isdigit()),
                None,
            )
            if layer_index is None:
                return False
            layer_count = len(tower.encoder.layers)
            edge = layer_count // 8
            use_more_bits = (
                layer_index < edge
                or layer_index >= 7 * edge
                or (layer_index - edge) % 3 == 2
            )
            bits = (
                8
                if use_more_bits
                and ("v_proj" in path or "down_proj" in path)
                else 4
            )
            return {"group_size": 64, "bits": bits, "mode": "affine"}

        nn.quantize(
            tower,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization["mode"],
            quantize_input=quantization["quantize_input"],
            class_predicate=quantization_predicate,
        )
    optimize_gemma4_positions(tower)
    fuse_gemma4_rope_layout(tower)
    encoder = make_segmented_gemma4_encoder(
        tower,
        None,
        segment_size=3,
        evaluate_segments=True,
    )
    parameters = [array for _, array in tree_flatten(tower.parameters())]
    mx.eval(*parameters)
    mx.synchronize()
    size = parameter_bytes(tower)
    del model
    gc.collect()
    return tower, encoder, processor, config, size


def timed(encoder, pixels) -> tuple[float, mx.array]:
    started = time.perf_counter()
    output = encoder(pixels)
    mx.eval(output)
    mx.synchronize()
    return (time.perf_counter() - started) * 1000, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="mlx-community/gemma-4-e4b-it-4bit",
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument(
        "--q-mode",
        choices=(
            "affine",
            "nvfp4",
            "nvfp4-weight",
            "nvfp4-mlp",
            "mixed-4-8",
            "mxfp8",
        ),
        default="affine",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mx.set_wired_limit(2 * 1024**3)
    quantization = {
        "affine": {
            "mode": "affine",
            "bits": 4,
            "group_size": 64,
            "quantize_input": False,
        },
        "nvfp4": {
            "mode": "nvfp4",
            "bits": 4,
            "group_size": 16,
            "quantize_input": True,
            "excluded": ["patch_embedder.input_proj"],
        },
        "nvfp4-weight": {
            "mode": "nvfp4",
            "bits": 4,
            "group_size": 16,
            "quantize_input": False,
            "excluded": ["patch_embedder.input_proj"],
        },
        "nvfp4-mlp": {
            "mode": "nvfp4",
            "bits": 4,
            "group_size": 16,
            "quantize_input": False,
            "included": ["mlp"],
        },
        "mixed-4-8": {
            "mode": "affine",
            "bits": 4,
            "group_size": 64,
            "quantize_input": False,
            "excluded": ["patch_embedder.input_proj"],
            "recipe": "mixed_4_8",
        },
        "mxfp8": {
            "mode": "mxfp8",
            "bits": 8,
            "group_size": 32,
            "quantize_input": True,
            "excluded": ["patch_embedder.input_proj"],
        },
    }[args.q_mode]
    bf16_tower, bf16_encoder, processor, config, bf16_bytes = make_encoder(
        args.checkpoint, False, quantization
    )
    q4_tower, q4_encoder, _, _, q4_bytes = make_encoder(
        args.checkpoint, True, quantization
    )

    image = ImageOps.fit(
        Image.open(args.image).convert("RGB"),
        (854, 480),
        method=Image.Resampling.LANCZOS,
    )
    prompt = apply_chat_template(
        processor,
        config,
        "Describe this image.",
        num_images=1,
    )
    pixels = prepare_inputs(
        processor,
        images=[image],
        prompts=prompt,
        add_special_tokens=False,
    )["pixel_values"]

    results = {}
    for batch_size in (1, 8):
        batch_pixels = (
            pixels if batch_size == 1 else mx.repeat(pixels, batch_size, axis=0)
        )
        for encoder in (bf16_encoder, q4_encoder):
            timed(encoder, batch_pixels)
            timed(encoder, batch_pixels)

        timings = {"bf16": [], "q4": []}
        outputs = {}
        for index in range(args.rounds):
            order = (
                (("bf16", bf16_encoder), ("q4", q4_encoder))
                if index % 2 == 0
                else (("q4", q4_encoder), ("bf16", bf16_encoder))
            )
            for name, encoder in order:
                elapsed, output = timed(encoder, batch_pixels)
                timings[name].append(elapsed)
                outputs[name] = output

        difference = (
            outputs["bf16"].astype(mx.float32)
            - outputs["q4"].astype(mx.float32)
        )
        equal = outputs["bf16"].view(mx.uint16) == outputs["q4"].view(mx.uint16)
        results[str(batch_size)] = {
            "bf16_ms": summarize(timings["bf16"]),
            "q4_ms": summarize(timings["q4"]),
            "q4_p50_change_percent": (
                summarize(timings["q4"])["p50"]
                / summarize(timings["bf16"])["p50"]
                - 1
            )
            * 100,
            "output": {
                "shape": list(outputs["bf16"].shape),
                "bit_equal_percent": mx.mean(equal).item() * 100,
                "mean_absolute_error": mx.mean(mx.abs(difference)).item(),
                "max_absolute_error": mx.max(mx.abs(difference)).item(),
                "relative_l2_error": (
                    mx.sqrt(mx.sum(difference**2))
                    / mx.sqrt(mx.sum(outputs["bf16"].astype(mx.float32) ** 2))
                ).item(),
            },
        }

    result = {
        "metadata": {
            "checkpoint": args.checkpoint,
            "rounds": args.rounds,
            "image_resolution": "854x480",
            "interleaved": True,
            "projector_included": False,
            "quantization": quantization,
        },
        "parameters": {
            "bf16_bytes": bf16_bytes,
            "q4_bytes": q4_bytes,
            "reduction_percent": (1 - q4_bytes / bf16_bytes) * 100,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

    del bf16_tower, q4_tower


if __name__ == "__main__":
    main()
