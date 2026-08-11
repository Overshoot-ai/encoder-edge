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

from .mlx_vision_optimizations import optimize_gemma4_positions


def measure(function, value, rounds: int) -> tuple[dict[str, float], object]:
    output = function(value)
    mx.eval(output)
    mx.synchronize()
    values = []
    for _ in range(rounds):
        started = time.perf_counter()
        output = function(value)
        mx.eval(output)
        mx.synchronize()
        values.append((time.perf_counter() - started) * 1000)
    ordered = sorted(values)
    return (
        {
            "mean": statistics.mean(values),
            "p50": ordered[math.ceil(len(ordered) * 0.5) - 1],
            "p90": ordered[math.ceil(len(ordered) * 0.9) - 1],
            "min": min(values),
            "max": max(values),
        },
        output,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    image = ImageOps.fit(
        Image.open(args.image).convert("RGB"),
        (854, 480),
        method=Image.Resampling.LANCZOS,
    )
    model, processor = load("mlx-community/gemma-4-e4b-it-4bit")
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
    optimize_gemma4_positions(tower)

    batch, _, height, width = pixels.shape
    num_patches = (height // tower.patch_size) * (width // tower.patch_size)
    output_length = num_patches // (tower.pooling_kernel_size**2)
    positions_np, padding_np, _ = tower._patch_positions_single(
        height,
        width,
        max_patches=num_patches,
    )
    positions = mx.array(np.tile(positions_np[None], (batch, 1, 1)))
    padding = mx.array(np.tile(padding_np[None], (batch, 1)))
    valid = ~padding
    attention_mask = mx.expand_dims(valid, 1) * mx.expand_dims(valid, 2)
    attention_mask = mx.where(attention_mask, mx.array(0.0, dtype=mx.bfloat16), -1e4)
    attention_mask = mx.expand_dims(attention_mask, 1)

    patch = mx.compile(lambda value: tower.patch_embedder(value, positions, padding))
    patch_metrics, hidden = measure(patch, pixels, args.rounds)
    layer_metrics = []
    for index, layer in enumerate(tower.encoder.layers):
        normalized = layer.input_layernorm(hidden)
        mx.eval(normalized)
        compiled_attention = mx.compile(
            lambda value, current=layer: current.self_attn(
                value,
                positions,
                attention_mask,
            )
        )
        attention_metrics, attention_output = measure(
            compiled_attention,
            normalized,
            args.rounds,
        )
        residual = hidden + layer.post_attention_layernorm(attention_output)
        mlp_input = layer.pre_feedforward_layernorm(residual)
        mx.eval(mlp_input)
        compiled_mlp = mx.compile(lambda value, current=layer: current.mlp(value))
        mlp_metrics, _ = measure(compiled_mlp, mlp_input, args.rounds)
        compiled_layer = mx.compile(
            lambda value, current=layer: current(value, positions, attention_mask)
        )
        metrics, hidden = measure(compiled_layer, hidden, args.rounds)
        metrics["layer"] = index
        metrics["attention_ms"] = attention_metrics
        metrics["mlp_ms"] = mlp_metrics
        layer_metrics.append(metrics)

    pool = mx.compile(
        lambda value: tower.pooler(
            value,
            positions,
            padding,
            output_length=output_length,
        )[0]
    )
    pool_metrics, pooled = measure(pool, hidden, args.rounds)
    compiled_projector = mx.compile(projector)
    projector_metrics, _ = measure(compiled_projector, pooled, args.rounds)
    whole = mx.compile(lambda value: projector(tower(value, None)))
    whole_metrics, _ = measure(whole, pixels, args.rounds)

    result = {
        "device": str(mx.default_device()),
        "rounds": args.rounds,
        "pixel_shape": list(pixels.shape),
        "patch_embedding_ms": patch_metrics,
        "layers_ms": layer_metrics,
        "layer_p50_sum_ms": sum(item["p50"] for item in layer_metrics),
        "layer_p50_mean_ms": statistics.mean(
            item["p50"] for item in layer_metrics
        ),
        "attention_p50_sum_ms": sum(
            item["attention_ms"]["p50"] for item in layer_metrics
        ),
        "mlp_p50_sum_ms": sum(item["mlp_ms"]["p50"] for item in layer_metrics),
        "pool_ms": pool_metrics,
        "projector_ms": projector_metrics,
        "whole_compiled_ms": whole_metrics,
        "note": "Isolated stage times synchronize each stage and do not add to whole-model latency.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
