import argparse
import gc
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm import load
from mlx_vlm.models.base import ensure_fused_sdpa
from mlx_vlm.models.gemma4.vision import apply_multidimensional_rope

from .mlx_kernel_candidate_benchmark import difference, measure_interleaved
from .mlx_vision_optimizations import optimize_gemma4_positions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-4-e4b-it-4bit",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mx.set_wired_limit(2 * 1024**3)
    model, _ = load(args.model)
    tower = model.vision_tower
    optimize_gemma4_positions(tower)
    pixels = mx.load(str(args.input))["pixels"]
    layer = tower.encoder.layers[0]
    del model
    gc.collect()
    mx.clear_cache()

    _, _, height, width = pixels.shape
    length = (height // tower.patch_size) * (width // tower.patch_size)
    positions_np, padding_np, _ = tower._patch_positions_single(
        height, width, max_patches=length
    )
    positions = mx.array(positions_np[None])
    padding = mx.array(padding_np[None])
    hidden = tower.patch_embedder(pixels, positions, padding)
    normalized = layer.input_layernorm(hidden)
    attention = layer.self_attn

    q = attention.q_proj(normalized).reshape(1, length, 12, 64)
    k = attention.k_proj(normalized).reshape(1, length, 12, 64)
    v = attention.v_proj(normalized).reshape(1, length, 12, 64)
    q = apply_multidimensional_rope(
        attention.q_norm(q), positions, attention.rope_base_frequency
    ).transpose(0, 2, 1, 3)
    k = apply_multidimensional_rope(
        attention.k_norm(k), positions, attention.rope_base_frequency
    ).transpose(0, 2, 1, 3)
    v = attention._v_norm(v).transpose(0, 2, 1, 3)
    attention_input = ensure_fused_sdpa(q, k, v, scale=1.0, mask=None)
    attention_input = attention_input.transpose(0, 2, 1, 3).reshape(1, length, -1)
    attention_output = attention.o_proj(attention_input)
    residual = hidden + layer.post_attention_layernorm(attention_output)
    mlp_input = layer.pre_feedforward_layernorm(residual)
    gate = layer.mlp.gate_proj(mlp_input)
    up = layer.mlp.up_proj(mlp_input)
    down_input = nn.gelu_approx(gate) * up
    mx.eval(normalized, attention_input, mlp_input, down_input)

    operations = {
        "q": (attention.q_proj, normalized),
        "k": (attention.k_proj, normalized),
        "v": (attention.v_proj, normalized),
        "o": (attention.o_proj, attention_input),
        "gate": (layer.mlp.gate_proj, mlp_input),
        "up": (layer.mlp.up_proj, mlp_input),
        "down": (layer.mlp.down_proj, down_input),
    }
    results = {}
    for name, (projection, value) in operations.items():
        packed = mx.contiguous(projection.linear.weight.T)
        mx.eval(packed)

        def stock(current=projection, current_value=value):
            return current(current_value)

        def pretransposed(
            current=projection,
            current_value=value,
            current_weight=packed,
        ):
            current_value = mx.clip(
                current_value,
                current.input_min,
                current.input_max,
            )
            output = current_value @ current_weight
            return mx.clip(output, current.output_min, current.output_max)

        metrics = measure_interleaved(
            {"stock": stock, "pretransposed": pretransposed},
            args.rounds,
        )
        stock_output = stock()
        candidate_output = pretransposed()
        metrics["difference"] = difference(stock_output, candidate_output)
        results[name] = metrics
        print(
            f"{name}: {metrics['stock']['p50_ms']:.3f} -> "
            f"{metrics['pretransposed']['p50_ms']:.3f} ms",
            flush=True,
        )

    output = {
        "metadata": {
            "model": args.model,
            "pixel_shape": list(pixels.shape),
            "sequence_length": length,
            "rounds": args.rounds,
            "device_info": mx.device_info(),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
