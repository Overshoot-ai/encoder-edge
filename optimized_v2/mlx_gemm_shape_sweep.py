import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm import load
from mlx_vlm.models.base import ensure_fused_sdpa
from mlx_vlm.models.gemma4.vision import apply_multidimensional_rope

from .mlx_kernel_candidate_benchmark import difference, measure_interleaved
from .mlx_vision_optimizations import optimize_gemma4_positions


def clipped_matmul(projection, value, weight=None):
    value = mx.clip(value, projection.input_min, projection.input_max)
    if weight is None:
        weight = projection.linear.weight
    output = value @ weight.T
    return mx.clip(output, projection.output_min, projection.output_max)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-4-e4b-it-4bit",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mx.set_wired_limit(2 * 1024**3)
    model, _ = load(args.model)
    tower = model.vision_tower
    optimize_gemma4_positions(tower)
    pixels = mx.load(str(args.input))["pixels"]
    layer = tower.encoder.layers[0]

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
    mx.eval(normalized, mlp_input, down_input)

    operations = {
        "attention_q": (attention.q_proj, normalized),
        "mlp_gate": (layer.mlp.gate_proj, mlp_input),
        "mlp_down": (layer.mlp.down_proj, down_input),
    }
    row_chunks = [256, 512, 768, 1024, 1536, 2048]
    column_chunks = [256, 384, 512, 768, 1024, 1536]
    pad_multiples = [64, 128, 256, 512]
    results = {}

    for operation_name, (projection, value) in operations.items():
        weight = projection.linear.weight
        rows = value.shape[-2]
        columns = weight.shape[0]
        candidates = {
            "stock": lambda p=projection, x=value: p(x),
        }
        for chunk in row_chunks:
            if chunk >= rows:
                continue

            def row_chunked(p=projection, x=value, size=chunk):
                clipped = mx.clip(x, p.input_min, p.input_max)
                parts = [
                    clipped[..., start : start + size, :] @ p.linear.weight.T
                    for start in range(0, clipped.shape[-2], size)
                ]
                return mx.clip(
                    mx.concatenate(parts, axis=-2),
                    p.output_min,
                    p.output_max,
                )

            candidates[f"row_chunk_{chunk}"] = row_chunked
        for chunk in column_chunks:
            if chunk >= columns:
                continue

            def column_chunked(p=projection, x=value, size=chunk):
                clipped = mx.clip(x, p.input_min, p.input_max)
                parts = [
                    clipped @ p.linear.weight[start : start + size].T
                    for start in range(0, p.linear.weight.shape[0], size)
                ]
                return mx.clip(
                    mx.concatenate(parts, axis=-1),
                    p.output_min,
                    p.output_max,
                )

            candidates[f"column_chunk_{chunk}"] = column_chunked
        for multiple in pad_multiples:
            padded_rows = ((rows + multiple - 1) // multiple) * multiple
            if padded_rows == rows:
                continue

            def row_padded(
                p=projection,
                x=value,
                target_rows=padded_rows,
                original_rows=rows,
            ):
                clipped = mx.clip(x, p.input_min, p.input_max)
                padded = mx.pad(
                    clipped,
                    [(0, 0), (0, target_rows - original_rows), (0, 0)],
                )
                output = padded @ p.linear.weight.T
                return mx.clip(
                    output[..., :original_rows, :],
                    p.output_min,
                    p.output_max,
                )

            candidates[f"row_pad_{padded_rows}"] = row_padded

        compiled = {name: mx.compile(function) for name, function in candidates.items()}
        metrics = measure_interleaved(compiled, args.rounds)
        reference = compiled["stock"]()
        differences = {
            name: difference(reference, function())
            for name, function in compiled.items()
            if name != "stock"
        }
        results[operation_name] = {
            "shape": [rows, weight.shape[1], columns],
            "metrics": metrics,
            "differences": differences,
        }
        winner = min(metrics, key=lambda name: metrics[name]["p50_ms"])
        print(
            f"{operation_name}: stock={metrics['stock']['p50_ms']:.3f} ms, "
            f"winner={winner} {metrics[winner]['p50_ms']:.3f} ms",
            flush=True,
        )

    output = {
        "metadata": {
            "model": args.model,
            "input": str(args.input),
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
