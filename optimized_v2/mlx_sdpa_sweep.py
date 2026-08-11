import argparse
import gc
import json
import math
from pathlib import Path

import mlx.core as mx
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
    hidden = layer.input_layernorm(tower.patch_embedder(pixels, positions, padding))
    attention = layer.self_attn
    q = attention.q_proj(hidden).reshape(1, length, 12, 64)
    k = attention.k_proj(hidden).reshape(1, length, 12, 64)
    v = attention.v_proj(hidden).reshape(1, length, 12, 64)
    q = apply_multidimensional_rope(
        attention.q_norm(q), positions, attention.rope_base_frequency
    ).transpose(0, 2, 1, 3)
    k = apply_multidimensional_rope(
        attention.k_norm(k), positions, attention.rope_base_frequency
    ).transpose(0, 2, 1, 3)
    v = attention._v_norm(v).transpose(0, 2, 1, 3)
    q_contiguous = mx.contiguous(q)
    k_contiguous = mx.contiguous(k)
    v_contiguous = mx.contiguous(v)
    mx.eval(q, k, v, q_contiguous, k_contiguous, v_contiguous)

    def sdpa(current_q=q, current_k=k, current_v=v):
        return ensure_fused_sdpa(
            current_q,
            current_k,
            current_v,
            scale=1.0,
            mask=None,
        )

    def chunked(size):
        return lambda: mx.concatenate(
            [
                ensure_fused_sdpa(
                    q[:, :, start : start + size],
                    k,
                    v,
                    scale=1.0,
                    mask=None,
                )
                for start in range(0, length, size)
            ],
            axis=2,
        )

    def padded(multiple):
        target = math.ceil(length / multiple) * multiple
        amount = target - length

        def run():
            padded_q = mx.pad(q, [(0, 0), (0, 0), (0, amount), (0, 0)])
            return ensure_fused_sdpa(
                padded_q,
                k,
                v,
                scale=1.0,
                mask=None,
            )[:, :, :length]

        return run

    def partition_heads(size):
        return lambda: mx.concatenate(
            [
                ensure_fused_sdpa(
                    q[:, start : start + size],
                    k[:, start : start + size],
                    v[:, start : start + size],
                    scale=1.0,
                    mask=None,
                )
                for start in range(0, q.shape[1], size)
            ],
            axis=1,
        )

    functions = {"full": sdpa}
    for size in (128, 256, 384, 512, 768, 1024, 1188):
        functions[f"query_chunk_{size}"] = chunked(size)
    for multiple in (64, 128, 256):
        functions[f"query_pad_{multiple}"] = padded(multiple)
    functions.update(
        {
            "contiguous_q": lambda: sdpa(q_contiguous, k, v),
            "contiguous_k": lambda: sdpa(q, k_contiguous, v),
            "contiguous_v": lambda: sdpa(q, k, v_contiguous),
            "contiguous_kv": lambda: sdpa(q, k_contiguous, v_contiguous),
            "contiguous_qkv": lambda: sdpa(
                q_contiguous, k_contiguous, v_contiguous
            ),
            "heads_6": partition_heads(6),
            "heads_4": partition_heads(4),
            "heads_3": partition_heads(3),
        }
    )
    metrics = measure_interleaved(functions, args.rounds)
    reference = functions["full"]()
    for name, function in functions.items():
        if name == "full":
            continue
        metrics[name]["difference"] = difference(reference, function())
    for name, values in sorted(metrics.items(), key=lambda item: item[1]["p50_ms"]):
        exact = values.get("difference", {}).get("bit_identical", True)
        print(f"{name}: {values['p50_ms']:.3f} ms exact={exact}", flush=True)

    result = {
        "metadata": {
            "model": args.model,
            "pixel_shape": list(pixels.shape),
            "sequence_length": length,
            "rounds": args.rounds,
            "device_info": mx.device_info(),
        },
        "results": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
