import argparse
import gc
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image, ImageOps
from mlx_vlm import load
from mlx_vlm.models.base import ensure_fused_sdpa
from mlx_vlm.models.gemma4.vision import apply_multidimensional_rope
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

from .mlx_vision_optimizations import optimize_gemma4_positions


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def measure(function, rounds: int) -> tuple[dict[str, float], object]:
    compiled = mx.compile(function)
    for _ in range(3):
        output = compiled()
        mx.eval(output)
    mx.synchronize()
    values = []
    for _ in range(rounds):
        started = time.perf_counter()
        output = compiled()
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
            "raw_ms": values,
        },
        output,
    )


def gemm_result(
    layer: int,
    name: str,
    metrics: dict,
    m: int,
    n: int,
    k: int,
    bytes_per_element: int = 2,
) -> dict:
    flops = 2 * m * n * k
    minimum_bytes = (m * k + n * k + m * n) * bytes_per_element
    seconds = metrics["p50_ms"] / 1000
    return {
        "layer": layer,
        "operation": name,
        "kind": "gemm",
        "m": m,
        "n": n,
        "k": k,
        "flops": flops,
        "minimum_bytes": minimum_bytes,
        "arithmetic_intensity_flops_per_byte": flops / minimum_bytes,
        "achieved_tflops_p50": flops / seconds / 1e12,
        "minimum_bandwidth_gbps_p50": minimum_bytes / seconds / 1e9,
        **metrics,
    }


def timed_result(layer: int, name: str, kind: str, metrics: dict, flops=0) -> dict:
    result = {
        "layer": layer,
        "operation": name,
        "kind": kind,
        "flops": flops,
        **metrics,
    }
    if flops:
        result["equivalent_tflops_p50"] = (
            flops / (metrics["p50_ms"] / 1000) / 1e12
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-4-e4b-it-4bit",
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=10)
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
    optimize_gemma4_positions(tower)
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
    valid = ~padding
    attention_mask = mx.expand_dims(valid, 1) * mx.expand_dims(valid, 2)
    attention_mask = mx.where(attention_mask, mx.array(0.0, dtype=mx.bfloat16), -1e4)
    attention_mask = mx.expand_dims(attention_mask, 1)

    patch_metrics, hidden = measure(
        lambda: tower.patch_embedder(pixels, positions, padding),
        args.rounds,
    )
    operations = [timed_result(-1, "patch_embedding", "stage", patch_metrics)]

    length = patch_tokens
    hidden_size = tower.config.hidden_size
    intermediate_size = tower.config.intermediate_size
    heads = tower.config.num_attention_heads
    head_dim = tower.config.head_dim

    for index, layer in enumerate(tower.encoder.layers):
        input_norm_metrics, normalized = measure(
            lambda current=layer, value=hidden: current.input_layernorm(value),
            args.rounds,
        )
        operations.append(
            timed_result(index, "input_layernorm", "normalization", input_norm_metrics)
        )
        attention = layer.self_attn
        projections = []
        for name, projection in (
            ("q_projection", attention.q_proj),
            ("k_projection", attention.k_proj),
            ("v_projection", attention.v_proj),
        ):
            metrics, output = measure(
                lambda current=projection, value=normalized: current(value),
                args.rounds,
            )
            operations.append(
                gemm_result(
                    index,
                    name,
                    metrics,
                    length,
                    hidden_size,
                    hidden_size,
                )
            )
            projections.append(output)
        q, k, v = projections
        q = q.reshape(batch, length, heads, head_dim)
        k = k.reshape(batch, length, heads, head_dim)
        v = v.reshape(batch, length, heads, head_dim)

        q_norm_metrics, q_normalized = measure(
            lambda: attention.q_norm(q),
            args.rounds,
        )
        k_norm_metrics, k_normalized = measure(
            lambda: attention.k_norm(k),
            args.rounds,
        )
        v_norm_metrics, v_normalized = measure(
            lambda: attention._v_norm(v),
            args.rounds,
        )
        operations.extend(
            [
                timed_result(index, "q_norm", "normalization", q_norm_metrics),
                timed_result(index, "k_norm", "normalization", k_norm_metrics),
                timed_result(index, "v_norm", "normalization", v_norm_metrics),
            ]
        )

        q_rope_metrics, q_rope = measure(
            lambda: apply_multidimensional_rope(
                q_normalized,
                positions,
                attention.rope_base_frequency,
            ),
            args.rounds,
        )
        k_rope_metrics, k_rope = measure(
            lambda: apply_multidimensional_rope(
                k_normalized,
                positions,
                attention.rope_base_frequency,
            ),
            args.rounds,
        )
        operations.extend(
            [
                timed_result(index, "q_rope", "rope", q_rope_metrics),
                timed_result(index, "k_rope", "rope", k_rope_metrics),
            ]
        )
        q_sdpa = q_rope.transpose(0, 2, 1, 3)
        k_sdpa = k_rope.transpose(0, 2, 1, 3)
        v_sdpa = v_normalized.transpose(0, 2, 1, 3)
        attention_flops = 4 * batch * heads * length * length * head_dim
        sdpa_metrics, sdpa_output = measure(
            lambda: ensure_fused_sdpa(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                scale=1.0,
                mask=attention_mask,
            ),
            args.rounds,
        )
        operations.append(
            timed_result(index, "sdpa", "attention", sdpa_metrics, attention_flops)
        )
        output_input = sdpa_output.transpose(0, 2, 1, 3).reshape(
            batch,
            length,
            hidden_size,
        )
        output_metrics, attention_output = measure(
            lambda: attention.o_proj(output_input),
            args.rounds,
        )
        operations.append(
            gemm_result(
                index,
                "output_projection",
                output_metrics,
                length,
                hidden_size,
                hidden_size,
            )
        )
        post_attention_metrics, normalized_attention = measure(
            lambda current=layer, value=attention_output: current.post_attention_layernorm(
                value
            ),
            args.rounds,
        )
        operations.append(
            timed_result(
                index,
                "post_attention_layernorm",
                "normalization",
                post_attention_metrics,
            )
        )
        residual = hidden + normalized_attention
        mx.eval(residual)
        pre_mlp_metrics, mlp_input = measure(
            lambda current=layer, value=residual: current.pre_feedforward_layernorm(
                value
            ),
            args.rounds,
        )
        operations.append(
            timed_result(index, "pre_mlp_layernorm", "normalization", pre_mlp_metrics)
        )
        mlp = layer.mlp
        gate_metrics, gate = measure(
            lambda: mlp.gate_proj(mlp_input),
            args.rounds,
        )
        up_metrics, up = measure(
            lambda: mlp.up_proj(mlp_input),
            args.rounds,
        )
        operations.extend(
            [
                gemm_result(
                    index,
                    "gate_projection",
                    gate_metrics,
                    length,
                    intermediate_size,
                    hidden_size,
                ),
                gemm_result(
                    index,
                    "up_projection",
                    up_metrics,
                    length,
                    intermediate_size,
                    hidden_size,
                ),
            ]
        )
        activated = nn.gelu_approx(gate) * up
        mx.eval(activated)
        down_metrics, mlp_output = measure(
            lambda: mlp.down_proj(activated),
            args.rounds,
        )
        operations.append(
            gemm_result(
                index,
                "down_projection",
                down_metrics,
                length,
                hidden_size,
                intermediate_size,
            )
        )
        post_mlp_metrics, normalized_mlp = measure(
            lambda current=layer, value=mlp_output: current.post_feedforward_layernorm(
                value
            ),
            args.rounds,
        )
        operations.append(
            timed_result(
                index,
                "post_mlp_layernorm",
                "normalization",
                post_mlp_metrics,
            )
        )

        whole_attention_metrics, _ = measure(
            lambda current=layer, value=normalized: current.self_attn(
                value,
                positions,
                attention_mask,
            ),
            args.rounds,
        )
        whole_mlp_metrics, _ = measure(
            lambda current=layer, value=mlp_input: current.mlp(value),
            args.rounds,
        )
        whole_layer_metrics, hidden = measure(
            lambda current=layer, value=hidden: current(
                value,
                positions,
                attention_mask,
            ),
            args.rounds,
        )
        operations.extend(
            [
                timed_result(
                    index,
                    "whole_attention",
                    "composite",
                    whole_attention_metrics,
                ),
                timed_result(
                    index,
                    "whole_mlp",
                    "composite",
                    whole_mlp_metrics,
                ),
                timed_result(
                    index,
                    "whole_layer",
                    "composite",
                    whole_layer_metrics,
                ),
            ]
        )
        print(f"profiled layer {index + 1}/16", flush=True)

    pool_metrics, pooled = measure(
        lambda: tower.pooler(
            hidden,
            positions,
            padding,
            output_length=output_tokens,
        )[0],
        args.rounds,
    )
    projector_metrics, _ = measure(lambda: projector(pooled), args.rounds)
    whole_encoder_metrics, _ = measure(
        lambda: projector(tower(pixels, None)),
        max(args.rounds, 20),
    )
    operations.extend(
        [
            timed_result(-1, "pool", "stage", pool_metrics),
            timed_result(-1, "projector", "stage", projector_metrics),
            timed_result(-1, "whole_encoder", "composite", whole_encoder_metrics),
        ]
    )

    gemms = [operation for operation in operations if operation["kind"] == "gemm"]
    sdpas = [operation for operation in operations if operation["operation"] == "sdpa"]
    theoretical_flops = sum(operation["flops"] for operation in gemms + sdpas)
    isolated_p50_ms = sum(operation["p50_ms"] for operation in gemms + sdpas)
    report = {
        "metadata": {
            "model": args.model,
            "device": str(mx.default_device()),
            "device_info": mx.device_info(),
            "rounds_per_operator": args.rounds,
            "pixel_shape": list(pixels.shape),
            "patch_tokens": patch_tokens,
            "visual_tokens": output_tokens,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
        "summary": {
            "theoretical_major_op_flops": theoretical_flops,
            "isolated_major_op_p50_sum_ms": isolated_p50_ms,
            "isolated_major_op_equivalent_tflops": theoretical_flops
            / (isolated_p50_ms / 1000)
            / 1e12,
            "whole_encoder_p50_ms": whole_encoder_metrics["p50_ms"],
            "whole_encoder_p90_ms": whole_encoder_metrics["p90_ms"],
            "whole_encoder_equivalent_tflops": theoretical_flops
            / (whole_encoder_metrics["p50_ms"] / 1000)
            / 1e12,
            "gemm_achieved_tflops_mean": statistics.mean(
                operation["achieved_tflops_p50"] for operation in gemms
            ),
            "gemm_achieved_tflops_min": min(
                operation["achieved_tflops_p50"] for operation in gemms
            ),
            "gemm_achieved_tflops_max": max(
                operation["achieved_tflops_p50"] for operation in gemms
            ),
        },
        "operations": operations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
