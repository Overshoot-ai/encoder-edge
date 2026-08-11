import argparse
import gc
import json
import math
import statistics
import time
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
            "mean": statistics.mean(values),
            "p50": percentile(values, 0.5),
            "p90": percentile(values, 0.9),
            "min": min(values),
            "max": max(values),
            "peak_memory_gb": mx.get_peak_memory() / 1e9,
        },
        output,
    )


def difference(reference, candidate) -> dict:
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
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
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
    tower = model.vision_tower
    projector = model.embed_vision
    del model
    gc.collect()
    mx.clear_cache()
    mx.synchronize()

    batch, _, height, width = pixels.shape
    patch_height = height // tower.patch_size
    patch_width = width // tower.patch_size
    num_patches = patch_height * patch_width
    output_length = num_patches // (tower.pooling_kernel_size**2)
    positions_np, padding_np, _ = tower._patch_positions_single(
        height,
        width,
        max_patches=num_patches,
    )
    positions = mx.array(np.tile(positions_np[None], (batch, 1, 1)))
    padding = mx.array(np.tile(padding_np[None], (batch, 1)))
    valid = ~padding
    mask = mx.expand_dims(valid, 1) * mx.expand_dims(valid, 2)
    mask = mx.where(mask, mx.array(0.0, dtype=mx.bfloat16), -1e4)
    mask = mx.expand_dims(mask, 1)

    channels_per_dimension = tower.config.head_dim // positions.shape[-1]
    half_per_dimension = channels_per_dimension // 2
    frequency_exponents = (2.0 / channels_per_dimension) * mx.arange(
        half_per_dimension
    ).astype(mx.float32)
    timescale = mx.power(
        tower.config.rope_parameters["rope_theta"],
        frequency_exponents,
    )
    rope_constants = []
    for dimension in range(positions.shape[-1]):
        sinusoid = (
            positions[..., dimension : dimension + 1].astype(mx.float32) / timescale
        )
        cosine = mx.cos(sinusoid)
        sine = mx.sin(sinusoid)
        rope_constants.append(
            (
                mx.expand_dims(
                    mx.concatenate([cosine, cosine], axis=-1).astype(mx.bfloat16),
                    2,
                ),
                mx.expand_dims(
                    mx.concatenate([sine, sine], axis=-1).astype(mx.bfloat16),
                    2,
                ),
            )
        )
    mx.eval(*(value for pair in rope_constants for value in pair))

    position_table = tower.patch_embedder.position_embedding_table
    qkv_weights = []
    gate_up_weights = []
    for layer in tower.encoder.layers:
        attention = layer.self_attn
        qkv_weights.append(
            mx.concatenate(
                [
                    attention.q_proj.linear.weight,
                    attention.k_proj.linear.weight,
                    attention.v_proj.linear.weight,
                ],
                axis=0,
            )
        )
        mlp = layer.mlp
        gate_up_weights.append(
            mx.concatenate(
                [mlp.gate_proj.linear.weight, mlp.up_proj.linear.weight],
                axis=0,
            )
        )
    mx.eval(*qkv_weights, *gate_up_weights)

    def gathered_patch_embedding(value):
        hidden_states = tower.patch_embedder._patchify(value)
        embeddings = (
            position_table[0, positions[..., 0]]
            + position_table[1, positions[..., 1]]
        )
        return hidden_states + embeddings

    def apply_precomputed_rope(value):
        result = []
        for dimension, (cosine, sine) in enumerate(rope_constants):
            start = dimension * channels_per_dimension
            part = value[..., start : start + channels_per_dimension]
            first, second = mx.split(part, 2, axis=-1)
            rotated = mx.concatenate([-second, first], axis=-1)
            result.append(part * cosine + rotated * sine)
        return mx.concatenate(result, axis=-1)

    def attention_forward(index, attention, value, precompute_rope, fuse_qkv):
        batch_size, length, _ = value.shape
        if fuse_qkv:
            clipped = mx.clip(
                value,
                attention.q_proj.input_min,
                attention.q_proj.input_max,
            )
            qkv = clipped @ qkv_weights[index].T
            q, k, v = mx.split(qkv, 3, axis=-1)
            q = mx.clip(q, attention.q_proj.output_min, attention.q_proj.output_max)
            k = mx.clip(k, attention.k_proj.output_min, attention.k_proj.output_max)
            v = mx.clip(v, attention.v_proj.output_min, attention.v_proj.output_max)
        else:
            q = attention.q_proj(value)
            k = attention.k_proj(value)
            v = attention.v_proj(value)
        q = q.reshape(batch_size, length, attention.num_heads, attention.head_dim)
        k = k.reshape(batch_size, length, attention.num_kv_heads, attention.head_dim)
        v = v.reshape(batch_size, length, attention.num_kv_heads, attention.head_dim)
        q = attention.q_norm(q)
        k = attention.k_norm(k)
        v = attention._v_norm(v)
        if precompute_rope:
            q = apply_precomputed_rope(q)
            k = apply_precomputed_rope(k)
        else:
            q = apply_multidimensional_rope(
                q,
                positions,
                attention.rope_base_frequency,
            )
            k = apply_multidimensional_rope(
                k,
                positions,
                attention.rope_base_frequency,
            )
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        output = ensure_fused_sdpa(q, k, v, scale=1.0, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, length, -1)
        return attention.o_proj(output)

    def mlp_forward(index, mlp, value, fuse_mlp):
        if not fuse_mlp:
            return mlp(value)
        clipped = mx.clip(value, mlp.gate_proj.input_min, mlp.gate_proj.input_max)
        gate_up = clipped @ gate_up_weights[index].T
        gate, up = mx.split(gate_up, 2, axis=-1)
        gate = mx.clip(gate, mlp.gate_proj.output_min, mlp.gate_proj.output_max)
        up = mx.clip(up, mlp.up_proj.output_min, mlp.up_proj.output_max)
        return mlp.down_proj(nn.gelu_approx(gate) * up)

    def candidate(value, *, precompute_rope: bool, fuse_qkv: bool, fuse_mlp: bool):
        hidden_states = gathered_patch_embedding(value)
        for index, layer in enumerate(tower.encoder.layers):
            normed = layer.input_layernorm(hidden_states)
            attention_output = attention_forward(
                index,
                layer.self_attn,
                normed,
                precompute_rope,
                fuse_qkv,
            )
            attention_output = layer.post_attention_layernorm(attention_output)
            hidden_states = hidden_states + attention_output
            normed = layer.pre_feedforward_layernorm(hidden_states)
            mlp_output = mlp_forward(index, layer.mlp, normed, fuse_mlp)
            mlp_output = layer.post_feedforward_layernorm(mlp_output)
            hidden_states = hidden_states + mlp_output
        hidden_states, _ = tower.pooler(
            hidden_states,
            positions,
            padding,
            output_length=output_length,
        )
        return projector(hidden_states)

    def baseline(value):
        return projector(tower(value, None))

    functions = {
        "baseline": mx.compile(baseline),
        "gather_control": mx.compile(
            lambda value: candidate(
                value,
                precompute_rope=False,
                fuse_qkv=False,
                fuse_mlp=False,
            )
        ),
        "fused_qkv": mx.compile(
            lambda value: candidate(
                value,
                precompute_rope=False,
                fuse_qkv=True,
                fuse_mlp=False,
            )
        ),
        "precomputed_rope": mx.compile(
            lambda value: candidate(
                value,
                precompute_rope=True,
                fuse_qkv=False,
                fuse_mlp=False,
            )
        ),
        "fused_mlp": mx.compile(
            lambda value: candidate(
                value,
                precompute_rope=False,
                fuse_qkv=False,
                fuse_mlp=True,
            )
        ),
        "fused_qkv_mlp": mx.compile(
            lambda value: candidate(
                value,
                precompute_rope=False,
                fuse_qkv=True,
                fuse_mlp=True,
            )
        ),
    }

    results = {}
    outputs = {}
    for name, function in functions.items():
        metrics, output = benchmark(function, pixels, args.rounds)
        results[name] = metrics
        outputs[name] = output
        print(f"{name}: p50={metrics['p50']:.3f} ms", flush=True)

    result = {
        "model": args.model,
        "device": str(mx.default_device()),
        "rounds": args.rounds,
        "pixel_shape": list(pixels.shape),
        "results": results,
        "comparisons_to_baseline": {
            name: difference(outputs["baseline"], output)
            for name, output in outputs.items()
            if name != "baseline"
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
