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
from mlx_vlm.models.base import ensure_fused_sdpa
from mlx_vlm.models.gemma4.vision import apply_multidimensional_rope
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

from .mlx_vision_optimizations import optimize_gemma4_positions


QKV_EPILOGUE = mx.fast.metal_kernel(
    name="gemma4_vision_qkv_epilogue",
    input_names=["qkv", "output_mins", "output_maxs", "norm_weights", "cosine", "sine"],
    output_names=["out"],
    source="""
        uint d = thread_position_in_threadgroup.x;
        uint token = threadgroup_position_in_grid.y;
        uint plane = threadgroup_position_in_grid.z;
        uint kind = plane / 12;
        uint head = plane % 12;
        uint lane = thread_index_in_simdgroup;
        uint simdgroup = simdgroup_index_in_threadgroup;

        uint input_index = token * 2304 + kind * 768 + head * 64 + d;
        float value = static_cast<float>(qkv[input_index]);
        value = clamp(
            value,
            static_cast<float>(output_mins[kind]),
            static_cast<float>(output_maxs[kind])
        );

        float partial = simd_sum(value * value);
        threadgroup float sums[2];
        if (lane == 0) sums[simdgroup] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simdgroup == 0) {
            float total = lane < 2 ? sums[lane] : 0.0f;
            total = simd_sum(total);
            if (lane == 0) sums[0] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float inverse_rms = rsqrt(sums[0] * (1.0f / 64.0f) + 1.0e-6f);
        T normalized = static_cast<T>(
            value * inverse_rms * static_cast<float>(norm_weights[kind * 64 + d])
        );
        threadgroup T normalized_values[64];
        normalized_values[d] = normalized;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        T result = normalized;
        if (kind < 2) {
            uint part_d = d & 31;
            uint partner = part_d < 16 ? d + 16 : d - 16;
            float rotated = part_d < 16
                ? -static_cast<float>(normalized_values[partner])
                : static_cast<float>(normalized_values[partner]);
            uint rope_index = token * 64 + d;
            result = static_cast<T>(
                static_cast<float>(normalized) * static_cast<float>(cosine[rope_index])
                + rotated * static_cast<float>(sine[rope_index])
            );
        }
        uint output_index = ((kind * 12 + head) * 2376 + token) * 64 + d;
        out[output_index] = result;
    """,
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def measure(function, rounds: int) -> tuple[dict, mx.array]:
    compiled = mx.compile(function)
    for _ in range(5):
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


def measure_interleaved(functions: dict, rounds: int) -> dict:
    compiled = {name: mx.compile(function) for name, function in functions.items()}
    for function in compiled.values():
        for _ in range(3):
            mx.eval(function())
    mx.synchronize()
    names = list(compiled)
    values = {name: [] for name in names}
    for round_index in range(rounds):
        offset = round_index % len(names)
        order = names[offset:] + names[:offset]
        if (round_index % 2):
            order.reverse()
        for name in order:
            started = time.perf_counter()
            output = compiled[name]()
            mx.eval(output)
            mx.synchronize()
            values[name].append((time.perf_counter() - started) * 1000)
    return {
        name: {
            "mean_ms": statistics.mean(samples),
            "p50_ms": percentile(samples, 0.5),
            "p90_ms": percentile(samples, 0.9),
            "min_ms": min(samples),
            "max_ms": max(samples),
            "raw_ms": samples,
        }
        for name, samples in values.items()
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


def same_scalar(left: mx.array, right: mx.array) -> bool:
    mx.eval(left, right)
    return bool(
        np.array_equal(
            np.array(left.view(mx.uint16), copy=True),
            np.array(right.view(mx.uint16), copy=True),
        )
    )


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
    optimize_gemma4_positions(tower)
    del model
    gc.collect()
    mx.clear_cache()

    batch, _, height, width = pixels.shape
    length = (height // tower.patch_size) * (width // tower.patch_size)
    positions_np, padding_np, _ = tower._patch_positions_single(
        height,
        width,
        max_patches=length,
    )
    positions = mx.array(np.tile(positions_np[None], (batch, 1, 1)))
    padding = mx.array(np.tile(padding_np[None], (batch, 1)))
    valid = ~padding
    mask = mx.expand_dims(valid, 1) * mx.expand_dims(valid, 2)
    mask = mx.where(mask, mx.array(0.0, dtype=mx.bfloat16), -1e4)
    mask = mx.expand_dims(mask, 1)

    layer = tower.encoder.layers[0]
    attention = layer.self_attn
    hidden = tower.patch_embedder(pixels, positions, padding)
    normalized = layer.input_layernorm(hidden)
    mx.eval(normalized, positions, mask)

    for other in (attention.k_proj, attention.v_proj):
        if not same_scalar(attention.q_proj.input_min, other.input_min):
            raise ValueError("Q/K/V input minimum clipping bounds differ")
        if not same_scalar(attention.q_proj.input_max, other.input_max):
            raise ValueError("Q/K/V input maximum clipping bounds differ")

    q_weight = attention.q_proj.linear.weight
    k_weight = attention.k_proj.linear.weight
    v_weight = attention.v_proj.linear.weight
    qkv_weight = mx.concatenate([q_weight, k_weight, v_weight], axis=0)
    qkv_weight_t = mx.contiguous(qkv_weight.T)
    qk_weight = mx.concatenate([q_weight, k_weight], axis=0)
    kv_weight = mx.concatenate([k_weight, v_weight], axis=0)
    batched_weight_t = mx.stack([q_weight.T, k_weight.T, v_weight.T], axis=0)
    mx.eval(qkv_weight, qkv_weight_t, qk_weight, kv_weight, batched_weight_t)

    channels_per_dimension = attention.head_dim // positions.shape[-1]
    half_per_dimension = channels_per_dimension // 2
    frequency_exponents = (2.0 / channels_per_dimension) * mx.arange(
        half_per_dimension
    ).astype(mx.float32)
    timescale = mx.power(attention.rope_base_frequency, frequency_exponents)
    cosine_parts = []
    sine_parts = []
    for dimension in range(positions.shape[-1]):
        sinusoid = (
            positions[..., dimension : dimension + 1].astype(mx.float32) / timescale
        )
        cosine = mx.cos(sinusoid)
        sine = mx.sin(sinusoid)
        cosine_parts.append(
            mx.concatenate([cosine, cosine], axis=-1).astype(mx.bfloat16)
        )
        sine_parts.append(mx.concatenate([sine, sine], axis=-1).astype(mx.bfloat16))
    rope_cosine = mx.concatenate(cosine_parts, axis=-1).reshape(length, 64)
    rope_sine = mx.concatenate(sine_parts, axis=-1).reshape(length, 64)
    output_mins = mx.stack(
        [
            attention.q_proj.output_min,
            attention.k_proj.output_min,
            attention.v_proj.output_min,
        ]
    )
    output_maxs = mx.stack(
        [
            attention.q_proj.output_max,
            attention.k_proj.output_max,
            attention.v_proj.output_max,
        ]
    )
    norm_weights = mx.stack(
        [
            attention.q_norm.weight,
            attention.k_norm.weight,
            mx.ones((64,), dtype=attention.q_norm.weight.dtype),
        ]
    )
    mx.eval(rope_cosine, rope_sine, output_mins, output_maxs, norm_weights)

    def clip_outputs(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        q = mx.clip(q, attention.q_proj.output_min, attention.q_proj.output_max)
        k = mx.clip(k, attention.k_proj.output_min, attention.k_proj.output_max)
        v = mx.clip(v, attention.v_proj.output_min, attention.v_proj.output_max)
        return mx.concatenate([q, k, v], axis=-1)

    def separate_projection() -> mx.array:
        return mx.concatenate(
            [
                attention.q_proj(normalized),
                attention.k_proj(normalized),
                attention.v_proj(normalized),
            ],
            axis=-1,
        )

    def shared_clip_separate_projection() -> mx.array:
        value = mx.clip(normalized, attention.q_proj.input_min, attention.q_proj.input_max)
        return clip_outputs(
            attention.q_proj.linear(value),
            attention.k_proj.linear(value),
            attention.v_proj.linear(value),
        )

    def wide_projection() -> mx.array:
        value = mx.clip(normalized, attention.q_proj.input_min, attention.q_proj.input_max)
        q, k, v = mx.split(value @ qkv_weight.T, 3, axis=-1)
        return clip_outputs(q, k, v)

    def wide_contiguous_projection() -> mx.array:
        value = mx.clip(normalized, attention.q_proj.input_min, attention.q_proj.input_max)
        q, k, v = mx.split(value @ qkv_weight_t, 3, axis=-1)
        return clip_outputs(q, k, v)

    def q_plus_kv_projection() -> mx.array:
        q = attention.q_proj(normalized)
        value = mx.clip(normalized, attention.q_proj.input_min, attention.q_proj.input_max)
        k, v = mx.split(value @ kv_weight.T, 2, axis=-1)
        return clip_outputs(q, k, v)

    def qk_plus_v_projection() -> mx.array:
        value = mx.clip(normalized, attention.q_proj.input_min, attention.q_proj.input_max)
        q, k = mx.split(value @ qk_weight.T, 2, axis=-1)
        v = attention.v_proj(normalized)
        return clip_outputs(q, k, v)

    def batched_projection() -> mx.array:
        value = mx.clip(normalized, attention.q_proj.input_min, attention.q_proj.input_max)
        projected = mx.matmul(value, batched_weight_t)
        return clip_outputs(projected[0], projected[1], projected[2])

    projection_functions = {
        "separate": separate_projection,
        "shared_clip_separate": shared_clip_separate_projection,
        "wide_qkv": wide_projection,
        "wide_qkv_contiguous_weight": wide_contiguous_projection,
        "q_plus_kv": q_plus_kv_projection,
        "qk_plus_v": qk_plus_v_projection,
        "batched_qkv": batched_projection,
    }
    projection_results = {}
    projection_outputs = {}
    for name, function in projection_functions.items():
        metrics, output = measure(function, args.rounds)
        projection_results[name] = metrics
        projection_outputs[name] = output
        print(f"projection {name}: {metrics['p50_ms']:.3f} ms", flush=True)
    for name, output in projection_outputs.items():
        if name != "separate":
            projection_results[name]["difference"] = difference(
                projection_outputs["separate"], output
            )

    def baseline_qkv_postprocess() -> mx.array:
        q = attention.q_proj(normalized).reshape(batch, length, 12, 64)
        k = attention.k_proj(normalized).reshape(batch, length, 12, 64)
        v = attention.v_proj(normalized).reshape(batch, length, 12, 64)
        q = apply_multidimensional_rope(
            attention.q_norm(q), positions, attention.rope_base_frequency
        ).transpose(0, 2, 1, 3)
        k = apply_multidimensional_rope(
            attention.k_norm(k), positions, attention.rope_base_frequency
        ).transpose(0, 2, 1, 3)
        v = attention._v_norm(v).transpose(0, 2, 1, 3)
        return mx.stack([q[0], k[0], v[0]], axis=0)

    def fused_qkv_epilogue() -> mx.array:
        value = mx.clip(normalized, attention.q_proj.input_min, attention.q_proj.input_max)
        qkv = value @ qkv_weight.T
        return QKV_EPILOGUE(
            inputs=[qkv, output_mins, output_maxs, norm_weights, rope_cosine, rope_sine],
            output_shapes=[(3, 12, length, 64)],
            output_dtypes=[qkv.dtype],
            grid=(64, length, 36),
            threadgroup=(64, 1, 1),
            template=[("T", qkv.dtype)],
        )[0]

    postprocess_results = {}
    postprocess_outputs = {}
    for name, function in {
        "baseline": baseline_qkv_postprocess,
        "fused_metal_epilogue": fused_qkv_epilogue,
    }.items():
        metrics, output = measure(function, args.rounds)
        postprocess_results[name] = metrics
        postprocess_outputs[name] = output
        print(f"qkv postprocess {name}: {metrics['p50_ms']:.3f} ms", flush=True)
    postprocess_results["fused_metal_epilogue"]["difference"] = difference(
        postprocess_outputs["baseline"], postprocess_outputs["fused_metal_epilogue"]
    )

    q = attention.q_proj(normalized).reshape(
        batch, length, attention.num_heads, attention.head_dim
    )
    k = attention.k_proj(normalized).reshape(
        batch, length, attention.num_kv_heads, attention.head_dim
    )
    v = attention.v_proj(normalized).reshape(
        batch, length, attention.num_kv_heads, attention.head_dim
    )
    q = apply_multidimensional_rope(
        attention.q_norm(q), positions, attention.rope_base_frequency
    ).transpose(0, 2, 1, 3)
    k = apply_multidimensional_rope(
        attention.k_norm(k), positions, attention.rope_base_frequency
    ).transpose(0, 2, 1, 3)
    v = attention._v_norm(v).transpose(0, 2, 1, 3)
    mx.eval(q, k, v)

    def chunked_sdpa(chunk_size: int) -> mx.array:
        outputs = []
        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)
            outputs.append(
                ensure_fused_sdpa(
                    q[:, :, start:end],
                    k,
                    v,
                    scale=1.0,
                    mask=mask[:, :, start:end],
                )
            )
        return mx.concatenate(outputs, axis=2)

    sdpa_functions = {
        "masked_full": lambda: ensure_fused_sdpa(q, k, v, scale=1.0, mask=mask),
        "unmasked_full": lambda: ensure_fused_sdpa(q, k, v, scale=1.0, mask=None),
        "masked_chunk_256": lambda: chunked_sdpa(256),
        "masked_chunk_512": lambda: chunked_sdpa(512),
        "masked_chunk_792": lambda: chunked_sdpa(792),
        "masked_chunk_1188": lambda: chunked_sdpa(1188),
    }
    sdpa_results = {}
    sdpa_outputs = {}
    for name, function in sdpa_functions.items():
        metrics, output = measure(function, args.rounds)
        sdpa_results[name] = metrics
        sdpa_outputs[name] = output
        print(f"sdpa {name}: {metrics['p50_ms']:.3f} ms", flush=True)
    for name, output in sdpa_outputs.items():
        if name != "masked_full":
            sdpa_results[name]["difference"] = difference(
                sdpa_outputs["masked_full"], output
            )

    interleaved_rounds = max(args.rounds, 60)
    interleaved_results = {
        "rounds": interleaved_rounds,
        "projection": measure_interleaved(projection_functions, interleaved_rounds),
        "qkv_postprocess": measure_interleaved(
            {
                "baseline": baseline_qkv_postprocess,
                "fused_metal_epilogue": fused_qkv_epilogue,
            },
            interleaved_rounds,
        ),
        "sdpa": measure_interleaved(sdpa_functions, interleaved_rounds),
    }
    for group, values in interleaved_results.items():
        if group == "rounds":
            continue
        print(f"interleaved {group}", flush=True)
        for name, metrics in values.items():
            print(f"  {name}: {metrics['p50_ms']:.3f} ms", flush=True)

    result = {
        "metadata": {
            "model": args.model,
            "device": str(mx.default_device()),
            "device_info": mx.device_info(),
            "rounds": args.rounds,
            "pixel_shape": list(pixels.shape),
            "sequence_length": length,
            "attention_shape": [batch, attention.num_heads, length, attention.head_dim],
        },
        "projection_results": projection_results,
        "qkv_postprocess_results": postprocess_results,
        "sdpa_results": sdpa_results,
        "interleaved_results": interleaved_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
