import argparse
import gc
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm import load

from .mlx_vision_optimizations import (
    _shapeless_multidimensional_rope,
    gemma4_rope_constants,
    gemma4_unpadded_inputs,
    optimize_gemma4_shapeless_positions,
    optimize_gemma4_shapeless_rope,
)


def metrics(reference, candidate):
    mx.eval(reference, candidate)
    reference_np = np.array(reference.astype(mx.float32), copy=True)
    candidate_np = np.array(candidate.astype(mx.float32), copy=True)
    if reference_np.shape != candidate_np.shape:
        return {
            "reference_shape": list(reference_np.shape),
            "shape": list(candidate_np.shape),
            "nan_count": int(np.isnan(candidate_np).sum()),
            "shape_mismatch": True,
        }
    return {
        "shape": list(candidate.shape),
        "nan_count": int(np.isnan(candidate_np).sum()),
        "differing_values": int(np.count_nonzero(reference_np != candidate_np)),
        "maximum_absolute_difference": float(
            np.nanmax(np.abs(reference_np - candidate_np), initial=0)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mx.set_wired_limit(2 * 1024**3)
    model, _ = load("mlx-community/gemma-4-e4b-it-4bit")
    tower = model.vision_tower
    optimize_gemma4_shapeless_positions(tower)
    optimize_gemma4_shapeless_rope(tower)
    layer = tower.encoder.layers[0]
    del model
    gc.collect()
    mx.clear_cache()

    cases = []
    for case_id in ("chartqa-0000", "chartqa-0004"):
        pixels = mx.load(
            str(args.corpus / "cases" / case_id / "input.safetensors")
        )["pixels"]
        positions, padding, _ = gemma4_unpadded_inputs(
            tower, pixels.shape[-2], pixels.shape[-1]
        )
        hidden = tower.patch_embedder(pixels, positions, padding)
        mx.eval(hidden)
        cosine, sine = gemma4_rope_constants(tower, positions)
        cases.append((mx.squeeze(hidden, axis=0), cosine, sine))

    def q_project(value):
        return mx.unflatten(
            layer.self_attn.q_proj(value),
            -1,
            (layer.self_attn.num_heads, layer.self_attn.head_dim),
        )

    def rope(value, cosine, sine):
        return _shapeless_multidimensional_rope(
            layer.self_attn.q_norm(q_project(value)),
            (cosine, sine),
            layer.self_attn.rope_base_frequency,
        )

    def rope_without_norm(value, cosine, sine):
        return _shapeless_multidimensional_rope(
            q_project(value),
            (cosine, sine),
            layer.self_attn.rope_base_frequency,
        )

    def q_sdpa(value, cosine, sine):
        q = rope(value, cosine, sine)
        q = mx.expand_dims(q.transpose(1, 0, 2), 0)
        return mx.fast.scaled_dot_product_attention(q, q, q, scale=1.0)

    functions = {
        "rope_cosine_identity": lambda value, cosine, sine: cosine,
        "input_norm": lambda value, cosine, sine: layer.input_layernorm(value),
        "q_input_clip": lambda value, cosine, sine: mx.clip(
            value,
            layer.self_attn.q_proj.input_min,
            layer.self_attn.q_proj.input_max,
        ),
        "q_linear_raw": lambda value, cosine, sine: layer.self_attn.q_proj.linear(value),
        "q_output_clip": lambda value, cosine, sine: mx.clip(
            layer.self_attn.q_proj.linear(value),
            layer.self_attn.q_proj.output_min,
            layer.self_attn.q_proj.output_max,
        ),
        "q_linear": lambda value, cosine, sine: layer.self_attn.q_proj(value),
        "q_projection": lambda value, cosine, sine: q_project(value),
        "q_norm": lambda value, cosine, sine: layer.self_attn.q_norm(q_project(value)),
        "q_rope_without_norm": rope_without_norm,
        "q_norm_rope": rope,
        "q_sdpa": q_sdpa,
        "attention": lambda value, cosine, sine: layer.self_attn(
            value, (cosine, sine), None
        ),
        "layer": lambda value, cosine, sine: layer(
            value, (cosine, sine), None
        ),
    }
    results = {}
    for name, function in functions.items():
        compiled = mx.compile(function, shapeless=True)
        first = compiled(*cases[0])
        mx.eval(first)
        candidate = compiled(*cases[1])
        reference = function(*cases[1])
        results[name] = metrics(reference, candidate)
        print(name, results[name], flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
