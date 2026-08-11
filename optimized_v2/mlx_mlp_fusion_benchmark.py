import argparse
import json
import math
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm import load
from mlx_vlm.models.base import ensure_fused_sdpa
from mlx_vlm.models.gemma4.vision import apply_multidimensional_rope

from .mlx_kernel_candidate_benchmark import difference, measure_interleaved
from .mlx_vision_optimizations import optimize_gemma4_positions


GELU_PRODUCT_EPILOGUE = mx.fast.metal_kernel(
    name="gemma4_vision_gelu_product_epilogue",
    input_names=["gate", "up"],
    output_names=["out"],
    source="""
        uint index = thread_position_in_grid.x;
        T gate_value = gate[index];
        T up_value = up[index];
        T cubic = static_cast<T>(metal::pow(
            gate_value,
            static_cast<T>(3.0f)
        ));
        T scaled_cubic = static_cast<T>(
            static_cast<T>(0.044715f) * cubic
        );
        T inner = static_cast<T>(gate_value + scaled_cubic);
        T tanh_input = static_cast<T>(
            static_cast<T>(0.7978845608028654f) * inner
        );
        T tanh_value = static_cast<T>(metal::precise::tanh(
            static_cast<float>(tanh_input)
        ));
        T shifted = static_cast<T>(static_cast<T>(1.0f) + tanh_value);
        T half_gate = static_cast<T>(static_cast<T>(0.5f) * gate_value);
        T activated = static_cast<T>(half_gate * shifted);
        out[index] = static_cast<T>(activated * up_value);
    """,
)


def gelu_product_epilogue(gate: mx.array, up: mx.array) -> mx.array:
    return GELU_PRODUCT_EPILOGUE(
        inputs=[gate, up],
        output_shapes=[gate.shape],
        output_dtypes=[gate.dtype],
        grid=(gate.size, 1, 1),
        threadgroup=(256, 1, 1),
        template=[("T", gate.dtype)],
    )[0]


@partial(mx.compile, shapeless=True)
def compiled_gelu_product(gate: mx.array, up: mx.array) -> mx.array:
    return (
        0.5
        * gate
        * (
            1
            + mx.tanh(
                math.sqrt(2 / math.pi)
                * (gate + 0.044715 * gate**3)
            )
        )
        * up
    )


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
        height,
        width,
        max_patches=length,
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
    attention_input = attention_input.transpose(0, 2, 1, 3).reshape(
        1, length, -1
    )
    attention_output = attention.o_proj(attention_input)
    residual = hidden + layer.post_attention_layernorm(attention_output)
    mlp_input = layer.pre_feedforward_layernorm(residual)
    gate = layer.mlp.gate_proj(mlp_input)
    up = layer.mlp.up_proj(mlp_input)
    mx.eval(gate, up)

    reference = nn.gelu_approx(gate) * up
    candidate = gelu_product_epilogue(gate, up)
    compiled_candidate = compiled_gelu_product(gate, up)
    mx.eval(reference, candidate, compiled_candidate)
    product_difference = difference(reference, candidate)
    compiled_product_difference = difference(reference, compiled_candidate)

    pointwise = measure_interleaved(
        {
            "stock": lambda: nn.gelu_approx(gate) * up,
            "metal_epilogue": lambda: gelu_product_epilogue(gate, up),
            "compiled_gelu_product": lambda: compiled_gelu_product(gate, up),
        },
        args.rounds,
    )
    projection_product = measure_interleaved(
        {
            "stock": lambda: nn.gelu_approx(
                layer.mlp.gate_proj(mlp_input)
            )
            * layer.mlp.up_proj(mlp_input),
            "metal_epilogue": lambda: gelu_product_epilogue(
                layer.mlp.gate_proj(mlp_input),
                layer.mlp.up_proj(mlp_input),
            ),
            "compiled_gelu_product": lambda: compiled_gelu_product(
                layer.mlp.gate_proj(mlp_input),
                layer.mlp.up_proj(mlp_input),
            ),
        },
        args.rounds,
    )
    result = {
        "metadata": {
            "model": args.model,
            "input": str(args.input),
            "pixel_shape": list(pixels.shape),
            "mlp_input_shape": list(mlp_input.shape),
            "product_shape": list(reference.shape),
            "dtype": str(reference.dtype),
            "rounds": args.rounds,
        },
        "product_difference": product_difference,
        "compiled_product_difference": compiled_product_difference,
        "pointwise": pointwise,
        "projection_product": projection_product,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not (
        product_difference["bit_identical"]
        and compiled_product_difference["bit_identical"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
