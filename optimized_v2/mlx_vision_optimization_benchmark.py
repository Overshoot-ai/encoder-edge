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
    output = None
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
    reference_float = np.array(reference.astype(mx.float32), copy=True)
    candidate_float = np.array(candidate.astype(mx.float32), copy=True)
    absolute = np.abs(reference_float - candidate_float)
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
    patch_size = tower.patch_size
    pool_size = tower.pooling_kernel_size
    patch_height = height // patch_size
    patch_width = width // patch_size
    num_patches = patch_height * patch_width
    output_length = num_patches // (pool_size**2)
    positions_np, padding_np, _ = tower._patch_positions_single(
        height,
        width,
        max_patches=num_patches,
    )
    positions = mx.array(np.tile(positions_np[None], (batch, 1, 1)))
    padding = mx.array(np.tile(padding_np[None], (batch, 1)))

    def finish(hidden_states):
        if tower.config.standardize:
            hidden_states = (hidden_states - tower.std_bias) * tower.std_scale
        return projector(hidden_states)

    def baseline(value):
        return projector(tower(value, None))

    def fixed_path(value, *, use_mask: bool, gather_positions: bool, reshape_pool: bool):
        if gather_positions:
            hidden_states = tower.patch_embedder._patchify(value)
            table = tower.patch_embedder.position_embedding_table
            position_embeddings = (
                table[0, positions[..., 0]] + table[1, positions[..., 1]]
            )
            inputs_embeds = hidden_states + position_embeddings
        else:
            inputs_embeds = tower.patch_embedder(value, positions, padding)

        attention_mask = None
        if use_mask:
            valid = ~padding
            attention_mask = mx.expand_dims(valid, 1) * mx.expand_dims(valid, 2)
            attention_mask = mx.where(
                attention_mask,
                mx.array(0.0, dtype=inputs_embeds.dtype),
                mx.array(-1e4, dtype=inputs_embeds.dtype),
            )
            attention_mask = mx.expand_dims(attention_mask, 1)

        hidden_states = tower.encoder(inputs_embeds, positions, attention_mask)
        if reshape_pool:
            hidden_size = hidden_states.shape[-1]
            hidden_states = hidden_states.reshape(
                batch,
                patch_height // pool_size,
                pool_size,
                patch_width // pool_size,
                pool_size,
                hidden_size,
            )
            hidden_states = hidden_states.astype(mx.float32).mean(axis=(2, 4))
            hidden_states = hidden_states.reshape(batch, output_length, hidden_size)
            hidden_states = hidden_states.astype(inputs_embeds.dtype)
            hidden_states = hidden_states * tower.pooler.root_hidden_size
        else:
            hidden_states, _ = tower.pooler(
                hidden_states,
                positions,
                padding,
                output_length=output_length,
            )
        return finish(hidden_states)

    functions = {
        "baseline": mx.compile(baseline),
        "fixed_compaction": mx.compile(
            lambda value: fixed_path(
                value,
                use_mask=True,
                gather_positions=False,
                reshape_pool=False,
            )
        ),
        "no_attention_mask": mx.compile(
            lambda value: fixed_path(
                value,
                use_mask=False,
                gather_positions=False,
                reshape_pool=False,
            )
        ),
        "gathered_positions": mx.compile(
            lambda value: fixed_path(
                value,
                use_mask=False,
                gather_positions=True,
                reshape_pool=False,
            )
        ),
        "reshape_pool": mx.compile(
            lambda value: fixed_path(
                value,
                use_mask=False,
                gather_positions=True,
                reshape_pool=True,
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

    comparisons = {
        name: difference(outputs["baseline"], output)
        for name, output in outputs.items()
        if name != "baseline"
    }
    result = {
        "model": args.model,
        "device": str(mx.default_device()),
        "rounds": args.rounds,
        "pixel_shape": list(pixels.shape),
        "pixel_dtype": str(pixels.dtype),
        "patch_grid": [patch_height, patch_width],
        "num_patches": num_patches,
        "output_length": output_length,
        "results": results,
        "comparisons_to_baseline": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
