"""Performance and numerical A/B for cell-local progressive ToMe on Gemma 4."""

import argparse
import gc
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm import load

from .mlx_early_pool_ab import (
    DEFAULT_CORPUS,
    MODEL_ID,
    _paired_summary,
    _run_interleaved,
    memory_snapshot,
    timing_summary,
)
from .mlx_mixed_shape_benchmark import output_difference
from .mlx_vision_optimizations import (
    fuse_gemma4_qkv_epilogue,
    make_segmented_gemma4_encoder,
    optimize_gemma4_positions,
    prepare_gemma4_rope_constants,
)


DEFAULT_OUTPUT = Path("benchmark-results/mlx-roofline/progressive-tome/performance.json")
TOME_SCHEDULE = ((8, 6), (12, 3), (16, 1))
TOME_SCHEDULES = {
    "late-safe": TOME_SCHEDULE,
    "very-late": ((12, 6), (14, 3), (16, 1)),
    "last-two": ((14, 6), (15, 3), (16, 1)),
    "final-block": ((15, 6), (16, 3), (16, 1)),
}


def _validate_merge_shapes(hidden, sizes, positions, target_tokens: int) -> None:
    if hidden.ndim != 4 or hidden.shape[0] != 1:
        raise ValueError("hidden must be [1, cells, tokens, width]")
    if sizes.shape != hidden.shape[:-1]:
        raise ValueError("sizes must match hidden [1, cells, tokens]")
    if positions.shape != (*hidden.shape[:-1], 2):
        raise ValueError("positions must be [1, cells, tokens, 2]")
    tokens = hidden.shape[2]
    if target_tokens <= 0 or target_tokens >= tokens:
        raise ValueError("target_tokens must be positive and smaller than tokens")
    source_count = (tokens + 1) // 2
    if tokens - target_tokens > source_count:
        raise ValueError("target requires more merges than the bipartite source set")


def tome_bipartite_merge(
    hidden, sizes, positions, target_tokens: int, *, position_mode="destination"
):
    """ToMe soft matching independently in each cell; no attention modification."""
    _validate_merge_shapes(hidden, sizes, positions, target_tokens)
    if position_mode not in ("destination", "centroid"):
        raise ValueError("position_mode must be destination or centroid")
    tokens = hidden.shape[2]
    source_indices = mx.arange(0, tokens, 2)
    destination_indices = mx.arange(1, tokens, 2)
    source = hidden[:, :, source_indices]
    destination = hidden[:, :, destination_indices]

    source_norm = mx.sqrt(mx.sum(source.astype(mx.float32) ** 2, axis=-1, keepdims=True))
    destination_norm = mx.sqrt(
        mx.sum(destination.astype(mx.float32) ** 2, axis=-1, keepdims=True)
    )
    normalized_source = source.astype(mx.float32) / mx.maximum(source_norm, 1e-12)
    normalized_destination = destination.astype(mx.float32) / mx.maximum(
        destination_norm, 1e-12
    )
    scores = normalized_source @ mx.swapaxes(normalized_destination, -1, -2)
    best_destination = mx.argmax(scores, axis=-1)
    best_score = mx.max(scores, axis=-1)
    merge_count = tokens - target_tokens
    selected_indices = mx.argsort(-best_score, axis=-1)[..., :merge_count]
    selected = mx.any(
        mx.expand_dims(mx.arange(source.shape[2]), (0, 1, 3))
        == mx.expand_dims(selected_indices, -2),
        axis=-1,
    )

    assignment = (
        mx.expand_dims(best_destination, -1)
        == mx.arange(destination.shape[2]).reshape(1, 1, 1, -1)
    ) & mx.expand_dims(selected, -1)
    assignment_float = assignment.astype(mx.float32)
    source_sizes = sizes[:, :, source_indices].astype(mx.float32)
    destination_sizes = sizes[:, :, destination_indices].astype(mx.float32)
    source_mass = source.astype(mx.float32) * mx.expand_dims(source_sizes, -1)
    destination_mass = destination.astype(mx.float32) * mx.expand_dims(
        destination_sizes, -1
    )
    merged_mass = destination_mass + mx.einsum(
        "bcst,bcsd->bctd", assignment_float, source_mass
    )
    merged_sizes = destination_sizes + mx.einsum(
        "bcst,bcs->bct", assignment_float, source_sizes
    )
    merged_destination = (merged_mass / mx.expand_dims(merged_sizes, -1)).astype(
        hidden.dtype
    )

    destination_positions = positions[:, :, destination_indices]
    if position_mode == "centroid":
        source_positions = positions[:, :, source_indices]
        position_mass = destination_positions.astype(mx.float32) * mx.expand_dims(
            destination_sizes, -1
        ) + mx.einsum(
            "bcst,bcsd->bctd",
            assignment_float,
            source_positions.astype(mx.float32) * mx.expand_dims(source_sizes, -1),
        )
        centroids = position_mass / mx.expand_dims(merged_sizes, -1)
        destination_distance = mx.sum(
            (destination_positions.astype(mx.float32) - centroids) ** 2, axis=-1
        )
        source_distance = mx.sum(
            (
                mx.expand_dims(source_positions.astype(mx.float32), 3)
                - mx.expand_dims(centroids, 2)
            )
            ** 2,
            axis=-1,
        )
        source_distance = mx.where(assignment, source_distance, float("inf"))
        nearest_source = mx.argmin(source_distance, axis=2)
        nearest_source_distance = mx.min(source_distance, axis=2)
        nearest_source_position = mx.take_along_axis(
            source_positions,
            mx.broadcast_to(
                mx.expand_dims(nearest_source, -1),
                (*nearest_source.shape, source_positions.shape[-1]),
            ),
            axis=2,
        )
        destination_positions = mx.where(
            mx.expand_dims(nearest_source_distance < destination_distance, -1),
            nearest_source_position,
            destination_positions,
        )

    keep_count = source.shape[2] - merge_count
    survivor_order = mx.argsort(selected.astype(mx.int32), axis=-1)[..., :keep_count]
    survivor_hidden = mx.take_along_axis(
        source, mx.expand_dims(survivor_order, -1), axis=2
    )
    survivor_sizes = mx.take_along_axis(source_sizes, survivor_order, axis=2)
    source_positions = positions[:, :, source_indices]
    survivor_positions = mx.take_along_axis(
        source_positions, mx.expand_dims(survivor_order, -1), axis=2
    )
    return (
        mx.concatenate((survivor_hidden, merged_destination), axis=2),
        mx.concatenate((survivor_sizes, merged_sizes), axis=2),
        mx.concatenate((survivor_positions, destination_positions), axis=2),
    )


def tome_bipartite_merge_numpy(hidden, sizes, positions, target_tokens: int):
    """Small NumPy reference used to test the MLX merge primitive."""
    hidden = np.asarray(hidden)
    sizes = np.asarray(sizes)
    positions = np.asarray(positions)
    _validate_merge_shapes(hidden, sizes, positions, target_tokens)
    tokens = hidden.shape[2]
    source_indices = np.arange(0, tokens, 2)
    destination_indices = np.arange(1, tokens, 2)
    output_hidden, output_sizes, output_positions = [], [], []
    for cell in range(hidden.shape[1]):
        source = hidden[0, cell, source_indices].astype(np.float32)
        destination = hidden[0, cell, destination_indices].astype(np.float32)
        source_unit = source / np.maximum(np.linalg.norm(source, axis=-1, keepdims=True), 1e-12)
        destination_unit = destination / np.maximum(
            np.linalg.norm(destination, axis=-1, keepdims=True), 1e-12
        )
        scores = source_unit @ destination_unit.T
        best_destination = scores.argmax(axis=-1)
        merge_count = tokens - target_tokens
        selected_indices = np.argsort(-scores.max(axis=-1), kind="stable")[:merge_count]
        selected = np.zeros(len(source_indices), dtype=bool)
        selected[selected_indices] = True
        destination_mass = destination * sizes[0, cell, destination_indices, None]
        destination_sizes = sizes[0, cell, destination_indices].astype(np.float32).copy()
        for source_offset in selected_indices:
            destination_offset = best_destination[source_offset]
            source_size = sizes[0, cell, source_indices[source_offset]]
            destination_mass[destination_offset] += source[source_offset] * source_size
            destination_sizes[destination_offset] += source_size
        survivors = np.flatnonzero(~selected)
        output_hidden.append(
            np.concatenate(
                (source[survivors], destination_mass / destination_sizes[:, None]), axis=0
            )
        )
        output_sizes.append(
            np.concatenate((sizes[0, cell, source_indices[survivors]], destination_sizes))
        )
        output_positions.append(
            np.concatenate(
                (
                    positions[0, cell, source_indices[survivors]],
                    positions[0, cell, destination_indices],
                ),
                axis=0,
            )
        )
    return (
        np.asarray(output_hidden)[None].astype(hidden.dtype),
        np.asarray(output_sizes)[None],
        np.asarray(output_positions)[None],
    )


def _cell_layout(value, patch_height: int, patch_width: int):
    trailing = value.shape[2:]
    return mx.transpose(
        value.reshape(1, patch_height // 3, 3, patch_width // 3, 3, *trailing),
        (0, 1, 3, 2, 4, *range(5, 5 + len(trailing))),
    ).reshape(1, patch_height * patch_width // 9, 9, *trailing)


def make_progressive_tome_encoder(
    tower,
    projector=None,
    *,
    schedule=TOME_SCHEDULE,
    proportional_attention=False,
    position_mode="destination",
    segment_size=3,
    evaluate_segments=True,
):
    """Build the benchmark-only 9->6->3->1 cell-local ToMe encoder."""
    if tuple(target for _, target in schedule) != (6, 3, 1):
        raise ValueError("Progressive ToMe schedule must reduce each cell 9->6->3->1")
    merge_blocks = tuple(block for block, _ in schedule)
    if any(left > right for left, right in zip(merge_blocks, merge_blocks[1:])):
        raise ValueError("Progressive ToMe merge blocks must be nondecreasing")
    if len(tower.encoder.layers) < merge_blocks[-1]:
        raise ValueError(
            f"Progressive ToMe requires at least {merge_blocks[-1]} blocks"
        )
    if segment_size <= 0:
        raise ValueError("segment_size must be positive")
    patch = mx.compile(lambda pixels, positions, padding: tower.patch_embedder(pixels, positions, padding))
    boundaries = (0, *merge_blocks, len(tower.encoder.layers))
    stages = []
    for start, stop in zip(boundaries, boundaries[1:]):
        stage = []
        for offset in range(start, stop, segment_size):
            layers = tuple(tower.encoder.layers[offset : min(offset + segment_size, stop)])

            def run(hidden, cosine, sine, mask, selected_layers=layers):
                for layer in selected_layers:
                    hidden = layer(hidden, (cosine, sine), mask)
                return hidden

            stage.append(mx.compile(run))
        stages.append(stage)

    def finish(hidden):
        hidden = hidden * tower.pooler.root_hidden_size
        if tower.config.standardize:
            hidden = (hidden - tower.std_bias) * tower.std_scale
        hidden = hidden.astype(mx.bfloat16)
        return hidden if projector is None else projector(hidden)

    finish = mx.compile(finish)

    def encode(pixels):
        if pixels.ndim != 4 or pixels.shape[0] != 1:
            raise ValueError("Progressive ToMe requires NCHW batch size 1")
        _, _, height, width = pixels.shape
        if height % tower.patch_size or width % tower.patch_size:
            raise ValueError("Pixel dimensions must be divisible by patch size")
        patch_height = height // tower.patch_size
        patch_width = width // tower.patch_size
        if patch_height % 3 or patch_width % 3:
            raise ValueError("Patch-grid dimensions must both be divisible by 3")
        patch_count = patch_height * patch_width
        positions_np, padding_np, _ = tower._patch_positions_single(
            height, width, max_patches=patch_count
        )
        if np.any(padding_np):
            raise ValueError("Progressive ToMe does not support padded patch tokens")
        positions = mx.array(positions_np[None])
        padding = mx.array(padding_np[None])
        rope = prepare_gemma4_rope_constants(tower, positions)
        if rope is None:
            raise RuntimeError("Progressive ToMe requires QKV-default RoPE constants")
        hidden = patch(pixels, positions, padding)
        for segment in stages[0]:
            hidden = segment(hidden, *rope, None)
            if evaluate_segments:
                mx.eval(hidden)
        hidden = _cell_layout(hidden, patch_height, patch_width)
        positions = _cell_layout(positions, patch_height, patch_width)
        sizes = mx.ones(hidden.shape[:-1], dtype=mx.float32)
        for stage, (_, target_tokens) in zip(stages[1:], schedule):
            hidden, sizes, positions = tome_bipartite_merge(
                hidden,
                sizes,
                positions,
                target_tokens,
                position_mode=position_mode,
            )
            sequence_hidden = hidden.reshape(1, -1, hidden.shape[-1])
            sequence_positions = positions.reshape(1, -1, 2)
            rope = prepare_gemma4_rope_constants(tower, sequence_positions)
            attention_bias = None
            if proportional_attention:
                attention_bias = mx.log(sizes.reshape(1, -1)).reshape(
                    1, 1, 1, -1
                ).astype(sequence_hidden.dtype)
            for segment in stage:
                sequence_hidden = segment(
                    sequence_hidden, *rope, attention_bias
                )
                if evaluate_segments:
                    mx.eval(sequence_hidden)
            hidden = sequence_hidden.reshape(
                1, hidden.shape[1], target_tokens, hidden.shape[-1]
            )
        return finish(hidden.reshape(1, hidden.shape[1], hidden.shape[-1]))

    return encode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--numerical-cases", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--schedule", choices=TOME_SCHEDULES, default="late-safe")
    parser.add_argument("--proportional-attention", action="store_true")
    parser.add_argument(
        "--position-mode", choices=("destination", "centroid"), default="destination"
    )
    args = parser.parse_args()
    if args.warmups < 1 or args.rounds < 1 or args.numerical_cases < 1:
        raise ValueError("warmups, rounds, and numerical-cases must be positive")
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    if not 0 <= args.case_index < len(manifest["cases"]):
        raise ValueError("case-index is outside the frozen corpus")

    mx.set_wired_limit(2 * 1024**3)
    model, _ = load(args.model)
    tower = model.vision_tower
    optimize_gemma4_positions(tower)
    fuse_gemma4_qkv_epilogue(tower)
    baseline = make_segmented_gemma4_encoder(
        tower, projector=None, segment_size=3, evaluate_segments=True
    )
    schedule = TOME_SCHEDULES[args.schedule]
    candidate = make_progressive_tome_encoder(
        tower,
        schedule=schedule,
        proportional_attention=args.proportional_attention,
        position_mode=args.position_mode,
    )
    del model
    gc.collect()
    mx.clear_cache()

    numerical = []
    cases = manifest["cases"][: args.numerical_cases]
    for case in cases:
        patch_height, patch_width = case["patch_grid"]
        if patch_height % 3 or patch_width % 3:
            raise ValueError(f"Corpus case {case['case_id']} is not divisible by 3")
        pixels = mx.load(str(args.corpus / "cases" / case["case_id"] / "input.safetensors"))["pixels"]
        reference, value = baseline(pixels), candidate(pixels)
        mx.eval(reference, value)
        expected_shape = (1, patch_height * patch_width // 9, 768)
        if value.shape != expected_shape or value.dtype != mx.bfloat16:
            raise RuntimeError(f"Invalid candidate contract for {case['case_id']}: {value.shape} {value.dtype}")
        difference = output_difference(reference, value)
        if difference["nan_count"] or difference["inf_count"]:
            raise RuntimeError(f"Non-finite candidate output for {case['case_id']}")
        numerical.append({"case_id": case["case_id"], "patch_grid": case["patch_grid"], "difference": difference})

    performance_case = manifest["cases"][args.case_index]
    pixels = mx.load(str(args.corpus / "cases" / performance_case["case_id"] / "input.safetensors"))["pixels"]
    arms = {"baseline": baseline, "progressive_tome": candidate}
    measured = _run_interleaved(arms, pixels, args.warmups, args.rounds)
    timing = {name: timing_summary(values) for name, values in measured["timings"].items()}
    result = {
        "metadata": {
            "benchmark": "gemma4_progressive_cell_local_tome_performance_ab",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "mlx_version": importlib.metadata.version("mlx"),
            "mlx_vlm_version": importlib.metadata.version("mlx-vlm"),
            "baseline_graph": "QKV-default segmented encoder + stock final pool",
            "candidate_graph": "same QKV-default blocks with cell-local progressive ToMe",
            "schedule_name": args.schedule,
            "schedule": [{"after_one_based_block": block, "tokens_per_cell": tokens} for block, tokens in schedule],
            "similarity": "hidden-state cosine",
            "matching": "ToMe bipartite soft matching within each 3x3 cell only",
            "token_aggregation": "size-weighted hidden averages",
            "positions": f"{args.position_mode} member integer coordinate retained for RoPE",
            "proportional_attention": args.proportional_attention,
            "global_merges": False,
            "scale_standardize": "once after final merge",
            "segment_size": 3,
            "performance_case": performance_case["case_id"],
            "warmups": args.warmups,
            "rounds": args.rounds,
        },
        "timing": timing,
        "paired": _paired_summary(measured["timings"]["baseline"], measured["timings"]["progressive_tome"]),
        "numerical_cases": numerical,
        "memory_final": memory_snapshot(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
