"""Resumable paired 30/100-case H200 gate for progressive cell-local ToMe."""

import argparse
import gc
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from mlx_vlm import load

from .mlx_early_pool_ab import DEFAULT_CORPUS, MODEL_ID, memory_snapshot
from .mlx_mixed_shape_benchmark import output_difference
from .mlx_progressive_tome_ab import TOME_SCHEDULES, make_progressive_tome_encoder
from .mlx_qkv_epilogue_quality_ab import (
    DATASET_ID,
    Gateway,
    encode_cases,
    feature_summary,
    quality_summary,
)
from .mlx_vision_optimizations import (
    fuse_gemma4_qkv_epilogue,
    make_segmented_gemma4_encoder,
    optimize_gemma4_positions,
)
from .overshoot_eval import CHARTQA_INSTRUCTIONS, chartqa_scores


DEFAULT_OUTPUT = Path("benchmark-results/mlx-roofline/progressive-tome/quality")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8002")
    parser.add_argument("--model", default="gemma-4-e4b-optimized")
    parser.add_argument("--limit", type=int, choices=(30, 100), default=30)
    parser.add_argument("--max-soft-tokens", type=int, default=273)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument("--schedule", choices=TOME_SCHEDULES, default="late-safe")
    parser.add_argument("--proportional-attention", action="store_true")
    parser.add_argument(
        "--position-mode", choices=("destination", "centroid"), default="destination"
    )
    args = parser.parse_args()
    schedule = TOME_SCHEDULES[args.schedule]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    if manifest["max_soft_tokens"] != args.max_soft_tokens:
        raise ValueError("Cached corpus soft-token budget does not match")
    if args.limit > len(manifest["cases"]):
        raise ValueError("Requested limit exceeds the frozen corpus")

    cases = []
    for cached in manifest["cases"][: args.limit]:
        patch_height, patch_width = cached["patch_grid"]
        if patch_height % 3 or patch_width % 3:
            raise ValueError(f"Corpus case {cached['case_id']} is not divisible by 3")
        pixels = mx.load(str(args.corpus / "cases" / cached["case_id"] / "input.safetensors"))["pixels"]
        mx.eval(pixels)
        cases.append({
            "case_id": cached["case_id"],
            "index": cached["dataset_index"],
            "query": cached["query"],
            "targets": cached["targets"],
            "prompt": CHARTQA_INSTRUCTIONS.format(question=cached["query"]),
            "pixels": pixels,
            "pixel_shape": cached["pixel_shape"],
        })

    mx.set_wired_limit(2 * 1024**3)
    model, _ = load(MODEL_ID)
    tower = model.vision_tower
    optimize_gemma4_positions(tower)
    fuse_gemma4_qkv_epilogue(tower)
    baseline_encode = make_segmented_gemma4_encoder(
        tower, projector=None, segment_size=3, evaluate_segments=True
    )
    candidate_encode = make_progressive_tome_encoder(
        tower,
        schedule=schedule,
        proportional_attention=args.proportional_attention,
        position_mode=args.position_mode,
    )
    del model
    gc.collect()
    mx.clear_cache()

    baseline_features, baseline_encodes = encode_cases(baseline_encode, cases, "baseline")
    candidate_features, candidate_encodes = encode_cases(candidate_encode, cases, "candidate")
    differences = []
    for baseline, candidate in zip(baseline_features, candidate_features):
        if baseline.shape != candidate.shape or candidate.shape[-1] != 768:
            raise RuntimeError(f"Feature contract mismatch: {baseline.shape} != {candidate.shape}")
        difference = output_difference(baseline, candidate)
        difference["bit_identical"] = difference["differing_values"] == 0
        if difference["nan_count"] or difference["inf_count"]:
            raise RuntimeError("Non-finite candidate features block quality evaluation")
        differences.append(difference)

    local_path = args.output / f"local-features-{len(cases)}.json"
    local_features = {
        "metadata": {
            "cases": len(cases), "schedule_name": args.schedule,
            "schedule": schedule, "proportional_attention": args.proportional_attention,
            "position_mode": args.position_mode,
        },
        "summary": feature_summary(differences),
        "cases": [
            {"case_id": case["case_id"], "pixel_shape": case["pixel_shape"], "baseline": baseline_encodes[index], "candidate": candidate_encodes[index], "difference": differences[index]}
            for index, case in enumerate(cases)
        ],
    }
    local_path.write_text(json.dumps(local_features, indent=2) + "\n")
    if args.features_only:
        print(json.dumps(local_features, indent=2))
        return

    raw_path = args.output / "paired_results.jsonl"
    records = [json.loads(line) for line in raw_path.read_text().splitlines()] if raw_path.exists() else []
    if len(records) > len(cases):
        raise ValueError("Existing paired results exceed requested case limit")
    for index, record in enumerate(records):
        if (
            record.get("case_id") != cases[index]["case_id"]
            or record.get("schedule") != [list(item) for item in schedule]
            or record.get("proportional_attention") != args.proportional_attention
            or record.get("position_mode") != args.position_mode
        ):
            raise ValueError("Existing results do not match this frozen-corpus schedule")
    resumed_records = len(records)

    gateway = Gateway(args.server, args.model)
    try:
        for index in range(len(records), len(cases)):
            case = cases[index]
            order = ["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"]
            arms = {}
            features = {"baseline": baseline_features[index], "candidate": candidate_features[index]}
            encodes = {"baseline": baseline_encodes[index], "candidate": candidate_encodes[index]}
            for variant in order:
                generation, gateway_metrics = gateway.complete(features[variant], case["prompt"], args.max_tokens)
                arms[variant] = {"generation": generation, **chartqa_scores(generation, case["targets"]), "gateway": gateway_metrics, "encoder": encodes[variant]}
            record = {
                "index": case["index"], "case_id": case["case_id"], "query": case["query"],
                "targets": case["targets"], "pixel_shape": case["pixel_shape"],
                "schedule": [list(item) for item in schedule], "arm_order": order,
                "proportional_attention": args.proportional_attention,
                "position_mode": args.position_mode,
                "feature_difference": differences[index], **arms,
                "exact_generation_agreement": arms["baseline"]["generation"] == arms["candidate"]["generation"],
                "parsed_answer_agreement": arms["baseline"]["parsed_answer"].casefold() == arms["candidate"]["parsed_answer"].casefold(),
            }
            records.append(record)
            with raw_path.open("a") as output:
                output.write(json.dumps(record) + "\n")
            print(f"quality {index + 1}/{len(cases)}: relaxed={arms['baseline']['relaxed_accuracy']:.0f}/{arms['candidate']['relaxed_accuracy']:.0f}", flush=True)
    finally:
        gateway.close()

    quality = quality_summary(records)
    next_stage = "extend_to_100" if args.limit == 30 and quality["quality_gate_pass"] else ("promote" if args.limit == 100 and quality["quality_gate_pass"] else "reject")
    result = {
        "metadata": {
            "benchmark": "gemma4_progressive_cell_local_tome_chartqa_quality_ab",
            "created_at": datetime.now(timezone.utc).isoformat(), "model_checkpoint": MODEL_ID,
            "decoder_model": args.model, "dataset": DATASET_ID, "dataset_split": "test",
            "dataset_fingerprint": manifest["dataset_fingerprint"], "corpus": str(args.corpus),
            "server": args.server, "cases": len(cases), "resumed_quality_records": resumed_records,
            "max_soft_tokens": args.max_soft_tokens, "max_generation_tokens": args.max_tokens,
            "temperature": 0, "schedule_name": args.schedule,
            "schedule": [{"after_one_based_block": block, "tokens_per_cell": tokens} for block, tokens in schedule],
            "baseline_graph": "QKV-default segmented encoder + stock final pool",
            "candidate_graph": "same QKV-default blocks with cell-local progressive ToMe",
            "matching": "hidden cosine ToMe bipartite soft matching, local to each 3x3 cell",
            "aggregation": "size-weighted hidden averages",
            "positions": f"{args.position_mode} member integer coordinate",
            "proportional_attention": args.proportional_attention, "global_merges": False,
            "segment_size": 3, "scale_standardize": "once after final merge",
            "feature_contract": "pre-projector BF16 [1,cells,768]",
            "mlx_version": importlib.metadata.version("mlx"), "mlx_vlm_version": importlib.metadata.version("mlx-vlm"),
            "device_info": mx.device_info(),
        },
        "numerical_features": feature_summary(differences), "quality": quality, "next_stage": next_stage,
        "memory_final": memory_snapshot(),
        "artifacts": {"summary": str(args.output / "summary.json"), "paired_results": str(raw_path), "local_features": str(local_path)},
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
