"""Resumable paired H200 ChartQA gate for fixed Gemma 4 early pooling."""

import argparse
import gc
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from mlx_vlm import load

from .mlx_early_pool_ab import (
    DEFAULT_CORPUS,
    MODEL_ID,
    POOL_POINTS,
    make_early_pool_encoder,
    memory_snapshot,
)
from .mlx_mixed_shape_benchmark import output_difference
from .mlx_qkv_epilogue_quality_ab import (
    DATASET_ID,
    Gateway,
    encode_cases,
    feature_summary,
    quality_summary,
)
from .mlx_vision_optimizations import (
    fuse_gemma4_qkv_epilogue,
    optimize_gemma4_positions,
)
from .overshoot_eval import CHARTQA_INSTRUCTIONS, chartqa_scores


DEFAULT_OUTPUT = Path("benchmark-results/mlx-roofline/early-pool/quality")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-after", type=int, choices=POOL_POINTS, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8002")
    parser.add_argument("--model", default="gemma-4-e4b-optimized")
    parser.add_argument("--limit", type=int, choices=(30, 100), default=30)
    parser.add_argument("--max-soft-tokens", type=int, default=273)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Write local corpus feature evidence without contacting the H200",
    )
    args = parser.parse_args()
    output_dir = args.output / f"pool-after-{args.pool_after}"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.corpus / "manifest.json").read_text())
    if manifest["max_soft_tokens"] != args.max_soft_tokens:
        raise ValueError("Cached corpus soft-token budget does not match")
    cases = []
    for cached in manifest["cases"][: args.limit]:
        pixels = mx.load(
            str(args.corpus / "cases" / cached["case_id"] / "input.safetensors")
        )["pixels"]
        mx.eval(pixels)
        cases.append(
            {
                "case_id": cached["case_id"],
                "index": cached["dataset_index"],
                "query": cached["query"],
                "targets": cached["targets"],
                "prompt": CHARTQA_INSTRUCTIONS.format(question=cached["query"]),
                "pixels": pixels,
                "pixel_shape": cached["pixel_shape"],
            }
        )

    mx.set_wired_limit(2 * 1024**3)
    model, _ = load(MODEL_ID)
    tower = model.vision_tower
    optimize_gemma4_positions(tower)
    fuse_gemma4_qkv_epilogue(tower)
    baseline_encode = make_early_pool_encoder(tower, None, None)
    candidate_encode = make_early_pool_encoder(tower, None, args.pool_after)
    del model
    gc.collect()
    mx.clear_cache()
    mx.synchronize()

    baseline_features, baseline_encodes = encode_cases(
        baseline_encode, cases, "baseline"
    )
    candidate_features, candidate_encodes = encode_cases(
        candidate_encode, cases, "candidate"
    )
    differences = []
    for baseline, candidate in zip(baseline_features, candidate_features):
        if baseline.shape != candidate.shape:
            raise RuntimeError(
                f"Feature shape mismatch: {baseline.shape} != {candidate.shape}"
            )
        metrics = output_difference(baseline, candidate)
        metrics["bit_identical"] = metrics["differing_values"] == 0
        differences.append(metrics)
    if any(item["nan_count"] or item["inf_count"] for item in differences):
        raise RuntimeError("Non-finite candidate features block quality evaluation")

    local_features = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pool_after_one_based_block": args.pool_after,
            "cases": len(cases),
            "corpus": str(args.corpus),
        },
        "summary": feature_summary(differences),
        "cases": [
            {
                "case_id": case["case_id"],
                "pixel_shape": case["pixel_shape"],
                "baseline": baseline_encodes[offset],
                "candidate": candidate_encodes[offset],
                "difference": differences[offset],
            }
            for offset, case in enumerate(cases)
        ],
    }
    local_path = output_dir / f"local-features-{len(cases)}.json"
    local_path.write_text(json.dumps(local_features, indent=2) + "\n")
    if args.features_only:
        print(json.dumps(local_features, indent=2))
        return

    raw_path = output_dir / "paired_results.jsonl"
    records = (
        [json.loads(line) for line in raw_path.read_text().splitlines()]
        if raw_path.exists()
        else []
    )
    if len(records) > len(cases):
        raise ValueError("Existing paired results exceed requested case limit")
    for offset, record in enumerate(records):
        if record.get("case_id") != cases[offset]["case_id"]:
            raise ValueError("Existing results are not the requested frozen-corpus prefix")
    resumed_records = len(records)

    gateway = Gateway(args.server, args.model)
    try:
        for offset, case in enumerate(cases[len(records) :], start=len(records)):
            arms = {}
            for variant, features, encodes in (
                ("baseline", baseline_features, baseline_encodes),
                ("candidate", candidate_features, candidate_encodes),
            ):
                generation, gateway_metrics = gateway.complete(
                    features[offset], case["prompt"], args.max_tokens
                )
                arms[variant] = {
                    "generation": generation,
                    **chartqa_scores(generation, case["targets"]),
                    "gateway": gateway_metrics,
                    "encoder": encodes[offset],
                }
            record = {
                "index": case["index"],
                "case_id": case["case_id"],
                "query": case["query"],
                "targets": case["targets"],
                "pixel_shape": case["pixel_shape"],
                "feature_difference": differences[offset],
                **arms,
                "exact_generation_agreement": (
                    arms["baseline"]["generation"] == arms["candidate"]["generation"]
                ),
                "parsed_answer_agreement": (
                    arms["baseline"]["parsed_answer"].casefold()
                    == arms["candidate"]["parsed_answer"].casefold()
                ),
            }
            records.append(record)
            with raw_path.open("a") as output:
                output.write(json.dumps(record) + "\n")
            print(
                f"quality {offset + 1}/{len(cases)} pool-after-{args.pool_after}: "
                f"relaxed={arms['baseline']['relaxed_accuracy']:.0f}/"
                f"{arms['candidate']['relaxed_accuracy']:.0f}",
                flush=True,
            )
    finally:
        gateway.close()

    quality = quality_summary(records)
    next_stage = (
        "extend_to_100"
        if args.limit == 30 and quality["quality_gate_pass"]
        else ("promote" if args.limit == 100 and quality["quality_gate_pass"] else "reject")
    )
    result = {
        "metadata": {
            "benchmark": "gemma4_fixed_early_pool_chartqa_quality_ab",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_checkpoint": MODEL_ID,
            "decoder_model": args.model,
            "dataset": DATASET_ID,
            "dataset_split": "test",
            "dataset_fingerprint": manifest["dataset_fingerprint"],
            "corpus": str(args.corpus),
            "server": args.server,
            "cases": len(cases),
            "resumed_quality_records": resumed_records,
            "max_soft_tokens": args.max_soft_tokens,
            "max_generation_tokens": args.max_tokens,
            "temperature": 0,
            "pool_after_one_based_block": args.pool_after,
            "pool_kernel": [3, 3],
            "pool_reduction": "unscaled valid-patch spatial mean",
            "post_pool_positions": "center coordinate of each 3x3 cell",
            "segment_size": 3,
            "evaluate_segments": True,
            "projector_location": "H200",
            "baseline_graph": "QKV-default segmented encoder + stock final pool",
            "candidate_graph": "same graph with one intermediate pool and final-only scale/standardize",
            "mlx_version": importlib.metadata.version("mlx"),
            "mlx_vlm_version": importlib.metadata.version("mlx-vlm"),
            "device_info": mx.device_info(),
        },
        "numerical_features": feature_summary(differences),
        "quality": quality,
        "next_stage": next_stage,
        "memory_final": memory_snapshot(),
        "artifacts": {
            "summary": str(output_dir / "summary.json"),
            "paired_results": str(raw_path),
            "local_features": str(local_path),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
