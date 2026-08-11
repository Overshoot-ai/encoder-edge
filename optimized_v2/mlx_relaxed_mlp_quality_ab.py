"""Paired H200 ChartQA gate for the stabilized relaxed paired vision MLP."""

import argparse
import gc
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from mlx_vlm import load

from .mlx_mixed_shape_benchmark import output_difference
from .mlx_qkv_epilogue_quality_ab import (
    DATASET_ID,
    DEFAULT_CORPUS,
    MODEL_ID,
    Gateway,
    encode_cases,
    feature_summary,
    memory_snapshot,
    quality_summary,
    time_encoder,
)
from .mlx_relaxed_mlp_ab import RelaxedPairedVisionMLP, wrap_relaxed_mlps
from .mlx_vision_optimizations import (
    fuse_gemma4_qkv_epilogue,
    make_segmented_gemma4_encoder,
    optimize_gemma4_positions,
)
from .overshoot_eval import CHARTQA_INSTRUCTIONS, chartqa_scores


DEFAULT_OUTPUT = Path(
    "benchmark-results/mlx-roofline/relaxed-mlp/quality-paired"
)


def load_encoder(candidate: bool):
    model, _ = load(MODEL_ID)
    tower = model.vision_tower
    optimize_gemma4_positions(tower)
    fuse_gemma4_qkv_epilogue(tower)
    if candidate:
        wrap_relaxed_mlps(tower, RelaxedPairedVisionMLP)
    encode = make_segmented_gemma4_encoder(
        tower,
        projector=None,
        segment_size=3,
        evaluate_segments=True,
    )
    del model
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    return tower, encode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8002")
    parser.add_argument("--model", default="gemma-4-e4b-optimized")
    parser.add_argument("--limit", type=int, choices=(30, 100), default=30)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--max-soft-tokens", type=int, default=273)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.rounds < 10:
        raise ValueError("--rounds must be at least 10")
    args.output.mkdir(parents=True, exist_ok=True)

    corpus_manifest = json.loads((args.corpus / "manifest.json").read_text())
    if corpus_manifest["max_soft_tokens"] != args.max_soft_tokens:
        raise ValueError("cached corpus soft-token budget does not match")
    cases = []
    for cached_case in corpus_manifest["cases"][: args.limit]:
        cases.append(
            {
                "case_id": cached_case["case_id"],
                "index": cached_case["dataset_index"],
                "query": cached_case["query"],
                "targets": cached_case["targets"],
                "prompt": CHARTQA_INSTRUCTIONS.format(
                    question=cached_case["query"]
                ),
                "pixels": mx.load(
                    str(
                        args.corpus
                        / "cases"
                        / cached_case["case_id"]
                        / "input.safetensors"
                    )
                )["pixels"],
                "pixel_shape": cached_case["pixel_shape"],
            }
        )
    mx.eval(*(case["pixels"] for case in cases))

    baseline_tower, baseline_encode = load_encoder(candidate=False)
    baseline_features, baseline_encodes = encode_cases(
        baseline_encode, cases, "baseline"
    )
    baseline_timing = time_encoder(
        baseline_encode, cases[0]["pixels"], args.warmups, args.rounds
    )
    del baseline_encode, baseline_tower
    gc.collect()
    mx.clear_cache()
    mx.synchronize()

    candidate_tower, candidate_encode = load_encoder(candidate=True)
    candidate_features, candidate_encodes = encode_cases(
        candidate_encode, cases, "candidate"
    )
    candidate_timing = time_encoder(
        candidate_encode, cases[0]["pixels"], args.warmups, args.rounds
    )
    differences = []
    for baseline, candidate in zip(baseline_features, candidate_features):
        metrics = output_difference(baseline, candidate)
        metrics["bit_identical"] = metrics["differing_values"] == 0
        differences.append(metrics)
    if any(metric["nan_count"] or metric["inf_count"] for metric in differences):
        raise RuntimeError("non-finite candidate features block quality evaluation")

    raw_path = args.output / "paired_results.jsonl"
    records = (
        [json.loads(line) for line in raw_path.read_text().splitlines()]
        if raw_path.exists()
        else []
    )
    if len(records) > len(cases):
        raise ValueError("existing paired results exceed requested case limit")
    for offset, record in enumerate(records):
        if record.get("case_id") != cases[offset]["case_id"]:
            raise ValueError("existing quality results are not the frozen corpus prefix")

    gateway = Gateway(args.server, args.model)
    try:
        for offset, case in enumerate(cases[len(records) :], start=len(records)):
            arm_results = {}
            for variant, features, encodes in (
                ("baseline", baseline_features, baseline_encodes),
                ("candidate", candidate_features, candidate_encodes),
            ):
                generation, gateway_metrics = gateway.complete(
                    features[offset], case["prompt"], args.max_tokens
                )
                arm_results[variant] = {
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
                **arm_results,
                "exact_generation_agreement": (
                    arm_results["baseline"]["generation"]
                    == arm_results["candidate"]["generation"]
                ),
                "parsed_answer_agreement": (
                    arm_results["baseline"]["parsed_answer"].casefold()
                    == arm_results["candidate"]["parsed_answer"].casefold()
                ),
            }
            records.append(record)
            with raw_path.open("a") as output:
                output.write(json.dumps(record) + "\n")
            print(
                f"quality {offset + 1}/{len(cases)}: "
                f"relaxed={arm_results['baseline']['relaxed_accuracy']:.0f}/"
                f"{arm_results['candidate']['relaxed_accuracy']:.0f}",
                flush=True,
            )
    finally:
        gateway.close()

    quality = quality_summary(records)
    stage = "extend_to_100" if args.limit == 30 and quality["quality_gate_pass"] else (
        "complete" if args.limit == 100 else "stop_at_30"
    )
    result = {
        "metadata": {
            "benchmark": "relaxed_paired_mlp_chartqa_quality_ab",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_checkpoint": MODEL_ID,
            "decoder_model": args.model,
            "dataset": DATASET_ID,
            "dataset_split": "test",
            "dataset_fingerprint": corpus_manifest["dataset_fingerprint"],
            "corpus": str(args.corpus),
            "server": args.server,
            "cases": len(cases),
            "temperature": 0,
            "segment_size": 3,
            "evaluate_segments": True,
            "baseline_graph": "reassociated fused QKV + stock MLP",
            "candidate_graph": "reassociated fused QKV + stabilized relaxed paired MLP",
            "mlx_version": importlib.metadata.version("mlx"),
            "mlx_vlm_version": importlib.metadata.version("mlx-vlm"),
            "device_info": mx.device_info(),
        },
        "encoder_timing": {
            "baseline": baseline_timing,
            "candidate": candidate_timing,
        },
        "numerical_features": feature_summary(differences),
        "quality": quality,
        "next_stage": stage,
        "memory_final": memory_snapshot(),
        "artifacts": {
            "summary": str(args.output / "summary.json"),
            "paired_results": str(raw_path),
        },
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
