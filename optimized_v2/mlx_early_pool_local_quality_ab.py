"""Paired local-MLX ChartQA fallback gate for fixed Gemma 4 early pooling."""

import argparse
import gc
import importlib.metadata
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from PIL import Image
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

from .mlx_early_pool_ab import (
    DEFAULT_CORPUS,
    MODEL_ID,
    POOL_POINTS,
    make_early_pool_encoder,
    memory_snapshot,
)
from .mlx_mixed_shape_benchmark import output_difference
from .mlx_qkv_epilogue_quality_ab import feature_summary
from .mlx_vision_optimizations import (
    fuse_gemma4_qkv_epilogue,
    optimize_gemma4_positions,
)
from .overshoot_eval import chartqa_scores


DEFAULT_OUTPUT = Path("benchmark-results/mlx-roofline/early-pool/quality-local")
WIRED_LIMIT = 2 * 1024**3


def concise_prompt(question: str) -> str:
    return (
        f"{question}\nInspect the chart and respond with only: "
        "Final Answer: <answer>"
    )


def generate_with_features(
    model,
    processor,
    prompt: str,
    prepared: dict,
    projected_features: mx.array,
    max_tokens: int,
) -> tuple[str, float]:
    kwargs = {
        key: value
        for key, value in prepared.items()
        if key not in ("input_ids", "pixel_values", "attention_mask")
    }
    started = time.perf_counter()
    result = generate(
        model,
        processor,
        prompt,
        input_ids=prepared["input_ids"],
        pixel_values=prepared["pixel_values"],
        mask=prepared.get("attention_mask"),
        cached_image_features=projected_features,
        temperature=0,
        max_tokens=max_tokens,
        verbose=False,
        **kwargs,
    )
    mx.synchronize()
    return result.text, (time.perf_counter() - started) * 1000


def summarize_quality(records: list[dict]) -> dict:
    arms = {}
    for variant in ("baseline", "candidate"):
        arms[variant] = {
            metric: statistics.mean(record[variant][metric] for record in records)
            for metric in ("exact_match", "relaxed_accuracy", "anywhere_accuracy")
        }

    def paired(metric: str) -> dict:
        outcomes = {
            "both_correct": 0,
            "baseline_only": 0,
            "candidate_only": 0,
            "both_incorrect": 0,
        }
        for record in records:
            baseline = bool(record["baseline"][metric])
            candidate = bool(record["candidate"][metric])
            if baseline and candidate:
                outcomes["both_correct"] += 1
            elif baseline:
                outcomes["baseline_only"] += 1
            elif candidate:
                outcomes["candidate_only"] += 1
            else:
                outcomes["both_incorrect"] += 1
        return outcomes

    relaxed = paired("relaxed_accuracy")
    gate_pass = bool(
        arms["candidate"]["relaxed_accuracy"]
        >= arms["baseline"]["relaxed_accuracy"]
        and relaxed["candidate_only"] >= relaxed["baseline_only"]
    )
    return {
        "arms": arms,
        "paired_relaxed_outcomes": relaxed,
        "paired_exact_outcomes": paired("exact_match"),
        "paired_anywhere_outcomes": paired("anywhere_accuracy"),
        "exact_generation_agreement": statistics.mean(
            record["exact_generation_agreement"] for record in records
        ),
        "parsed_answer_agreement": statistics.mean(
            record["parsed_answer_agreement"] for record in records
        ),
        "relaxed_accuracy_delta": (
            arms["candidate"]["relaxed_accuracy"]
            - arms["baseline"]["relaxed_accuracy"]
        ),
        "gate_definition": (
            "candidate relaxed accuracy >= baseline and candidate-only relaxed "
            "wins >= baseline-only relaxed losses"
        ),
        "quality_gate_pass": gate_pass,
        "decision": "requires_h200_confirmation" if gate_pass else "reject",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-after", type=int, choices=POOL_POINTS, required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    if args.limit > len(manifest["cases"]):
        raise ValueError("--limit exceeds the frozen corpus")
    output_dir = args.output / f"pool-after-{args.pool_after}"
    output_dir.mkdir(parents=True, exist_ok=True)

    mx.set_wired_limit(WIRED_LIMIT)
    model, processor = load(MODEL_ID)
    processor.image_processor.max_soft_tokens = manifest["max_soft_tokens"]
    tower = model.vision_tower
    projector = model.embed_vision
    optimize_gemma4_positions(tower)
    fuse_gemma4_qkv_epilogue(tower)
    baseline_encode = make_early_pool_encoder(tower, None, None)
    candidate_encode = make_early_pool_encoder(tower, None, args.pool_after)

    cases = manifest["cases"][: args.limit]
    baseline_features = []
    candidate_features = []
    differences = []
    feature_records = []
    for offset, case in enumerate(cases):
        pixels = mx.load(
            str(args.corpus / "cases" / case["case_id"] / "input.safetensors")
        )["pixels"]
        mx.eval(pixels)
        started = time.perf_counter()
        baseline = baseline_encode(pixels)
        mx.eval(baseline)
        mx.synchronize()
        baseline_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        candidate = candidate_encode(pixels)
        mx.eval(candidate)
        mx.synchronize()
        candidate_ms = (time.perf_counter() - started) * 1000
        if (
            baseline.shape != candidate.shape
            or baseline.shape[-1] != 768
            or baseline.dtype != mx.bfloat16
            or candidate.dtype != mx.bfloat16
        ):
            raise RuntimeError(
                f"Invalid pre-projector contract for {case['case_id']}: "
                f"{baseline.shape}/{baseline.dtype}, {candidate.shape}/{candidate.dtype}"
            )
        difference = output_difference(baseline, candidate)
        difference["bit_identical"] = difference["differing_values"] == 0
        if difference["nan_count"] or difference["inf_count"]:
            raise RuntimeError(f"Non-finite features for {case['case_id']}")
        baseline_features.append(baseline)
        candidate_features.append(candidate)
        differences.append(difference)
        feature_records.append(
            {
                "case_id": case["case_id"],
                "pixel_shape": case["pixel_shape"],
                "feature_shape": list(baseline.shape),
                "baseline_encode_ms": baseline_ms,
                "candidate_encode_ms": candidate_ms,
                "difference": difference,
            }
        )
        print(
            f"features {offset + 1}/{len(cases)}: {case['case_id']} "
            f"tokens={baseline.shape[1]}",
            flush=True,
        )

    del baseline_encode, candidate_encode, tower
    model.vision_tower = None
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    memory_after_tower_release = memory_snapshot()

    raw_path = output_dir / "paired_results.jsonl"
    records = (
        [json.loads(line) for line in raw_path.read_text().splitlines()]
        if raw_path.exists()
        else []
    )
    if len(records) > len(cases):
        raise ValueError("Existing paired results exceed the requested limit")
    for offset, record in enumerate(records):
        if record.get("case_id") != cases[offset]["case_id"]:
            raise ValueError("Existing results are not the frozen-corpus prefix")
        if record.get("pool_after_one_based_block") != args.pool_after:
            raise ValueError("Existing results use a different early-pool schedule")
    resumed_records = len(records)

    for offset in range(resumed_records, len(cases)):
        case = cases[offset]
        prompt = concise_prompt(case["query"])
        chat_prompt = apply_chat_template(
            processor, model.config, prompt, num_images=1
        )
        image = Image.open(
            args.corpus / "cases" / case["case_id"] / "source.png"
        ).convert("RGB")
        prepared = prepare_inputs(
            processor,
            images=[image],
            prompts=chat_prompt,
            add_special_tokens=False,
        )
        if list(prepared["pixel_values"].shape) != case["pixel_shape"]:
            raise RuntimeError(
                f"Prepared image geometry changed for {case['case_id']}: "
                f"{prepared['pixel_values'].shape} != {case['pixel_shape']}"
            )

        projected = {
            "baseline": projector(baseline_features[offset]),
            "candidate": projector(candidate_features[offset]),
        }
        mx.eval(projected["baseline"], projected["candidate"])
        mx.synchronize()
        order = ["baseline", "candidate"]
        if offset % 2:
            order.reverse()
        arms = {}
        for variant in order:
            generation, generation_ms = generate_with_features(
                model,
                processor,
                chat_prompt,
                prepared,
                projected[variant],
                args.max_tokens,
            )
            arms[variant] = {
                "generation": generation,
                **chartqa_scores(generation, case["targets"]),
                "generation_ms": generation_ms,
            }
        record = {
            "position": offset,
            "dataset_index": case["dataset_index"],
            "case_id": case["case_id"],
            "query": case["query"],
            "targets": case["targets"],
            "pixel_shape": case["pixel_shape"],
            "pool_after_one_based_block": args.pool_after,
            "arm_order": order,
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
            f"quality {offset + 1}/{len(cases)}: order={'/'.join(order)} "
            f"relaxed={arms['baseline']['relaxed_accuracy']:.0f}/"
            f"{arms['candidate']['relaxed_accuracy']:.0f} "
            f"agreement={int(record['exact_generation_agreement'])}",
            flush=True,
        )

    quality = summarize_quality(records)
    report = {
        "metadata": {
            "benchmark": "gemma4_fixed_early_pool_local_mlx_quality_ab",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": MODEL_ID,
            "dataset": manifest["dataset"],
            "dataset_split": manifest["dataset_split"],
            "dataset_fingerprint": manifest["dataset_fingerprint"],
            "corpus": str(args.corpus),
            "cases": len(cases),
            "resumed_quality_records": resumed_records,
            "pool_after_one_based_block": args.pool_after,
            "decoder": "single shared local MLX Gemma E4B instance",
            "projector": "single shared local model.embed_vision instance",
            "feature_contract": "pre-projector BF16 [1,tokens,768]",
            "prompt": "concise ChartQA Final Answer only",
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "arm_order": "alternating by frozen-corpus position",
            "wired_limit_bytes": WIRED_LIMIT,
            "mlx_version": importlib.metadata.version("mlx"),
            "mlx_vlm_version": importlib.metadata.version("mlx-vlm"),
            "device_info": mx.device_info(),
            "note": "Local paired fallback; a passing schedule still requires H200 confirmation.",
        },
        "numerical_features": feature_summary(differences),
        "feature_records": feature_records,
        "quality": quality,
        "memory_after_tower_release": memory_after_tower_release,
        "memory_final": memory_snapshot(),
        "artifacts": {
            "summary": str(output_dir / "summary.json"),
            "paired_results": str(raw_path),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
