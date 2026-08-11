import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

from .mlx_vision_optimizations import (
    encode_gemma4_unpadded_batch1,
    optimize_gemma4_positions,
)
from .overshoot_eval import CHARTQA_INSTRUCTIONS, chartqa_scores


MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
WIRED_LIMIT = 2 * 1024**3


def generate_with_features(
    model,
    processor,
    prompt: str,
    prepared: dict,
    features: mx.array,
    max_tokens: int,
) -> str:
    kwargs = {
        key: value
        for key, value in prepared.items()
        if key not in ("input_ids", "pixel_values", "attention_mask")
    }
    result = generate(
        model,
        processor,
        prompt,
        input_ids=prepared["input_ids"],
        pixel_values=prepared["pixel_values"],
        mask=prepared.get("attention_mask"),
        cached_image_features=features,
        temperature=0,
        max_tokens=max_tokens,
        verbose=False,
        **kwargs,
    )
    return result.text


def summarize(records: list[dict]) -> dict:
    agreements = sum(record["exact_string_agreement"] for record in records)
    baseline_correct = sum(record["baseline_scores"]["relaxed_accuracy"] for record in records)
    candidate_correct = sum(record["candidate_scores"]["relaxed_accuracy"] for record in records)
    baseline_only = sum(record["disagreement_class"] == "baseline_only" for record in records)
    candidate_only = sum(record["disagreement_class"] == "candidate_only" for record in records)
    return {
        "samples": len(records),
        "exact_string_agreements": agreements,
        "exact_string_agreement_rate": agreements / len(records),
        "baseline_relaxed_correct": baseline_correct,
        "candidate_relaxed_correct": candidate_correct,
        "baseline_relaxed_accuracy": baseline_correct / len(records),
        "candidate_relaxed_accuracy": candidate_correct / len(records),
        "baseline_only_correct": baseline_only,
        "candidate_only_correct": candidate_only,
        "net_correctness_change": candidate_only - baseline_only,
        "feature_bit_mismatch_samples": sum(
            not record["feature_bit_identical"] for record in records
        ),
        "gate_b1": agreements / len(records) >= 0.97,
        "gate_b2": candidate_only - baseline_only >= 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--prompt-mode",
        choices=("concise", "reasoning"),
        default="concise",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.corpus / "manifest.json").read_text())
    if args.limit > len(manifest["cases"]):
        raise ValueError("Tier B limit exceeds the frozen corpus")
    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "raw_results.jsonl"
    records = (
        [json.loads(line) for line in raw_path.read_text().splitlines()]
        if raw_path.exists()
        else []
    )
    if len(records) > args.limit:
        raise ValueError("Existing Tier B output is longer than requested limit")

    model, processor = load(MODEL_ID)
    processor.image_processor.max_soft_tokens = manifest["max_soft_tokens"]
    optimize_gemma4_positions(model.vision_tower)
    encode = mx.compile(
        lambda value: encode_gemma4_unpadded_batch1(
            model.vision_tower,
            model.embed_vision,
            value,
        )
    )

    for position in range(len(records), args.limit):
        case = manifest["cases"][position]
        image = Image.open(
            args.corpus / "cases" / case["case_id"] / "source.png"
        ).convert("RGB")
        if args.prompt_mode == "reasoning":
            prompt = CHARTQA_INSTRUCTIONS.format(question=case["query"])
        else:
            prompt = (
                f"{case['query']}\nInspect the chart and respond with only: "
                "Final Answer: <answer>"
            )
        chat_prompt = apply_chat_template(
            processor,
            model.config,
            prompt,
            num_images=1,
        )
        prepared = prepare_inputs(
            processor,
            images=[image],
            prompts=chat_prompt,
            add_special_tokens=False,
        )

        mx.set_wired_limit(0)
        baseline_features = encode(prepared["pixel_values"])
        mx.eval(baseline_features)
        mx.synchronize()
        mx.set_wired_limit(WIRED_LIMIT)
        candidate_features = encode(prepared["pixel_values"])
        mx.eval(candidate_features)
        mx.synchronize()
        baseline_bits = np.array(baseline_features.view(mx.uint16), copy=True)
        candidate_bits = np.array(candidate_features.view(mx.uint16), copy=True)
        feature_identical = bool(np.array_equal(baseline_bits, candidate_bits))

        arms = (
            ("baseline", baseline_features),
            ("candidate", candidate_features),
        )
        if position % 2:
            arms = tuple(reversed(arms))
        generations = {}
        for name, features in arms:
            generations[name] = generate_with_features(
                model,
                processor,
                chat_prompt,
                prepared,
                features,
                args.max_tokens,
            )
        baseline_scores = chartqa_scores(generations["baseline"], case["targets"])
        candidate_scores = chartqa_scores(generations["candidate"], case["targets"])
        baseline_correct = bool(baseline_scores["relaxed_accuracy"])
        candidate_correct = bool(candidate_scores["relaxed_accuracy"])
        exact_agreement = generations["baseline"] == generations["candidate"]
        if exact_agreement:
            disagreement_class = "agreement"
        elif baseline_correct and not candidate_correct:
            disagreement_class = "baseline_only"
        elif candidate_correct and not baseline_correct:
            disagreement_class = "candidate_only"
        elif baseline_correct and candidate_correct:
            disagreement_class = "both_correct_differently"
        else:
            disagreement_class = "both_wrong_differently"
        record = {
            "position": position,
            "dataset_index": case["dataset_index"],
            "case_id": case["case_id"],
            "query": case["query"],
            "targets": case["targets"],
            "feature_bit_identical": feature_identical,
            "baseline_generation": generations["baseline"],
            "candidate_generation": generations["candidate"],
            "exact_string_agreement": exact_agreement,
            "disagreement_class": disagreement_class,
            "baseline_scores": baseline_scores,
            "candidate_scores": candidate_scores,
        }
        records.append(record)
        with raw_path.open("a") as output:
            output.write(json.dumps(record) + "\n")
        print(
            f"{position + 1}/{args.limit} agreement={int(exact_agreement)} "
            f"features={int(feature_identical)} class={disagreement_class}",
            flush=True,
        )

    summary = summarize(records)
    report = {
        "metadata": {
            "model": MODEL_ID,
            "candidate": "2 GiB MLX wired limit",
            "decoder": "local MLX decoder shared by both arms",
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "prompt_mode": args.prompt_mode,
            "sample_source": str(args.corpus / "manifest.json"),
            "note": "Absolute accuracy is not comparable to H200/vLLM results; paired differential is the gate.",
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
        "summary": summary,
        "disagreements": [
            record for record in records if not record["exact_string_agreement"]
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if summary["feature_bit_mismatch_samples"] or not summary["gate_b1"] or not summary["gate_b2"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
