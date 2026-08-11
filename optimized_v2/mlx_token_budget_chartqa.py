import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from .mlx_client import MLXBinaryStreamingImageClient
from .overshoot_eval import CHARTQA_INSTRUCTIONS, chartqa_scores, percentile


def run(client, image, prompt: str, budget: int) -> tuple[str, dict]:
    fragments = []
    done = None
    for event in client.stream(
        image,
        prompt,
        max_tokens=512,
        max_soft_tokens=budget,
    ):
        if event["type"] == "token":
            fragments.append(event["text"])
        else:
            done = event
    if done is None:
        raise RuntimeError("Split request did not return metrics")
    return "".join(fragments), done


def summarize(records: list[dict]) -> dict:
    return {
        "samples": len(records),
        "exact_match": statistics.mean(record["exact_match"] for record in records),
        "relaxed_accuracy": statistics.mean(
            record["relaxed_accuracy"] for record in records
        ),
        "anywhere_accuracy": statistics.mean(
            record["anywhere_accuracy"] for record in records
        ),
        "actual_visual_tokens_mean": statistics.mean(
            record["visual_tokens"] for record in records
        ),
        "client_encode_ms_p50": percentile(
            [record["client_encode_ms"] for record in records],
            0.5,
        ),
        "client_encode_ms_p90": percentile(
            [record["client_encode_ms"] for record in records],
            0.9,
        ),
        "pipeline_ttft_ms_p50": percentile(
            [record["pipeline_ttft_ms"] for record in records],
            0.5,
        ),
        "pipeline_ttft_ms_p90": percentile(
            [record["pipeline_ttft_ms"] for record in records],
            0.9,
        ),
        "pipeline_e2e_ms_p50": percentile(
            [record["pipeline_e2e_ms"] for record in records],
            0.5,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", default="gemma-4-e4b-optimized")
    parser.add_argument(
        "--checkpoint",
        default="mlx-community/gemma-4-e4b-it-4bit",
    )
    parser.add_argument("--budgets", default="273,203,144,69")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--project-on-server", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    budgets = [int(value) for value in args.budgets.split(",")]

    from datasets import load_dataset

    dataset = load_dataset("HuggingFaceM4/ChartQA", split="test")
    client = MLXBinaryStreamingImageClient(
        args.checkpoint,
        args.server,
        args.model,
        project_on_server=args.project_on_server,
    )
    summaries = []
    args.output.mkdir(parents=True, exist_ok=True)
    for budget in budgets:
        budget_output = args.output / f"budget-{budget}"
        budget_output.mkdir(parents=True, exist_ok=True)
        raw_path = budget_output / "raw_results.jsonl"
        records = (
            [json.loads(line) for line in raw_path.read_text().splitlines()]
            if raw_path.exists()
            else []
        )
        for index in range(len(records), min(args.limit, len(dataset))):
            sample = dataset[index]
            image = sample["image"].convert("RGB")
            prompt = CHARTQA_INSTRUCTIONS.format(question=sample["query"])
            for attempt in range(3):
                try:
                    answer, metrics = run(client, image, prompt, budget)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2)
            scores = chartqa_scores(answer, sample["label"])
            record = {
                "index": index,
                "processor_soft_token_budget": budget,
                "query": sample["query"],
                "targets": sample["label"],
                "generation": answer,
                **scores,
                **metrics,
            }
            records.append(record)
            with raw_path.open("a") as output:
                output.write(json.dumps(record) + "\n")
            print(
                f"budget={budget} {index + 1}/{args.limit} "
                f"tokens={metrics['visual_tokens']} "
                f"relaxed={scores['relaxed_accuracy']:.0f}",
                flush=True,
            )
        summary = {
            "processor_soft_token_budget": budget,
            **summarize(records),
        }
        summaries.append(summary)
        (budget_output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )

    report = {
        "metadata": {
            "dataset": "HuggingFaceM4/ChartQA",
            "split": "test",
            "model": args.model,
            "checkpoint": args.checkpoint,
            "server": args.server,
            "budgets": budgets,
            "limit": args.limit,
            "max_tokens": 512,
            "project_on_server": args.project_on_server,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
        "summaries": summaries,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
