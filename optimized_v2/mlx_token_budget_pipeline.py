import argparse
import json
import math
import statistics
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

from .mlx_client import MLXBinaryStreamingImageClient


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "p50": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "min": min(values),
        "max": max(values),
    }


def run(client, image, budget: int, index: int) -> dict:
    question = f"Describe this image. budget={budget} nonce={uuid.uuid4()} rep={index}"
    done = None
    for event in client.stream(
        image,
        question,
        max_tokens=1,
        max_soft_tokens=budget,
    ):
        if event["type"] == "done":
            done = event
    if done is None:
        raise RuntimeError("Split request did not return metrics")
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", default="gemma-4-e4b-optimized")
    parser.add_argument(
        "--checkpoint",
        default="mlx-community/gemma-4-e4b-it-4bit",
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--budgets", default="273,203,144,69")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    budgets = [int(value) for value in args.budgets.split(",")]

    source = Image.open(args.image).convert("RGB")
    image = ImageOps.fit(source, (854, 480), method=Image.Resampling.LANCZOS)
    client = MLXBinaryStreamingImageClient(
        args.checkpoint,
        args.server,
        args.model,
    )
    results = []
    fields = (
        "client_preprocess_ms",
        "client_encode_ms",
        "client_serialize_ms",
        "request_bytes",
        "tensor_bytes",
        "remote_ttft_ms",
        "gateway_ttft_ms",
        "gateway_prepare_ms",
        "vllm_ttft_ms",
        "transport_ttft_ms",
        "pipeline_ttft_ms",
    )
    for budget in budgets:
        run(client, image, budget, -1)
        records = [run(client, image, budget, index) for index in range(args.rounds)]
        result = {
            "processor_soft_token_budget": budget,
            "actual_visual_tokens": records[0]["visual_tokens"],
            "rounds": args.rounds,
            "summary": {
                field: summarize([record[field] for record in records])
                for field in fields
                if all(record[field] is not None for record in records)
            },
            "raw": records,
        }
        results.append(result)
        print(json.dumps(result["summary"], indent=2), flush=True)

    report = {
        "metadata": {
            "model": args.model,
            "checkpoint": args.checkpoint,
            "server": args.server,
            "budgets": budgets,
            "rounds": args.rounds,
            "max_tokens": 1,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
