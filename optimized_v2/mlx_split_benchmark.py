import argparse
import json
import math
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps

from .mlx_client import MLXBinaryStreamingImageClient
from .protocol import encode_raw_request


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


def run(client, image: Image.Image, question: str) -> tuple[str, dict]:
    text = []
    done = None
    for event in client.stream(image, question, max_tokens=1):
        if event["type"] == "token":
            text.append(event["text"])
        else:
            done = event
    if done is None:
        raise RuntimeError("Split request did not return completion metrics")
    return "".join(text), done


def run_local(client, image: Image.Image, question: str) -> dict:
    features, preprocess_ms, encode_ms = client.encode_image(image, question)
    started = time.perf_counter()
    tensor = np.array(features.view(mx.uint16), copy=True).tobytes(order="C")
    payload = encode_raw_request(
        tensor,
        tuple(features.shape),
        question,
        client.model_name,
        max_tokens=1,
    )
    return {
        "client_preprocess_ms": preprocess_ms,
        "client_encode_ms": encode_ms,
        "client_serialize_ms": (time.perf_counter() - started) * 1000,
        "request_bytes": len(payload),
        "tensor_bytes": len(tensor),
        "visual_tokens": features.shape[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="mlx-community/gemma-4-e4b-it-4bit",
    )
    parser.add_argument("--server")
    parser.add_argument("--model", default="gemma-4-e4b-optimized")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--project-on-server", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.local_only and args.server is None:
        parser.error("--server is required unless --local-only is set")

    source = Image.open(args.image).convert("RGB")
    image = ImageOps.fit(source, (854, 480), method=Image.Resampling.LANCZOS)
    client = MLXBinaryStreamingImageClient(
        args.checkpoint,
        args.server or "http://127.0.0.1:1",
        args.model,
        project_on_server=args.project_on_server,
    )
    warmup_question = f"Describe this image. warmup={uuid.uuid4()}"
    if args.local_only:
        run_local(client, image, warmup_question)
    else:
        run(client, image, warmup_question)

    records = []
    for index in range(args.rounds):
        question = f"Describe this image. nonce={uuid.uuid4()} rep={index}"
        if args.local_only:
            metrics = run_local(client, image, question)
            record = {"rep": index, **metrics}
        else:
            answer, metrics = run(client, image, question)
            record = {"rep": index, "answer": answer, **metrics}
        records.append(record)
        message = (
            f"split {index + 1}/{args.rounds} "
            f"encode={metrics['client_encode_ms']:.3f}ms"
        )
        if not args.local_only:
            message += f" ttft={metrics['pipeline_ttft_ms']:.3f}ms"
        print(message, flush=True)

    fields = (
        "client_preprocess_ms",
        "client_encode_ms",
        "client_serialize_ms",
        "request_bytes",
        "tensor_bytes",
        "visual_tokens",
        "remote_ttft_ms",
        "gateway_ttft_ms",
        "gateway_prepare_ms",
        "vllm_ttft_ms",
        "transport_ttft_ms",
        "pipeline_ttft_ms",
        "remote_e2e_ms",
        "pipeline_e2e_ms",
    )
    summary = {
        field: summarize([record[field] for record in records])
        for field in fields
        if all(field in record and record[field] is not None for record in records)
    }
    result = {
        "metadata": {
            "benchmark": (
                "mlx_e4b_split_local_480p"
                if args.local_only
                else "mlx_e4b_split_ttft_480p"
            ),
            "checkpoint": args.checkpoint,
            "model": args.model,
            "rounds": args.rounds,
            "image_resolution": "854x480",
            "max_tokens": 1,
            "temperature": 0,
            "unique_nonce": True,
            "sequential": True,
            "local_only": args.local_only,
            "project_on_server": args.project_on_server,
            "client_load_ms": client.load_ms,
            "client_active_memory_gb": client.active_memory_gb,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
        "summary": summary,
        "raw": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
