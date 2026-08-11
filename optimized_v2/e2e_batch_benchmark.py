import argparse
import base64
import concurrent.futures
import http.client
import io
import json
import math
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

from .mlx_client import MLXBinaryStreamingImageClient
from .protocol import CONTENT_TYPE, encode_raw_request


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "p50": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "minimum": min(values),
        "maximum": max(values),
    }


def read_stream(response, request_started: float) -> dict:
    first_token_at = None
    try:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            choices = event.get("choices", [])
            text = choices[0].get("delta", {}).get("content") if choices else None
            if text:
                first_token_at = first_token_at or time.perf_counter()
    finally:
        response.read()
    finished = time.perf_counter()
    first_token_at = first_token_at or finished
    return {
        "remote_ttft_ms": (first_token_at - request_started) * 1000,
        "remote_e2e_ms": (finished - request_started) * 1000,
    }


def send_split(
    host: str,
    port: int,
    path: str,
    payload: bytes,
    batch_started: float,
) -> dict:
    request_started = time.perf_counter()
    connection = http.client.HTTPConnection(host, port, timeout=300)
    try:
        connection.request(
            "POST",
            path,
            body=payload,
            headers={"Content-Type": CONTENT_TYPE, "Accept": "text/event-stream"},
        )
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(
                f"Gateway returned HTTP {response.status}: "
                f"{response.read().decode(errors='replace')}"
            )
        metrics = read_stream(response, request_started)
        for header, name in (
            ("X-Gateway-TTFT-Ms", "gateway_ttft_ms"),
            ("X-Gateway-Prepare-Ms", "gateway_prepare_ms"),
            ("X-vLLM-TTFT-Ms", "vllm_ttft_ms"),
        ):
            value = response.getheader(header)
            metrics[name] = float(value) if value is not None else None
        metrics["pipeline_ttft_ms"] = (
            request_started - batch_started
        ) * 1000 + metrics["remote_ttft_ms"]
        metrics["pipeline_e2e_ms"] = (
            request_started - batch_started
        ) * 1000 + metrics["remote_e2e_ms"]
        return metrics
    finally:
        connection.close()


def send_full(
    host: str,
    port: int,
    path: str,
    payload: bytes,
    batch_started: float,
) -> dict:
    request_started = time.perf_counter()
    connection = http.client.HTTPConnection(host, port, timeout=300)
    try:
        connection.request(
            "POST",
            path,
            body=payload,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(
                f"vLLM returned HTTP {response.status}: "
                f"{response.read().decode(errors='replace')}"
            )
        metrics = read_stream(response, request_started)
        metrics["pipeline_ttft_ms"] = (
            request_started - batch_started
        ) * 1000 + metrics["remote_ttft_ms"]
        metrics["pipeline_e2e_ms"] = (
            request_started - batch_started
        ) * 1000 + metrics["remote_e2e_ms"]
        return metrics
    finally:
        connection.close()


def parse_server(server: str) -> tuple[str, int, str]:
    parsed = urlsplit(server)
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError("Server must be an HTTP URL")
    return (
        parsed.hostname,
        parsed.port or 80,
        parsed.path.rstrip("/") + "/v1/chat/completions",
    )


def full_payload(image: Image.Image, model: str, question: str) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    return json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": question},
                    ],
                }
            ],
            "max_tokens": 1,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()


def run_split_batch(
    client,
    image,
    question,
    batch_size,
    host,
    port,
    path,
) -> dict:
    batch_started = time.perf_counter()
    prompt = apply_chat_template(
        client.processor,
        client.config,
        question,
        num_images=1,
    )
    inputs = prepare_inputs(
        client.processor,
        images=[image],
        prompts=prompt,
        add_special_tokens=False,
    )
    pixels = inputs["pixel_values"]
    if batch_size > 1:
        pixels = mx.repeat(pixels, batch_size, axis=0)
    preprocessed = time.perf_counter()
    features = client.encode_vision(pixels)
    mx.eval(features)
    mx.synchronize()
    encoded = time.perf_counter()
    if features.shape[0] != batch_size or features.shape[2] not in (768, 2560):
        raise RuntimeError(f"Unexpected batched features shape {features.shape}")

    payloads = []
    for index in range(batch_size):
        feature = features[index]
        tensor = np.array(feature.view(mx.uint16), copy=True).tobytes(order="C")
        payloads.append(
            encode_raw_request(
                tensor,
                tuple(feature.shape),
                f"{question} request={index}",
                client.model_name,
                max_tokens=1,
            )
        )
    serialized = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        records = list(
            executor.map(
                lambda payload: send_split(
                    host, port, path, payload, batch_started
                ),
                payloads,
            )
        )
    finished = time.perf_counter()
    return {
        "batch_size": batch_size,
        "batch_preprocess_ms": (preprocessed - batch_started) * 1000,
        "batch_encode_ms": (encoded - preprocessed) * 1000,
        "batch_serialize_ms": (serialized - encoded) * 1000,
        "batch_wall_ms": (finished - batch_started) * 1000,
        "images_per_second": batch_size / (finished - batch_started),
        "request_bytes": [len(payload) for payload in payloads],
        "requests": records,
    }


def run_full_batch(image, model, question, batch_size, host, port, path) -> dict:
    batch_started = time.perf_counter()
    payloads = [
        full_payload(image, model, f"{question} request={index}")
        for index in range(batch_size)
    ]
    serialized = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        records = list(
            executor.map(
                lambda payload: send_full(host, port, path, payload, batch_started),
                payloads,
            )
        )
    finished = time.perf_counter()
    return {
        "batch_size": batch_size,
        "batch_serialize_ms": (serialized - batch_started) * 1000,
        "batch_wall_ms": (finished - batch_started) * 1000,
        "images_per_second": batch_size / (finished - batch_started),
        "request_bytes": [len(payload) for payload in payloads],
        "requests": records,
    }


def run_split_pipeline(
    client,
    image,
    question,
    request_count,
    host,
    port,
    path,
    max_in_flight,
) -> dict:
    batch_started = time.perf_counter()
    event_groups = client.complete_many(
        [
            (image, f"{question} request={index}")
            for index in range(request_count)
        ],
        max_tokens=1,
        max_in_flight=max_in_flight,
    )
    records = [
        next(event for event in events if event["type"] == "done")
        for events in event_groups
    ]
    finished = time.perf_counter()
    return {
        "batch_size": request_count,
        "batch_preprocess_ms": sum(
            record["client_preprocess_ms"] for record in records
        ),
        "batch_encode_ms": sum(record["client_encode_ms"] for record in records),
        "batch_serialize_ms": sum(
            record["client_serialize_ms"] for record in records
        ),
        "batch_wall_ms": (finished - batch_started) * 1000,
        "images_per_second": request_count / (finished - batch_started),
        "request_bytes": [record["request_bytes"] for record in records],
        "requests": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("full", "split", "split-pipeline"))
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", default="mlx-community/gemma-4-e4b-it-4bit")
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--project-on-server", action="store_true")
    parser.add_argument("--strict-qkv", action="store_true")
    parser.add_argument("--max-in-flight", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    host, port, path = parse_server(args.server)
    image = ImageOps.fit(
        Image.open(args.image).convert("RGB"),
        (854, 480),
        method=Image.Resampling.LANCZOS,
    )
    client = (
        MLXBinaryStreamingImageClient(
            args.checkpoint,
            args.server,
            args.model,
            project_on_server=args.project_on_server,
            use_qkv_epilogue=not args.strict_qkv,
        )
        if args.mode.startswith("split")
        else None
    )

    def run_batch(batch_size: int, warmup: bool = False) -> dict:
        question = f"Describe this image. nonce={uuid.uuid4()} warmup={warmup}"
        if args.mode == "split":
            return run_split_batch(
                client, image, question, batch_size, host, port, path
            )
        if args.mode == "split-pipeline":
            return run_split_pipeline(
                client,
                image,
                question,
                batch_size,
                host,
                port,
                path,
                args.max_in_flight,
            )
        return run_full_batch(image, args.model, question, batch_size, host, port, path)

    results = {}
    for batch_size in batch_sizes:
        run_batch(batch_size, warmup=True)
        rounds = []
        for index in range(args.rounds):
            record = run_batch(batch_size)
            rounds.append(record)
            print(
                f"{args.mode} B{batch_size} {index + 1}/{args.rounds} "
                f"wall={record['batch_wall_ms']:.3f}ms "
                f"throughput={record['images_per_second']:.3f}/s",
                flush=True,
            )
        request_records = [
            request for record in rounds for request in record["requests"]
        ]
        results[str(batch_size)] = {
            "batch_size": batch_size,
            "rounds": args.rounds,
            "batch_wall_ms": summarize(
                [record["batch_wall_ms"] for record in rounds]
            ),
            "images_per_second": summarize(
                [record["images_per_second"] for record in rounds]
            ),
            "pipeline_ttft_ms": summarize(
                [record["pipeline_ttft_ms"] for record in request_records]
            ),
            "pipeline_e2e_ms": summarize(
                [record["pipeline_e2e_ms"] for record in request_records]
            ),
            "remote_ttft_ms": summarize(
                [record["remote_ttft_ms"] for record in request_records]
            ),
            "remote_e2e_ms": summarize(
                [record["remote_e2e_ms"] for record in request_records]
            ),
            "request_bytes": summarize(
                [value for record in rounds for value in record["request_bytes"]]
            ),
            "raw": rounds,
        }
        component_fields = ["batch_serialize_ms"]
        if args.mode.startswith("split"):
            component_fields.extend(("batch_preprocess_ms", "batch_encode_ms"))
        for field in component_fields:
            results[str(batch_size)][field] = summarize(
                [record[field] for record in rounds]
            )

    report = {
        "metadata": {
            "mode": args.mode,
            "server": args.server,
            "model": args.model,
            "checkpoint": args.checkpoint if args.mode.startswith("split") else None,
            "batch_sizes": batch_sizes,
            "rounds": args.rounds,
            "max_tokens": 1,
            "temperature": 0,
            "project_on_server": args.project_on_server,
            "qkv_epilogue": not args.strict_qkv,
            "max_in_flight": (
                args.max_in_flight if args.mode == "split-pipeline" else None
            ),
            "image_resolution": "854x480 JPEG quality 90 for full H200",
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
