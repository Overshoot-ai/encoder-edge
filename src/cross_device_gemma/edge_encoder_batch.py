from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from .chat_client import ChatClient
from .chat_protocol import ENCODER_MODEL, encode_chat_request
from .edge_encoder import EdgeEncoder


def load_requests(path: Path, default_max_tokens: int) -> list[dict]:
    requests = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on line {line_number}") from error
        if not isinstance(item, dict):
            raise ValueError(f"Line {line_number} must be a JSON object")
        image = item.get("image")
        prompt = item.get("prompt")
        max_tokens = item.get("max_tokens", default_max_tokens)
        if not isinstance(image, str) or not image:
            raise ValueError(f"Line {line_number} has no image")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Line {line_number} has no prompt")
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or not 1 <= max_tokens <= 1024
        ):
            raise ValueError(f"Line {line_number} has invalid max_tokens")
        image_path = Path(image).expanduser()
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        requests.append(
            {
                "id": item.get("id", len(requests)),
                "image": image_path.resolve(),
                "prompt": prompt,
                "max_tokens": max_tokens,
            }
        )
    if not requests:
        raise ValueError("Batch input contains no requests")
    return requests


def _complete(client, payload: bytes, item_started: float) -> tuple[str, dict]:
    remote_worker_started = time.perf_counter()
    text = []
    done = None
    for event in client.stream(payload):
        if event["type"] == "token":
            text.append(event["text"])
        else:
            done = event
    finished = time.perf_counter()
    metrics = dict(done or {})
    metrics.pop("type", None)
    metrics["pipeline_ttft_ms"] = (
        (remote_worker_started - item_started) * 1000
        + metrics.get("remote_ttft_ms", 0)
    )
    metrics["pipeline_e2e_ms"] = (finished - item_started) * 1000
    return "".join(text), metrics


def run_batch(args, token: str | None, api_key: str | None, progress=None) -> dict:
    if args.max_in_flight < 1:
        raise ValueError("max_in_flight must be positive")
    requests = load_requests(args.input, args.max_tokens)
    load_started = time.perf_counter()
    encoder = EdgeEncoder(
        ENCODER_MODEL,
        cache_dir=args.cache_dir,
        token=token,
        progress=progress,
        split_point="vision_pre_projector",
        require_optimized=True,
    )
    load_ms = (time.perf_counter() - load_started) * 1000
    client = ChatClient(
        args.server,
        api_key=api_key,
        max_connections=args.max_in_flight,
    )
    batch_started = time.perf_counter()
    futures = []
    executor = ThreadPoolExecutor(max_workers=args.max_in_flight)
    try:
        for index, request in enumerate(requests, 1):
            item_started = time.perf_counter()
            if progress:
                progress(f"Encoding image {index}/{len(requests)}: {request['image']}")
            with Image.open(request["image"]) as image:
                features, local_metrics = encoder.encode(image)
            payload = encode_chat_request(
                features,
                request["prompt"],
                max_tokens=request["max_tokens"],
                compression="zstd" if args.zstd else None,
            )
            future = executor.submit(_complete, client, payload, item_started)
            futures.append((request, local_metrics, len(payload), future))

        results = []
        for request, local_metrics, request_bytes, future in futures:
            response, remote_metrics = future.result()
            results.append(
                {
                    "id": request["id"],
                    "image": str(request["image"]),
                    "prompt": request["prompt"],
                    "response": response,
                    "metrics": {
                        **local_metrics,
                        **remote_metrics,
                        "request_bytes": request_bytes,
                    },
                }
            )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        client.close()

    batch_wall_ms = (time.perf_counter() - batch_started) * 1000
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(result, separators=(",", ":")) + "\n" for result in results)
    )
    return {
        "requests": len(results),
        "model_load_ms": load_ms,
        "batch_wall_ms": batch_wall_ms,
        "images_per_second": len(results) / (batch_wall_ms / 1000),
        "output": str(args.output),
    }
