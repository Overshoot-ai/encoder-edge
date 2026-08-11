"""Compare fresh and persistent split HTTP connections with one encoded tensor."""

import argparse
import http.client
import importlib.metadata
import json
import time
import uuid
from pathlib import Path

from PIL import Image

from .mlx_client import MLXBinaryStreamingImageClient
from .mlx_vision_quantization_ab import summarize
from .protocol import CONTENT_TYPE


def send(connection, host, port, path, payload):
    if connection is None:
        connection = http.client.HTTPConnection(host, port, timeout=300)
    started = time.perf_counter()
    connection.request(
        "POST",
        path,
        body=payload,
        headers={"Content-Type": CONTENT_TYPE, "Accept": "text/event-stream"},
    )
    response = connection.getresponse()
    if response.status != 200:
        raise RuntimeError(f"Gateway returned {response.status}: {response.read()!r}")
    first_token = None
    for line in response:
        if not line.startswith(b"data: "):
            continue
        data = line[6:].strip()
        if data == b"[DONE]":
            break
        event = json.loads(data)
        choices = event.get("choices", [])
        text = choices[0].get("delta", {}).get("content") if choices else None
        if text and first_token is None:
            first_token = time.perf_counter()
    response.read()
    finished = time.perf_counter()
    return connection, {
        "remote_ttft_ms": ((first_token or finished) - started) * 1000,
        "remote_e2e_ms": (finished - started) * 1000,
        "gateway_ttft_ms": float(response.getheader("X-Gateway-TTFT-Ms")),
        "gateway_prepare_ms": float(response.getheader("X-Gateway-Prepare-Ms")),
        "vllm_ttft_ms": float(response.getheader("X-vLLM-TTFT-Ms")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    client = MLXBinaryStreamingImageClient(
        "mlx-community/gemma-4-e4b-it-4bit",
        args.server,
        args.model,
        project_on_server=True,
    )
    payload, prepared = client._prepare_request(
        Image.open(args.image),
        f"Describe this image. nonce={uuid.uuid4()}",
        1,
        None,
    )
    persistent = None
    records = {"fresh": [], "persistent": []}
    for name in records:
        connection = persistent if name == "persistent" else None
        connection, _ = send(
            connection, client.host, client.port, client.path, payload
        )
        if name == "persistent":
            persistent = connection
        else:
            connection.close()

    for round_index in range(args.rounds):
        order = ("fresh", "persistent") if round_index % 2 == 0 else ("persistent", "fresh")
        for name in order:
            connection = persistent if name == "persistent" else None
            connection, metrics = send(
                connection, client.host, client.port, client.path, payload
            )
            records[name].append(metrics)
            if name == "persistent":
                persistent = connection
            else:
                connection.close()
    if persistent is not None:
        persistent.close()

    result = {
        "metadata": {
            "mlx_version": importlib.metadata.version("mlx"),
            "mlx_vlm_version": importlib.metadata.version("mlx-vlm"),
            "rounds": args.rounds,
            "request_bytes": len(payload),
            "visual_tokens": prepared["visual_tokens"],
            "interleaved": True,
        },
        "results": {
            name: {
                metric: {
                    **summarize([record[metric] for record in values]),
                    "raw": [record[metric] for record in values],
                }
                for metric in values[0]
            }
            for name, values in records.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
