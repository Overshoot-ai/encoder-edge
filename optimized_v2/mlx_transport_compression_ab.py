"""Compare raw and lossless-zstd feature transport on persistent connections."""

import argparse
import http.client
import importlib.metadata
import json
import time
import uuid
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from .mlx_client import MLXBinaryStreamingImageClient
from .mlx_transport_connection_ab import send
from .mlx_vision_quantization_ab import summarize
from .protocol import encode_raw_request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 5:
        raise ValueError("Compression A/B requires at least five rounds")

    client = MLXBinaryStreamingImageClient(
        "mlx-community/gemma-4-e4b-it-4bit",
        args.server,
        args.model,
        project_on_server=True,
    )
    question = f"Describe this image. nonce={uuid.uuid4()}"
    features, _, _ = client.encode_image(Image.open(args.image), question)
    tensor_bytes = np.array(features.view(mx.uint16), copy=True).tobytes(order="C")
    payloads = {}
    serialization_ms = {}
    for name, compression in (("raw", None), ("zstd", "zstd")):
        started = time.perf_counter()
        payloads[name] = encode_raw_request(
            tensor_bytes,
            tuple(features.shape),
            question,
            client.model_name,
            max_tokens=1,
            compression=compression,
        )
        serialization_ms[name] = (time.perf_counter() - started) * 1000

    connections = {
        name: http.client.HTTPConnection(client.host, client.port, timeout=300)
        for name in payloads
    }
    records = {name: [] for name in payloads}
    try:
        for name in records:
            connections[name], _ = send(
                connections[name],
                client.host,
                client.port,
                client.path,
                payloads[name],
            )
        for round_index in range(args.rounds):
            order = ("raw", "zstd") if round_index % 2 == 0 else ("zstd", "raw")
            for name in order:
                connections[name], metrics = send(
                    connections[name],
                    client.host,
                    client.port,
                    client.path,
                    payloads[name],
                )
                records[name].append(metrics)
    finally:
        for connection in connections.values():
            connection.close()

    result = {
        "metadata": {
            "mlx_version": importlib.metadata.version("mlx"),
            "mlx_vlm_version": importlib.metadata.version("mlx-vlm"),
            "zstandard_version": importlib.metadata.version("zstandard"),
            "rounds": args.rounds,
            "raw_tensor_bytes": len(tensor_bytes),
            "request_bytes": {name: len(payload) for name, payload in payloads.items()},
            "serialization_ms": serialization_ms,
            "visual_tokens": features.shape[0],
            "interleaved": True,
            "persistent_connection_per_arm": True,
            "bit_identical_transport": True,
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
