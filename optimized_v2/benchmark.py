import argparse
import json
import statistics
from pathlib import Path

from PIL import Image

from optimized.client import StreamingImageClient

from .client import BinaryStreamingImageClient


def run(client, image: Image.Image, question: str) -> tuple[str, dict]:
    text = []
    done = None
    for event in client.stream(image, question):
        if event["type"] == "token":
            text.append(event["text"])
        else:
            done = event
    if done is None:
        raise RuntimeError("Client did not produce a completion event")
    return "".join(text), done


def summarize(results: list[dict]) -> dict:
    fields = (
        "client_preprocess_ms",
        "client_encode_ms",
        "client_serialize_ms",
        "request_bytes",
        "remote_ttft_ms",
        "pipeline_ttft_ms",
        "remote_e2e_ms",
        "pipeline_e2e_ms",
    )
    return {
        field: statistics.median(result[field] for result in results)
        for field in fields
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--legacy-server", default="http://127.0.0.1:8001")
    parser.add_argument("--binary-server", default="http://127.0.0.1:8002")
    parser.add_argument("--model", default="gemma-4-12b-optimized")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", default="What is shown in this image?")
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if args.rounds < 1:
        raise ValueError("rounds must be positive")

    image = Image.open(args.image).convert("RGB")
    clients = {
        "base64": StreamingImageClient(
            args.artifact, args.legacy_server, args.model
        ),
        "binary": BinaryStreamingImageClient(
            args.artifact, args.binary_server, args.model
        ),
    }
    for client in clients.values():
        run(client, image, args.question)

    results = {"base64": [], "binary": []}
    answers = {"base64": [], "binary": []}
    for round_index in range(args.rounds):
        order = ("base64", "binary") if round_index % 2 == 0 else ("binary", "base64")
        for name in order:
            answer, metrics = run(clients[name], image, args.question)
            answers[name].append(answer)
            results[name].append(metrics)

    outputs_equal = all(
        base64_answer == binary_answer
        for base64_answer, binary_answer in zip(
            answers["base64"], answers["binary"], strict=True
        )
    )
    report = {
        "rounds": args.rounds,
        "outputs_equal": outputs_equal,
        "base64": summarize(results["base64"]),
        "binary": summarize(results["binary"]),
    }
    report["improvement"] = {
        "request_bytes_reduction_percent": 100
        * (1 - report["binary"]["request_bytes"] / report["base64"]["request_bytes"]),
        "packaging_ms_reduction_percent": 100
        * (
            1
            - report["binary"]["client_serialize_ms"]
            / report["base64"]["client_serialize_ms"]
        ),
        "remote_ttft_ms_delta": report["binary"]["remote_ttft_ms"]
        - report["base64"]["remote_ttft_ms"],
        "pipeline_e2e_ms_delta": report["binary"]["pipeline_e2e_ms"]
        - report["base64"]["pipeline_e2e_ms"],
    }
    print(json.dumps(report, indent=2))
    if not outputs_equal:
        raise SystemExit("Binary and base64 outputs differ")


if __name__ == "__main__":
    main()
