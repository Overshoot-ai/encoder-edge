from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image

from .edge_encoder import EdgeEncoder, UnsupportedModelError, tensor_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="edge-encoder",
        description="Run the vision component of a Hugging Face multimodal model locally",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    encode = subparsers.add_parser("encode")
    encode.add_argument("--model", required=True)
    encode.add_argument("--revision")
    encode.add_argument("--input", type=Path, required=True)
    encode.add_argument("--output", type=Path, required=True)
    encode.add_argument("--metadata", type=Path)
    encode.add_argument("--cache-dir", type=Path)
    encode.add_argument("--allow-full-download", action="store_true")
    encode.add_argument(
        "--dtype", choices=("auto", "float32", "bfloat16"), default="auto"
    )
    benchmark = subparsers.add_parser(
        "benchmark", help="Compare optimized MLX with the PyTorch compatibility path"
    )
    benchmark.add_argument("--model", required=True)
    benchmark.add_argument("--revision")
    benchmark.add_argument("--input", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--cache-dir", type=Path)
    benchmark.add_argument("--warmups", type=int, default=1)
    benchmark.add_argument("--rounds", type=int, default=5)
    chat = subparsers.add_parser(
        "chat", help="Encode an image locally and stream a remote chat completion"
    )
    chat.add_argument("--server", required=True)
    chat.add_argument("--image", type=Path, required=True)
    chat.add_argument("--prompt", required=True)
    chat.add_argument("--max-tokens", type=int, default=128)
    chat.add_argument("--cache-dir", type=Path)
    chat.add_argument("--zstd", action="store_true")
    gateway = subparsers.add_parser(
        "gateway", help="Translate Edge Encoder requests to a projector-aware vLLM server"
    )
    gateway.add_argument("--upstream", default="http://127.0.0.1:8001")
    gateway.add_argument("--host", default="127.0.0.1")
    gateway.add_argument("--port", type=int, default=8002)
    batch = subparsers.add_parser(
        "batch", help="Pipeline independent image chat requests from JSONL"
    )
    batch.add_argument("--server", required=True)
    batch.add_argument("--input", type=Path, required=True)
    batch.add_argument("--output", type=Path, required=True)
    batch.add_argument("--max-tokens", type=int, default=128)
    batch.add_argument("--max-in-flight", type=int, default=4)
    batch.add_argument("--cache-dir", type=Path)
    batch.add_argument("--zstd", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "gateway":
        from .chat_gateway import serve_gateway

        serve_gateway(args.upstream, args.host, args.port)
        return
    if args.command == "chat":
        run_chat(args)
        return
    if args.command == "batch":
        from .edge_encoder_batch import run_batch

        try:
            summary = run_batch(
                args,
                token=os.environ.get("HF_TOKEN"),
                api_key=os.environ.get("EDGE_ENCODER_API_KEY"),
                progress=lambda message: print(message, file=sys.stderr),
            )
        except (UnsupportedModelError, RuntimeError, OSError, ValueError) as error:
            print(f"edge-encoder: error: {error}", file=sys.stderr)
            raise SystemExit(2) from error
        print(json.dumps(summary, indent=2))
        return
    if args.command == "benchmark":
        from .edge_encoder_benchmark import benchmark, format_report

        try:
            report = benchmark(args)
        except (RuntimeError, OSError, ValueError) as error:
            print(f"edge-encoder: error: {error}", file=sys.stderr)
            raise SystemExit(2) from error
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(format_report(report))
        print(f"\nFull report: {args.output}")
        return
    try:
        encoder = EdgeEncoder(
            args.model,
            revision=args.revision,
            cache_dir=args.cache_dir,
            dtype=args.dtype,
            token=os.environ.get("HF_TOKEN"),
            progress=lambda message: print(message, file=sys.stderr),
            allow_full_download=args.allow_full_download,
        )
        with Image.open(args.input) as image:
            features, metrics = encoder.encode(image)
    except (UnsupportedModelError, RuntimeError, OSError) as error:
        print(f"edge-encoder: error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    payload, dtype_name = tensor_bytes(features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    metadata = encoder.metadata(features, dtype_name, metrics)
    metadata_path = args.metadata or args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def run_chat(args: argparse.Namespace) -> None:
    from .chat_client import stream_chat
    from .chat_protocol import ENCODER_MODEL, encode_chat_request

    total_started = time.perf_counter()
    try:
        encoder = EdgeEncoder(
            ENCODER_MODEL,
            cache_dir=args.cache_dir,
            token=os.environ.get("HF_TOKEN"),
            progress=lambda message: print(message, file=sys.stderr),
            split_point="vision_pre_projector",
            require_optimized=True,
        )
        with Image.open(args.image) as image:
            features, local_metrics = encoder.encode(image)
        payload = encode_chat_request(
            features,
            args.prompt,
            max_tokens=args.max_tokens,
            compression="zstd" if args.zstd else None,
        )
        done = None
        for event in stream_chat(
            args.server,
            payload,
            api_key=os.environ.get("EDGE_ENCODER_API_KEY"),
        ):
            if event["type"] == "token":
                print(event["text"], end="", flush=True)
            else:
                done = event
        print()
        metrics = {
            **local_metrics,
            **(done or {}),
            "request_bytes": len(payload),
            "visual_tokens": features.shape[0],
            "hidden_size": features.shape[1],
            "pipeline_e2e_ms": (time.perf_counter() - total_started) * 1000,
        }
        metrics.pop("type", None)
        print(json.dumps(metrics), file=sys.stderr)
    except (UnsupportedModelError, RuntimeError, OSError, ValueError) as error:
        print(f"edge-encoder: error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
