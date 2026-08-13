from __future__ import annotations

import argparse
import json
import os
import sys
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
