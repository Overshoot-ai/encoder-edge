from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .edge_encoder import EdgeEncoder, tensor_bytes


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values: list[float]) -> dict:
    return {
        "p50_ms": statistics.median(values),
        "p90_ms": percentile(values, 0.9),
        "minimum_ms": min(values),
        "maximum_ms": max(values),
        "measurements_ms": values,
    }


def compare_bfloat16(left: Path, right: Path) -> dict:
    left_words = np.fromfile(left, dtype="<u2")
    right_words = np.fromfile(right, dtype="<u2")
    if left_words.shape != right_words.shape:
        raise RuntimeError(
            f"Benchmark outputs have different element counts: "
            f"{left_words.size} and {right_words.size}"
        )
    left_float = (left_words.astype(np.uint32) << 16).view(np.float32)
    right_float = (right_words.astype(np.uint32) << 16).view(np.float32)
    difference = left_float.astype(np.float64) - right_float.astype(np.float64)
    denominator = np.linalg.norm(left_float.astype(np.float64))
    return {
        "elements": int(left_words.size),
        "bit_identical_fraction": float(np.mean(left_words == right_words)),
        "relative_l2_difference": float(
            np.linalg.norm(difference) / denominator
        ),
        "maximum_absolute_difference": float(
            np.max(np.abs(difference))
        ),
        "finite": bool(np.isfinite(left_float).all() and np.isfinite(right_float).all()),
    }


def run_worker(args: argparse.Namespace) -> None:
    optimize = args.arm == "optimized"
    encoder = EdgeEncoder(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        token=args.token,
        progress=None,
        optimize=optimize,
    )
    expected_backend = "mlx" if optimize else "pytorch"
    if encoder.backend != expected_backend:
        raise RuntimeError(
            f"{args.arm} benchmark requires {expected_backend}, got {encoder.backend}"
        )

    with Image.open(args.input) as source:
        image = source.convert("RGB")
    for _ in range(args.warmups):
        encoder.encode(image)

    preprocess_measurements = []
    encode_measurements = []
    features = None
    for _ in range(args.rounds):
        features, metrics = encoder.encode(image)
        if encoder.backend != expected_backend:
            raise RuntimeError(
                f"{args.arm} backend failed and switched to {encoder.backend}"
            )
        preprocess_measurements.append(metrics["preprocess_ms"])
        encode_measurements.append(metrics["encode_ms"])

    payload, dtype_name = tensor_bytes(features)
    args.tensor.write_bytes(payload)
    metadata = encoder.metadata(features, dtype_name, {})
    args.result.write_text(
        json.dumps(
            {
                "arm": args.arm,
                "backend": metadata["backend"],
                "device": metadata["device"],
                "revision": metadata["revision"],
                "tensor": metadata["tensor"],
                "timing": {
                    "preprocess": summarize(preprocess_measurements),
                    "encode": summarize(encode_measurements),
                    "total_local_work": summarize(
                        [
                            preprocess + encode
                            for preprocess, encode in zip(
                                preprocess_measurements, encode_measurements
                            )
                        ]
                    ),
                },
                "optimization_profile": metadata.get("optimization_profile"),
            },
            indent=2,
        )
        + "\n"
    )


def run_arm(
    arm: str,
    model: str,
    revision: str | None,
    image: Path,
    cache_dir: Path | None,
    warmups: int,
    rounds: int,
    root: Path,
) -> tuple[dict, Path]:
    result = root / f"{arm}.json"
    tensor = root / f"{arm}.bin"
    command = [
        sys.executable,
        "-m",
        "cross_device_gemma.edge_encoder_benchmark",
        "--worker",
        "--arm",
        arm,
        "--model",
        model,
        "--input",
        str(image),
        "--result",
        str(result),
        "--tensor",
        str(tensor),
        "--warmups",
        str(warmups),
        "--rounds",
        str(rounds),
    ]
    if revision:
        command.extend(("--revision", revision))
    if cache_dir:
        command.extend(("--cache-dir", str(cache_dir)))
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"{arm} benchmark failed: {detail}")
    return json.loads(result.read_text()), tensor


def benchmark(args: argparse.Namespace) -> dict:
    if args.warmups < 1 or args.rounds < 2:
        raise ValueError("benchmark requires at least 1 warmup and 2 measured rounds")
    image = args.input.resolve()
    with tempfile.TemporaryDirectory(prefix="edge-encoder-benchmark-") as temporary:
        root = Path(temporary)
        print("Benchmarking PyTorch/MPS compatibility backend...", file=sys.stderr)
        baseline, baseline_tensor = run_arm(
            "baseline",
            args.model,
            args.revision,
            image,
            args.cache_dir,
            args.warmups,
            args.rounds,
            root,
        )
        print("Benchmarking optimized MLX backend...", file=sys.stderr)
        optimized, optimized_tensor = run_arm(
            "optimized",
            args.model,
            args.revision,
            image,
            args.cache_dir,
            args.warmups,
            args.rounds,
            root,
        )
        numerical = compare_bfloat16(baseline_tensor, optimized_tensor)

    if baseline["tensor"]["shape"] != optimized["tensor"]["shape"]:
        raise RuntimeError("Benchmark backends produced different tensor shapes")
    if baseline["revision"] != optimized["revision"]:
        raise RuntimeError("Benchmark backends resolved different model revisions")
    baseline_encode_p50 = baseline["timing"]["encode"]["p50_ms"]
    optimized_encode_p50 = optimized["timing"]["encode"]["p50_ms"]
    baseline_total_p50 = baseline["timing"]["total_local_work"]["p50_ms"]
    optimized_total_p50 = optimized["timing"]["total_local_work"]["p50_ms"]
    return {
        "benchmark": "edge_encoder_optimized_vs_pytorch",
        "model": args.model,
        "revision": baseline["revision"],
        "input": str(image),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "protocol": {
            "process_isolation": True,
            "warmups_per_arm": args.warmups,
            "measured_rounds_per_arm": args.rounds,
            "model_loading_excluded": True,
            "measurement": "native preprocessing and synchronized encoder execution",
            "preprocessing": "each backend's production-native image processor",
        },
        "baseline": baseline,
        "optimized": optimized,
        "comparison": {
            "encoder_p50_speedup": baseline_encode_p50 / optimized_encode_p50,
            "encoder_p50_latency_reduction_percent": (
                (baseline_encode_p50 - optimized_encode_p50)
                / baseline_encode_p50
                * 100
            ),
            "total_local_work_p50_speedup": (
                baseline_total_p50 / optimized_total_p50
            ),
            "total_local_work_p50_latency_reduction_percent": (
                (baseline_total_p50 - optimized_total_p50)
                / baseline_total_p50
                * 100
            ),
            "numerical": numerical,
        },
    }


def format_report(report: dict) -> str:
    baseline = report["baseline"]
    optimized = report["optimized"]
    comparison = report["comparison"]
    numerical = comparison["numerical"]
    shape = " x ".join(str(value) for value in optimized["tensor"]["shape"])
    return "\n".join(
        (
            "Edge Encoder Optimization A/B",
            f"Model: {report['model']}@{report['revision'][:12]}",
            f"Input: {report['input']}",
            "",
            "Backend                 Encoder p50   Encoder p90   Total local p50",
            "----------------------  ------------  ------------  ---------------",
            f"PyTorch/{baseline['device']:<14}"
            f"{baseline['timing']['encode']['p50_ms']:>10.1f} ms"
            f"{baseline['timing']['encode']['p90_ms']:>12.1f} ms"
            f"{baseline['timing']['total_local_work']['p50_ms']:>15.1f} ms",
            f"Optimized MLX         "
            f"{optimized['timing']['encode']['p50_ms']:>10.1f} ms"
            f"{optimized['timing']['encode']['p90_ms']:>12.1f} ms"
            f"{optimized['timing']['total_local_work']['p50_ms']:>15.1f} ms",
            "",
            f"Encoder speedup: {comparison['encoder_p50_speedup']:.2f}x",
            "Encoder latency reduction: "
            f"{comparison['encoder_p50_latency_reduction_percent']:.1f}%",
            "Total local-work reduction: "
            f"{comparison['total_local_work_p50_latency_reduction_percent']:.1f}%",
            f"Output contract: [{shape}] {optimized['tensor']['dtype']}",
            "Numerical profile: accuracy-qualified "
            f"(relative L2 {numerical['relative_l2_difference']:.4f}, "
            f"finite: {str(numerical['finite']).lower()})",
        )
    )


def worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--arm", choices=("baseline", "optimized"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--tensor", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--token")
    return parser


if __name__ == "__main__":
    run_worker(worker_parser().parse_args())
