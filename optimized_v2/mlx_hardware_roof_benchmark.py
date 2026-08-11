import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def measure(function, inputs: tuple, rounds: int) -> dict:
    compiled = mx.compile(function)
    for _ in range(5):
        output = compiled(*inputs)
        mx.eval(output)
    mx.synchronize()
    values = []
    for _ in range(rounds):
        started = time.perf_counter()
        output = compiled(*inputs)
        mx.eval(output)
        mx.synchronize()
        values.append((time.perf_counter() - started) * 1000)
    return {
        "mean_ms": statistics.mean(values),
        "p50_ms": percentile(values, 0.5),
        "p90_ms": percentile(values, 0.9),
        "min_ms": min(values),
        "max_ms": max(values),
        "raw_ms": values,
    }


def benchmark_gemm(name: str, m: int, n: int, k: int, rounds: int) -> dict:
    left = mx.ones((m, k), dtype=mx.bfloat16)
    weight = mx.ones((n, k), dtype=mx.bfloat16)
    mx.eval(left, weight)
    metrics = measure(lambda x, w: x @ w.T, (left, weight), rounds)
    flops = 2 * m * n * k
    return {
        "name": name,
        "m": m,
        "n": n,
        "k": k,
        "dtype": "bfloat16",
        "flops": flops,
        "achieved_tflops_p50": flops / (metrics["p50_ms"] / 1000) / 1e12,
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gemm_shapes = [
        ("square_1024", 1024, 1024, 1024),
        ("square_2048", 2048, 2048, 2048),
        ("square_4096", 4096, 4096, 4096),
        ("vision_projection", 2376, 768, 768),
        ("vision_fused_qkv", 2376, 2304, 768),
        ("vision_mlp_up", 2376, 3072, 768),
        ("vision_mlp_down", 2376, 768, 3072),
    ]
    gemms = []
    for name, m, n, k in gemm_shapes:
        result = benchmark_gemm(name, m, n, k, args.rounds)
        gemms.append(result)
        print(
            f"{name}: {result['p50_ms']:.3f} ms, "
            f"{result['achieved_tflops_p50']:.3f} TFLOP/s",
            flush=True,
        )

    elements = 32 * 1024 * 1024
    first = mx.ones((elements,), dtype=mx.bfloat16)
    second = mx.ones((elements,), dtype=mx.bfloat16)
    mx.eval(first, second)
    streaming = measure(lambda x, y: x * 0.5 + y, (first, second), args.rounds)
    transferred_bytes = elements * 2 * 3
    streaming["elements"] = elements
    streaming["minimum_bytes_per_iteration"] = transferred_bytes
    streaming["minimum_bandwidth_gbps_p50"] = (
        transferred_bytes / (streaming["p50_ms"] / 1000) / 1e9
    )
    print(
        f"streaming_triad: {streaming['p50_ms']:.3f} ms, "
        f"{streaming['minimum_bandwidth_gbps_p50']:.3f} GB/s",
        flush=True,
    )

    result = {
        "metadata": {
            "device": str(mx.default_device()),
            "device_info": mx.device_info(),
            "rounds": args.rounds,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
        "gemms": gemms,
        "streaming_triad": streaming,
        "summary": {
            "empirical_bf16_compute_roof_tflops": max(
                item["achieved_tflops_p50"] for item in gemms
            ),
            "empirical_streaming_bandwidth_roof_gbps": streaming[
                "minimum_bandwidth_gbps_p50"
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
