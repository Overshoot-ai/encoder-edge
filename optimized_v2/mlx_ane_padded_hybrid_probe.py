"""Gate padded ANE execution and a 25% ANE / 75% MLX Gemma projection."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from .mlx_ane_projection_probe import (
    basic_metrics,
    localized_metrics,
    make_inputs,
    timing_summary,
    write_json,
)


LOGICAL_SEQUENCE = 2376
PHYSICAL_SEQUENCE = 2432
INPUT_CHANNELS = 768
ANE_CHANNELS = 768
OUTPUT_CHANNELS = 3072
CORRECTNESS_LIMIT = 0.005
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
DEFAULT_CIDER = Path(
    os.environ.get("CIDER_ANE_EXPERIMENTAL", "cider-ane/experimental")
)
DEFAULT_OUTPUT = Path(
    "benchmark-results/projector-shift/ane-projection-probe/"
    "padded-2376-hybrid-v1"
)


def extract_full_weight(model_id: str) -> tuple[np.ndarray, dict[str, object]]:
    model_path = hf_hub_download(model_id, "model.safetensors")
    key = "vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight"
    with safe_open(model_path, framework="pt", device="cpu") as checkpoint:
        if key not in checkpoint.keys():
            raise KeyError(f"missing real Gemma gate projection: {key}")
        stored = checkpoint.get_tensor(key)
        stored_dtype = str(stored.dtype)
        weight = np.ascontiguousarray(stored.float().numpy())
    if weight.shape != (OUTPUT_CHANNELS, INPUT_CHANNELS):
        raise ValueError(
            f"expected gate weight {(OUTPUT_CHANNELS, INPUT_CHANNELS)}, got {weight.shape}"
        )
    return weight, {
        "projection": key,
        "checkpoint_file": str(model_path),
        "stored_shape": list(weight.shape),
        "stored_dtype": stored_dtype,
        "expanded_dtype": str(weight.dtype),
        "dequantization": "exact BF16-to-FP32 value expansion (no quantization scales)",
        "ane_weight_rows": [0, ANE_CHANNELS],
        "gpu_weight_rows": [ANE_CHANNELS, OUTPUT_CHANNELS],
    }


class ProjectionRunner:
    def __init__(self, bridge, handle: int, weight: np.ndarray):
        self.bridge = bridge
        self.handle = handle
        self.weight = weight
        self.ane_weight = weight[:ANE_CHANNELS]
        self.full_weight_bf16 = mx.array(weight).astype(mx.bfloat16)
        self.gpu_weight_bf16 = self.full_weight_bf16[ANE_CHANNELS:]
        self.padded_input = np.empty(
            (PHYSICAL_SEQUENCE, INPUT_CHANNELS), dtype=np.float32
        )
        self.physical_ane_output = np.empty(
            (PHYSICAL_SEQUENCE, ANE_CHANNELS), dtype=np.float32
        )
        mx.eval(self.full_weight_bf16, self.gpu_weight_bf16)
        mx.synchronize()

    def prepare_ane_input(self, source_bf16: mx.array) -> None:
        source_f32 = mx.contiguous(source_bf16.astype(mx.float32))
        mx.eval(source_f32)
        mx.synchronize()
        source_np = np.asarray(source_f32)
        if source_np.shape != (LOGICAL_SEQUENCE, INPUT_CHANNELS):
            raise ValueError(f"invalid logical input shape: {source_np.shape}")
        self.padded_input.fill(0.0)
        self.padded_input[:LOGICAL_SEQUENCE] = source_np

    def run_ane_prepared(self) -> None:
        self.bridge.run_rowmajor(
            self.handle,
            self.padded_input,
            PHYSICAL_SEQUENCE,
            self.physical_ane_output,
        )

    def full_bf16(self, source_bf16: mx.array) -> mx.array:
        output = source_bf16 @ self.full_weight_bf16.T
        mx.eval(output)
        mx.synchronize()
        return output

    def gpu_only(self, source_bf16: mx.array) -> mx.array:
        output = source_bf16 @ self.gpu_weight_bf16.T
        mx.eval(output)
        mx.synchronize()
        return output

    def ane_only_end_to_end(self, source_bf16: mx.array) -> mx.array:
        self.prepare_ane_input(source_bf16)
        self.run_ane_prepared()
        output = mx.array(
            self.physical_ane_output[:LOGICAL_SEQUENCE].astype(np.float16)
        ).astype(mx.bfloat16)
        mx.eval(output)
        mx.synchronize()
        return output

    def hybrid(self, source_bf16: mx.array, pool) -> mx.array:
        self.prepare_ane_input(source_bf16)
        ane_future = pool.submit(self.run_ane_prepared)
        gpu_output = source_bf16 @ self.gpu_weight_bf16.T
        mx.eval(gpu_output)
        mx.synchronize()
        ane_future.result()
        ane_output = mx.array(
            self.physical_ane_output[:LOGICAL_SEQUENCE].astype(np.float16)
        ).astype(mx.bfloat16)
        merged = mx.concatenate((ane_output, gpu_output), axis=-1)
        mx.eval(merged)
        mx.synchronize()
        return merged


def numpy_f32(value: mx.array) -> np.ndarray:
    value_f32 = value.astype(mx.float32)
    mx.eval(value_f32)
    mx.synchronize()
    return np.asarray(value_f32)


def benchmark_summary(values: list[float]) -> dict[str, object]:
    summary = timing_summary(values)
    summary["p50_ms"] = summary["median_ms"]
    return summary


def timed(function) -> tuple[float, mx.array]:
    started = time.perf_counter_ns()
    output = function()
    return (time.perf_counter_ns() - started) / 1e6, output


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    if not args.cider_experimental.is_dir():
        raise FileNotFoundError(args.cider_experimental)
    if not (args.cider_experimental / "libane_bridge_v6.dylib").is_file():
        raise FileNotFoundError(args.cider_experimental / "libane_bridge_v6.dylib")
    if LOGICAL_SEQUENCE >= PHYSICAL_SEQUENCE or PHYSICAL_SEQUENCE % 64:
        raise ValueError("physical sequence must be a larger 64-aligned length")

    sys.path.insert(0, str(args.cider_experimental))
    from split_linear import ANEBridge

    weight, weight_metadata = extract_full_weight(args.model)
    bridge = ANEBridge.shared()
    handle = bridge.load(
        INPUT_CHANNELS,
        ANE_CHANNELS,
        PHYSICAL_SEQUENCE,
        weight[:ANE_CHANNELS],
    )
    runner = ProjectionRunner(bridge, handle, weight)
    raw_inputs = make_inputs(LOGICAL_SEQUENCE, args.seed)
    correctness = {}

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="padded-ane"
    ) as pool:
        for name, raw_input in raw_inputs.items():
            source_bf16 = mx.array(raw_input).astype(mx.bfloat16)
            mx.eval(source_bf16)
            mx.synchronize()
            source_exact = numpy_f32(source_bf16)
            exact_ane_reference = source_exact @ weight[:ANE_CHANNELS].T
            full_bf16_reference = runner.full_bf16(source_bf16)
            runner.prepare_ane_input(source_bf16)
            runner.run_ane_prepared()
            ane_np = runner.physical_ane_output[:LOGICAL_SEQUENCE].copy()
            hybrid_output = runner.hybrid(source_bf16, pool)
            hybrid_np = numpy_f32(hybrid_output)
            full_reference_np = numpy_f32(full_bf16_reference)
            correctness[name] = {
                "padded_ane_trimmed_vs_numpy_exact": {
                    **basic_metrics(exact_ane_reference, ane_np),
                    "localization": localized_metrics(exact_ane_reference, ane_np),
                },
                "hybrid_vs_full_bf16_gpu": {
                    **basic_metrics(full_reference_np, hybrid_np),
                    "localization": localized_metrics(full_reference_np, hybrid_np),
                },
            }

        benchmark_source = mx.array(raw_inputs["seeded_random"]).astype(mx.bfloat16)
        mx.eval(benchmark_source)
        mx.synchronize()
        functions = {
            "full_bf16_gpu": lambda: runner.full_bf16(benchmark_source),
            "padded_ane_only_end_to_end": lambda: runner.ane_only_end_to_end(
                benchmark_source
            ),
            "gpu_only_2304": lambda: runner.gpu_only(benchmark_source),
            "concurrent_hybrid_end_to_end": lambda: runner.hybrid(
                benchmark_source, pool
            ),
        }
        for _ in range(args.warmups):
            for function in functions.values():
                function()

        samples = {name: [] for name in functions}
        names = list(functions)
        final_outputs = {}
        for round_index in range(args.rounds):
            offset = round_index % len(names)
            order = names[offset:] + names[:offset]
            if round_index % 2:
                order.reverse()
            for name in order:
                elapsed, final_outputs[name] = timed(functions[name])
                samples[name].append(elapsed)

    timings = {name: benchmark_summary(values) for name, values in samples.items()}
    benchmark_reference = numpy_f32(final_outputs["full_bf16_gpu"])
    benchmark_hybrid = numpy_f32(final_outputs["concurrent_hybrid_end_to_end"])
    benchmark_metrics = basic_metrics(benchmark_reference, benchmark_hybrid)
    maximum_ane_error = max(
        case["padded_ane_trimmed_vs_numpy_exact"]["relative_l2"]
        for case in correctness.values()
    )
    maximum_hybrid_error = max(
        case["hybrid_vs_full_bf16_gpu"]["relative_l2"]
        for case in correctness.values()
    )
    speedup = (
        timings["full_bf16_gpu"]["median_ms"]
        / timings["concurrent_hybrid_end_to_end"]["median_ms"]
    )
    correctness_pass = maximum_hybrid_error <= CORRECTNESS_LIMIT
    speed_pass = speedup > 1.0
    return {
        "status": "completed",
        "probe": "logical-2376 padded-to-2432 ANE and 25%-ANE whole gate hybrid",
        "shape_validation": {
            "logical_input": [LOGICAL_SEQUENCE, INPUT_CHANNELS],
            "physical_ane_input": [PHYSICAL_SEQUENCE, INPUT_CHANNELS],
            "physical_ane_output": [PHYSICAL_SEQUENCE, ANE_CHANNELS],
            "trimmed_ane_output": [LOGICAL_SEQUENCE, ANE_CHANNELS],
            "hybrid_output": [LOGICAL_SEQUENCE, OUTPUT_CHANNELS],
            "padding_rows": PHYSICAL_SEQUENCE - LOGICAL_SEQUENCE,
            "physical_sequence_modulo_64": PHYSICAL_SEQUENCE % 64,
        },
        "weight": weight_metadata,
        "correctness": correctness,
        "benchmark_random_input_hybrid_vs_full_bf16": benchmark_metrics,
        "timings": timings,
        "performance": {
            "warmups": args.warmups,
            "balanced_rounds": args.rounds,
            "speedup_hybrid_vs_full_bf16_gpu": speedup,
            "hybrid_minus_full_bf16_gpu_ms": (
                timings["concurrent_hybrid_end_to_end"]["median_ms"]
                - timings["full_bf16_gpu"]["median_ms"]
            ),
            "wall_timing_scope": (
                "Per-call conversion, synchronization, zero-padding, C transposes, "
                "ANE/GPU execution, trim, dtype conversion, concat, and final sync"
            ),
            "concurrency": (
                "ANE run_rowmajor executes in one worker thread while the main thread "
                "evaluates the 2304-channel MLX projection"
            ),
        },
        "gate": {
            "relative_l2_limit": CORRECTNESS_LIMIT,
            "maximum_padded_ane_relative_l2": maximum_ane_error,
            "maximum_hybrid_relative_l2": maximum_hybrid_error,
            "correctness_pass": correctness_pass,
            "speed_pass": speed_pass,
            "production_integration_allowed": correctness_pass and speed_pass,
        },
        "decision": (
            "PASS: eligible for production integration"
            if correctness_pass and speed_pass
            else "FAIL: preserve production; do not integrate"
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "mlx_version": importlib.metadata.version("mlx"),
            "mlx_device": mx.device_info(),
            "cider_source": str(args.cider_experimental),
            "cider_source_reused_without_modification": True,
        },
    }


def worker_main(args: argparse.Namespace) -> int:
    try:
        result = run_probe(args)
        write_json(args.result, result)
        print(json.dumps(result, indent=2))
        return 0
    except BaseException as error:
        failure = {
            "status": "failure",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(args.result, failure)
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1


def parent_main(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "result.json"
    command = [
        sys.executable,
        "-m",
        "optimized_v2.mlx_ane_padded_hybrid_probe",
        "--worker",
        "--model",
        args.model,
        "--cider-experimental",
        str(args.cider_experimental),
        "--warmups",
        str(args.warmups),
        "--rounds",
        str(args.rounds),
        "--seed",
        str(args.seed),
        "--result",
        str(result_path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        returncode = None
        timed_out = True
    (args.output / "worker.stdout.log").write_text(stdout)
    (args.output / "worker.stderr.log").write_text(stderr)
    if result_path.exists():
        result = json.loads(result_path.read_text())
    else:
        result = {
            "status": "private_ane_failure" if not timed_out else "timeout",
            "returncode": returncode,
            "signal": -returncode if returncode is not None and returncode < 0 else None,
            "decision": "FAIL: preserve production; do not integrate",
        }
    result["subprocess"] = {
        "returncode": returncode,
        "timed_out": timed_out,
        "stdout_log": "worker.stdout.log",
        "stderr_log": "worker.stderr.log",
    }
    write_json(result_path, result)
    summary = {
        "status": result["status"],
        "probe": result.get("probe"),
        "shape_validation": result.get("shape_validation"),
        "pattern_correctness": {
            name: {
                "ane_relative_l2": case[
                    "padded_ane_trimmed_vs_numpy_exact"
                ]["relative_l2"],
                "ane_cosine": case["padded_ane_trimmed_vs_numpy_exact"]["cosine"],
                "hybrid_relative_l2": case["hybrid_vs_full_bf16_gpu"][
                    "relative_l2"
                ],
                "hybrid_cosine": case["hybrid_vs_full_bf16_gpu"]["cosine"],
            }
            for name, case in result.get("correctness", {}).items()
        },
        "timings": result.get("timings"),
        "performance": result.get("performance"),
        "gate": result.get("gate"),
        "decision": result.get("decision", "FAIL: preserve production; do not integrate"),
        "detail_artifact": "result.json",
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if result["status"] == "completed" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--cider-experimental", type=Path, default=DEFAULT_CIDER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    if args.worker and args.result is None:
        parser.error("--worker requires --result")
    if args.warmups < 5 or args.rounds < 30:
        parser.error("this gate requires at least 5 warmups and 30 rounds")
    return args


def main() -> int:
    args = parse_args()
    return worker_main(args) if args.worker else parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
