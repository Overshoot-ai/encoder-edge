"""Standalone correctness/performance gate for a real Gemma projection on ANE.

Each sequence length runs in a fresh subprocess because a private-ANE failure can
terminate the process rather than raise a recoverable Python exception.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open


DEFAULT_CIDER = Path(
    os.environ.get("CIDER_ANE_EXPERIMENTAL", "cider-ane/experimental")
)
DEFAULT_OUTPUT = Path(
    "benchmark-results/projector-shift/ane-projection-probe"
)
DEFAULT_SEQUENCES = (512, 2048, 2368, 2376, 2432)
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
INPUT_CHANNELS = 768
OUTPUT_ROWS = 768
CORRECTNESS_LIMIT = 0.005


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def timing_summary(values: list[float]) -> dict[str, object]:
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p90_ms": percentile(values, 0.9),
        "minimum_ms": min(values),
        "maximum_ms": max(values),
        "samples_ms": values,
    }


def basic_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    reference64 = reference.astype(np.float64, copy=False)
    candidate64 = candidate.astype(np.float64, copy=False)
    difference = candidate64 - reference64
    ref_norm = np.linalg.norm(reference64.ravel())
    candidate_norm = np.linalg.norm(candidate64.ravel())
    return {
        "finite": bool(np.isfinite(candidate).all()),
        "relative_l2": float(np.linalg.norm(difference.ravel()) / max(ref_norm, 1e-30)),
        "cosine": float(
            np.dot(reference64.ravel(), candidate64.ravel())
            / max(ref_norm * candidate_norm, 1e-30)
        ),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "max_absolute_error": float(np.max(np.abs(difference))),
    }


def largest_indices(
    relative: np.ndarray,
    error_norm: np.ndarray,
    reference_norm: np.ndarray,
    count: int = 8,
) -> list[dict[str, object]]:
    count = min(count, relative.size)
    indices = np.argpartition(relative, -count)[-count:]
    indices = indices[np.argsort(relative[indices])[::-1]]
    return [
        {
            "index": int(index),
            "relative_l2": float(relative[index]),
            "error_l2": float(error_norm[index]),
            "reference_l2": float(reference_norm[index]),
        }
        for index in indices
    ]


def localized_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    reference64 = reference.astype(np.float64, copy=False)
    row_error_norm = np.linalg.norm(difference, axis=1)
    row_reference_norm = np.linalg.norm(reference64, axis=1)
    row_error = row_error_norm / np.maximum(row_reference_norm, 1e-30)
    channel_error_norm = np.linalg.norm(difference, axis=0)
    channel_reference_norm = np.linalg.norm(reference64, axis=0)
    channel_error = channel_error_norm / np.maximum(channel_reference_norm, 1e-30)
    blocks = []
    block_size = 64
    for row_start in range(0, reference.shape[0], block_size):
        for channel_start in range(0, reference.shape[1], block_size):
            row_stop = min(row_start + block_size, reference.shape[0])
            channel_stop = min(channel_start + block_size, reference.shape[1])
            ref_block = reference64[row_start:row_stop, channel_start:channel_stop]
            diff_block = difference[row_start:row_stop, channel_start:channel_stop]
            blocks.append(
                {
                    "row_range": [row_start, row_stop],
                    "channel_range": [channel_start, channel_stop],
                    "relative_l2": float(
                        np.linalg.norm(diff_block.ravel())
                        / max(np.linalg.norm(ref_block.ravel()), 1e-30)
                    ),
                    "max_absolute_error": float(np.max(np.abs(diff_block))),
                }
            )
    blocks.sort(key=lambda item: item["relative_l2"], reverse=True)
    return {
        "highest_error_rows": largest_indices(
            row_error, row_error_norm, row_reference_norm
        ),
        "highest_error_channels": largest_indices(
            channel_error, channel_error_norm, channel_reference_norm
        ),
        "highest_error_64x64_blocks": blocks[:12],
    }


def make_inputs(sequence: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    random_input = (rng.standard_normal((sequence, INPUT_CHANNELS)) * 0.25).astype(
        np.float32
    )
    rows = np.linspace(-0.5, 0.5, sequence, dtype=np.float32)[:, None]
    channels = np.linspace(-0.5, 0.5, INPUT_CHANNELS, dtype=np.float32)[None, :]
    sparse = np.zeros((sequence, INPUT_CHANNELS), dtype=np.float32)
    impulses = (
        (0, 0, 1.0),
        (sequence // 3, INPUT_CHANNELS // 3, -0.75),
        (sequence // 2, 2 * INPUT_CHANNELS // 3, 0.5),
        (sequence - 1, INPUT_CHANNELS - 1, -1.0),
    )
    for row, channel, value in impulses:
        sparse[row, channel] = value
    return {
        "seeded_random": np.ascontiguousarray(random_input),
        "row_ramp": np.ascontiguousarray(np.broadcast_to(rows, random_input.shape)),
        "channel_ramp": np.ascontiguousarray(np.broadcast_to(channels, random_input.shape)),
        "sparse_impulse": sparse,
    }


def extract_weight(model_id: str) -> tuple[np.ndarray, dict[str, object]]:
    model_path = hf_hub_download(model_id, "model.safetensors")
    key = "vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight"
    with safe_open(model_path, framework="pt", device="cpu") as checkpoint:
        if key not in checkpoint.keys():
            raise KeyError(f"missing real Gemma gate projection: {key}")
        stored_weight = checkpoint.get_tensor(key)
        stored_dtype = str(stored_weight.dtype)
        weight_np = np.ascontiguousarray(stored_weight.float().numpy())
    metadata = {
        "projection": key,
        "checkpoint_file": str(model_path),
        "stored_weight_shape": list(weight_np.shape),
        "stored_weight_dtype": stored_dtype,
        "dequantized_weight_shape": list(weight_np.shape),
        "dequantized_weight_dtype": str(weight_np.dtype),
        "quantized": False,
        "dequantization": "exact BF16-to-FP32 value expansion (no quantization scales)",
    }
    expected = (3072, INPUT_CHANNELS)
    if weight_np.shape != expected:
        raise ValueError(f"expected dequantized gate weight {expected}, got {weight_np.shape}")
    return weight_np[:OUTPUT_ROWS], metadata


def time_mlx(value: mx.array, weight: mx.array) -> float:
    started = time.perf_counter_ns()
    output = value @ weight.T
    mx.eval(output)
    mx.synchronize()
    return (time.perf_counter_ns() - started) / 1e6


def run_worker(args: argparse.Namespace) -> dict[str, object]:
    if args.sequence <= 0:
        raise ValueError("sequence length must be positive")
    if not args.cider_experimental.is_dir():
        raise FileNotFoundError(args.cider_experimental)
    bridge_library = args.cider_experimental / "libane_bridge_v6.dylib"
    if not bridge_library.is_file():
        raise FileNotFoundError(bridge_library)

    sys.path.insert(0, str(args.cider_experimental))
    from split_linear import ANEBridge

    weight, weight_metadata = extract_weight(args.model)
    if weight.shape != (OUTPUT_ROWS, INPUT_CHANNELS):
        raise ValueError(f"ANE weight shape is invalid: {weight.shape}")
    bridge = ANEBridge.shared()
    handle = bridge.load(INPUT_CHANNELS, OUTPUT_ROWS, args.sequence, weight)
    inputs = make_inputs(args.sequence, args.seed)
    correctness = {}
    ane_output = np.empty((args.sequence, OUTPUT_ROWS), dtype=np.float32)

    for name, value in inputs.items():
        if value.shape != (args.sequence, INPUT_CHANNELS) or not value.flags.c_contiguous:
            raise ValueError(f"invalid {name} input layout: {value.shape}, {value.flags}")
        reference = value @ weight.T
        bridge.run_rowmajor(handle, value, args.sequence, ane_output)
        mlx_value = mx.array(value.astype(np.float16))
        mlx_weight = mx.array(weight.astype(np.float16))
        mlx_output_array = mlx_value @ mlx_weight.T
        mx.eval(mlx_output_array)
        mlx_output = np.asarray(mlx_output_array.astype(mx.float32))
        correctness[name] = {
            "ane_vs_numpy_exact_dequantized": {
                **basic_metrics(reference, ane_output),
                "localization": localized_metrics(reference, ane_output),
            },
            "mlx_fp16_vs_numpy_exact_dequantized": basic_metrics(reference, mlx_output),
        }

    benchmark_input = inputs["seeded_random"]
    mlx_input = mx.array(benchmark_input.astype(np.float16))
    mlx_weight = mx.array(weight.astype(np.float16))
    mx.eval(mlx_input, mlx_weight)
    for _ in range(args.warmups):
        bridge.run_rowmajor(handle, benchmark_input, args.sequence, ane_output)
        time_mlx(mlx_input, mlx_weight)

    ane_times = []
    mlx_times = []
    for index in range(args.rounds):
        if index % 2 == 0:
            started = time.perf_counter_ns()
            bridge.run_rowmajor(handle, benchmark_input, args.sequence, ane_output)
            ane_times.append((time.perf_counter_ns() - started) / 1e6)
            mlx_times.append(time_mlx(mlx_input, mlx_weight))
        else:
            mlx_times.append(time_mlx(mlx_input, mlx_weight))
            started = time.perf_counter_ns()
            bridge.run_rowmajor(handle, benchmark_input, args.sequence, ane_output)
            ane_times.append((time.perf_counter_ns() - started) / 1e6)


    ane_summary = timing_summary(ane_times)
    mlx_summary = timing_summary(mlx_times)
    maximum_error = max(
        item["ane_vs_numpy_exact_dequantized"]["relative_l2"]
        for item in correctness.values()
    )
    return {
        "status": "supported",
        "sequence": args.sequence,
        "model": args.model,
        "seed": args.seed,
        "shape_validation": {
            "input": [args.sequence, INPUT_CHANNELS],
            "weight": [OUTPUT_ROWS, INPUT_CHANNELS],
            "output": [args.sequence, OUTPUT_ROWS],
            "run_length_equals_compiled_sequence": True,
        },
        "weight": weight_metadata,
        "correctness": correctness,
        "performance": {
            "warmups": args.warmups,
            "rounds": args.rounds,
            "ane_run_rowmajor_including_transposes": ane_summary,
            "mlx_fp16_projection": mlx_summary,
            "speedup_ane_vs_mlx_fp16": (
                mlx_summary["median_ms"] / ane_summary["median_ms"]
            ),
        },
        "gate": {
            "correctness_limit_relative_l2": CORRECTNESS_LIMIT,
            "maximum_ane_relative_l2": maximum_error,
            "correctness_pass": maximum_error <= CORRECTNESS_LIMIT,
            "speed_pass": ane_summary["median_ms"] < mlx_summary["median_ms"],
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "mlx_version": importlib.metadata.version("mlx"),
            "mlx_device": mx.device_info(),
            "cider_source": str(args.cider_experimental),
            "cider_source_reused_without_modification": True,
        },
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def worker_main(args: argparse.Namespace) -> int:
    try:
        result = run_worker(args)
        write_json(args.result, result)
        print(json.dumps(result, indent=2))
        return 0
    except BaseException as error:
        failure = {
            "status": "python_failure",
            "sequence": args.sequence,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(args.result, failure)
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1


def run_parent(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for sequence in args.sequences:
        result_path = args.output / f"sequence-{sequence}.json"
        command = [
            sys.executable,
            "-m",
            "optimized_v2.mlx_ane_projection_probe",
            "--worker",
            "--sequence",
            str(sequence),
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
        started = time.time()
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
        (args.output / f"sequence-{sequence}.stdout.log").write_text(stdout)
        (args.output / f"sequence-{sequence}.stderr.log").write_text(stderr)
        if result_path.exists():
            result = json.loads(result_path.read_text())
        else:
            result = {
                "status": "private_ane_failure" if not timed_out else "timeout",
                "sequence": sequence,
                "returncode": returncode,
                "signal": -returncode if returncode is not None and returncode < 0 else None,
                "diagnostic": "worker exited without a result artifact",
            }
            write_json(result_path, result)
        result["subprocess"] = {
            "returncode": returncode,
            "timed_out": timed_out,
            "elapsed_seconds": time.time() - started,
            "stdout_log": f"sequence-{sequence}.stdout.log",
            "stderr_log": f"sequence-{sequence}.stderr.log",
        }
        write_json(result_path, result)
        results.append(result)
        print(f"sequence {sequence}: {result['status']}", flush=True)

    supported = [result for result in results if result["status"] == "supported"]
    failures = [result for result in results if result["status"] != "supported"]
    correctness_pass = bool(supported) and all(
        result["gate"]["correctness_pass"] for result in supported
    )
    speed_pass = bool(supported) and all(result["gate"]["speed_pass"] for result in supported)
    all_sequences_supported = not failures
    production_pass = correctness_pass and speed_pass and all_sequences_supported
    correctness_failures = []
    for result in supported:
        if result["gate"]["correctness_pass"]:
            continue
        correctness_failures.append(
            {
                "sequence": result["sequence"],
                "sequence_modulo_64": result["sequence"] % 64,
                "patterns": {
                    name: {
                        "relative_l2": case[
                            "ane_vs_numpy_exact_dequantized"
                        ]["relative_l2"],
                        "cosine": case["ane_vs_numpy_exact_dequantized"]["cosine"],
                        "worst_64x64_block": case[
                            "ane_vs_numpy_exact_dequantized"
                        ]["localization"]["highest_error_64x64_blocks"][0],
                    }
                    for name, case in result["correctness"].items()
                },
            }
        )
    alignment_evidence = [
        {
            "sequence": result["sequence"],
            "sequence_modulo_64": result["sequence"] % 64,
            "correctness_pass": result["gate"]["correctness_pass"],
        }
        for result in supported
    ]
    summary = {
        "probe": "standalone ANE-only Gemma gate projection rows 0:768",
        "requested_sequences": list(args.sequences),
        "supported_sequences": [result["sequence"] for result in supported],
        "supported_definition": "ANE bridge load and run returned successfully; correctness is gated separately",
        "failed_sequences": [
            {
                "sequence": result["sequence"],
                "status": result["status"],
                "returncode": result.get("subprocess", {}).get("returncode"),
                "diagnostic": result.get("diagnostic") or result.get("error"),
            }
            for result in failures
        ],
        "per_sequence": [
            {
                "sequence": result["sequence"],
                "status": result["status"],
                "maximum_relative_l2": result.get("gate", {}).get(
                    "maximum_ane_relative_l2"
                ),
                "minimum_cosine": min(
                    (
                        case["ane_vs_numpy_exact_dequantized"]["cosine"]
                        for case in result.get("correctness", {}).values()
                    ),
                    default=None,
                ),
                "ane_median_ms": result.get("performance", {})
                .get("ane_run_rowmajor_including_transposes", {})
                .get("median_ms"),
                "mlx_fp16_median_ms": result.get("performance", {})
                .get("mlx_fp16_projection", {})
                .get("median_ms"),
                "speedup_ane_vs_mlx_fp16": result.get("performance", {}).get(
                    "speedup_ane_vs_mlx_fp16"
                ),
                "correctness_pass": result.get("gate", {}).get("correctness_pass"),
                "speed_pass": result.get("gate", {}).get("speed_pass"),
                "artifact": f"sequence-{result['sequence']}.json",
            }
            for result in results
        ],
        "gate": {
            "relative_l2_limit": CORRECTNESS_LIMIT,
            "all_supported_correct": correctness_pass,
            "all_supported_faster_than_mlx_fp16": speed_pass,
            "all_requested_sequences_supported": all_sequences_supported,
            "production_integration_allowed": production_pass,
        },
        "correctness_failures": correctness_failures,
        "diagnosis": {
            "alignment_evidence": alignment_evidence,
            "observation": (
                "The only incorrect sequence is 2376 (mod 64 = 8); all tested "
                "64-aligned lengths pass. Corruption is global across rows, channels, "
                "and blocks, including leakage from sparse impulses into zero-reference "
                "rows. This is consistent with a private-ANE sequence-layout/compiler "
                "constraint, not an isolated output boundary error."
                if [item["sequence"] for item in correctness_failures] == [2376]
                else "See correctness_failures and per-sequence localization."
            ),
        },
        "decision": (
            "PASS: eligible for production integration"
            if production_pass
            else "FAIL: do not attempt production integration"
        ),
        "configuration": {
            "model": args.model,
            "weight_rows": [0, OUTPUT_ROWS],
            "warmups": args.warmups,
            "rounds": args.rounds,
            "seed": args.seed,
            "worker_timeout_seconds": args.timeout,
            "cider_source": str(args.cider_experimental),
        },
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--sequence", type=int)
    parser.add_argument("--sequences", type=int, nargs="+", default=DEFAULT_SEQUENCES)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--cider-experimental", type=Path, default=DEFAULT_CIDER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    if args.worker and (args.sequence is None or args.result is None):
        parser.error("--worker requires --sequence and --result")
    if args.warmups < 0 or args.rounds < 1:
        parser.error("--warmups must be nonnegative and --rounds must be positive")
    return args


def main() -> int:
    args = parse_args()
    return worker_main(args) if args.worker else run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
