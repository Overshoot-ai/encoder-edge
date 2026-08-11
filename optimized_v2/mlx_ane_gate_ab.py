"""Isolated Cider ANE+GPU gate-projection gate for Gemma 4 vision."""

import argparse
import importlib.metadata
import json
import resource
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm import load

from .mlx_vision_quantization_ab import summarize


def metrics(reference, candidate):
    reference = np.asarray(reference.astype(mx.float32))
    candidate = np.asarray(candidate.astype(mx.float32))
    difference = candidate - reference
    return {
        "finite": bool(np.isfinite(candidate).all()),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "relative_l2_error": float(
            np.linalg.norm(difference.ravel()) / np.linalg.norm(reference.ravel())
        ),
        "cosine_similarity": float(
            np.dot(reference.ravel(), candidate.ravel())
            / (np.linalg.norm(reference) * np.linalg.norm(candidate))
        ),
    }


def timed(function, value):
    started = time.perf_counter()
    output = function(value)
    mx.eval(output)
    mx.synchronize()
    return (time.perf_counter() - started) * 1000, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cider-experimental", type=Path, required=True)
    parser.add_argument("--sequence", type=int, default=2376)
    parser.add_argument("--ane-fraction", type=float, default=0.65)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.cider_experimental))
    from split_linear import ANEBridge, SplitLinear

    model, _ = load("mlx-community/gemma-4-e4b-it-4bit")
    linear = model.vision_tower.encoder.layers[0].mlp.gate_proj.linear
    fp16_linear = nn.Linear(768, 3072, bias=False)
    fp16_linear.weight = linear.weight.astype(mx.float16)
    mx.eval(linear.weight, fp16_linear.weight)
    mx.random.seed(31)
    value = (mx.random.normal((1, args.sequence, 768)) * 0.25).astype(mx.bfloat16)
    mx.eval(value)

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    bridge = ANEBridge.shared()
    split = SplitLinear(
        linear,
        bridge,
        args.sequence,
        ane_frac=args.ane_fraction,
        name="vision.layer.0.gate_proj",
    )
    SplitLinear.set_prefill(True)
    functions = {
        "bf16_gpu": lambda x: linear(x),
        "fp16_gpu": lambda x: fp16_linear(x.astype(mx.float16)),
        "ane_gpu_split": lambda x: split(x),
    }
    for function in functions.values():
        for _ in range(args.warmups):
            mx.eval(function(value))
    mx.synchronize()

    timings = {name: [] for name in functions}
    outputs = {}
    names = list(functions)
    for round_index in range(args.rounds):
        offset = round_index % len(names)
        order = names[offset:] + names[:offset]
        if round_index % 2:
            order.reverse()
        for name in order:
            elapsed, outputs[name] = timed(functions[name], value)
            timings[name].append(elapsed)

    summaries = {
        name: {**summarize(values), "raw": values}
        for name, values in timings.items()
    }
    result = {
        "metadata": {
            "mlx_version": importlib.metadata.version("mlx"),
            "device_info": mx.device_info(),
            "shape": [args.sequence, 768, 3072],
            "ane_fraction": args.ane_fraction,
            "ane_output_channels": split.ane_oc,
            "gpu_output_channels": split.gpu_oc,
            "warmups": args.warmups,
            "rounds": args.rounds,
            "ane_model_count": bridge.model_count,
            "rss_max_before_bytes": rss_before,
            "rss_max_after_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "timings": summaries,
        "speedup_vs_bf16": summaries["bf16_gpu"]["p50"]
        / summaries["ane_gpu_split"]["p50"],
        "speedup_vs_fp16": summaries["fp16_gpu"]["p50"]
        / summaries["ane_gpu_split"]["p50"],
        "error_vs_bf16": metrics(outputs["bf16_gpu"], outputs["ane_gpu_split"]),
        "fp16_error_vs_bf16": metrics(outputs["bf16_gpu"], outputs["fp16_gpu"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
