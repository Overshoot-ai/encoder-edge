"""Measure MLX Gemma 4 BF16/Q4 vision latency under controlled CPU load."""

import argparse
import importlib.metadata
import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs
from PIL import Image, ImageOps

from .mlx_vision_quantization_ab import make_encoder, summarize


@contextmanager
def cpu_pressure(worker_count: int, settle_seconds: float):
    workers = []
    try:
        for _ in range(worker_count):
            workers.append(
                subprocess.Popen(
                    ["/usr/bin/yes"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
        if settle_seconds:
            time.sleep(settle_seconds)
        yield
    finally:
        for worker in workers:
            worker.terminate()
        for worker in workers:
            try:
                worker.wait(timeout=2)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()


def timed(encoder, pixels):
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    output = encoder(pixels)
    mx.eval(output)
    mx.synchronize()
    return {
        "wall_ms": (time.perf_counter() - wall_start) * 1000,
        "process_cpu_ms": (time.process_time() - cpu_start) * 1000,
    }, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default="mlx-community/gemma-4-e4b-it-4bit"
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8])
    parser.add_argument(
        "--pressure-workers", nargs="+", type=int, default=[0, 2, 4, 8, 4, 2, 0]
    )
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 1 or any(value < 0 for value in args.pressure_workers):
        parser.error("rounds must be positive and pressure workers must be non-negative")

    mx.set_wired_limit(2 * 1024**3)
    quantization = {
        "mode": "affine",
        "bits": 4,
        "group_size": 64,
        "quantize_input": False,
    }
    bf16_tower, bf16_encoder, processor, config, bf16_bytes = make_encoder(
        args.checkpoint, False, quantization
    )
    q4_tower, q4_encoder, _, _, q4_bytes = make_encoder(
        args.checkpoint, True, quantization
    )

    image = ImageOps.fit(
        Image.open(args.image).convert("RGB"),
        (854, 480),
        method=Image.Resampling.LANCZOS,
    )
    prompt = apply_chat_template(
        processor, config, "Describe this image.", num_images=1
    )
    pixels = prepare_inputs(
        processor,
        images=[image],
        prompts=prompt,
        add_special_tokens=False,
    )["pixel_values"]

    observations = []
    for pass_index, worker_count in enumerate(args.pressure_workers):
        with cpu_pressure(worker_count, args.settle_seconds):
            observation = {
                "pass": pass_index,
                "pressure_workers": worker_count,
                "load_average_before": list(os.getloadavg()),
                "batches": {},
            }
            for batch_size in args.batches:
                batch_pixels = (
                    pixels
                    if batch_size == 1
                    else mx.repeat(pixels, batch_size, axis=0)
                )
                timed(bf16_encoder, batch_pixels)
                timed(q4_encoder, batch_pixels)
                samples = {
                    "bf16": {"wall_ms": [], "process_cpu_ms": []},
                    "q4": {"wall_ms": [], "process_cpu_ms": []},
                }
                outputs = {}
                for round_index in range(args.rounds):
                    order = (
                        (("bf16", bf16_encoder), ("q4", q4_encoder))
                        if round_index % 2 == 0
                        else (("q4", q4_encoder), ("bf16", bf16_encoder))
                    )
                    for name, encoder in order:
                        sample, outputs[name] = timed(encoder, batch_pixels)
                        for metric, value in sample.items():
                            samples[name][metric].append(value)

                difference = outputs["bf16"].astype(mx.float32) - outputs[
                    "q4"
                ].astype(mx.float32)
                reference_norm = mx.linalg.norm(outputs["bf16"].astype(mx.float32))
                observation["batches"][str(batch_size)] = {
                    name: {
                        metric: {
                            **summarize(values),
                            "raw": values,
                        }
                        for metric, values in metrics.items()
                    }
                    for name, metrics in samples.items()
                }
                observation["batches"][str(batch_size)]["relative_l2_error"] = (
                    mx.linalg.norm(difference).item() / reference_norm.item()
                )
            observation["load_average_after"] = list(os.getloadavg())
            observations.append(observation)
            print(
                f"pass={pass_index} workers={worker_count} "
                + " ".join(
                    f"B{batch} bf16={observation['batches'][str(batch)]['bf16']['wall_ms']['p50']:.1f}ms "
                    f"q4={observation['batches'][str(batch)]['q4']['wall_ms']['p50']:.1f}ms"
                    for batch in args.batches
                ),
                flush=True,
            )

    result = {
        "metadata": {
            "checkpoint": args.checkpoint,
            "mlx_version": importlib.metadata.version("mlx"),
            "mlx_vlm_version": importlib.metadata.version("mlx-vlm"),
            "device_info": mx.device_info(),
            "rounds": args.rounds,
            "pressure_schedule": args.pressure_workers,
            "pressure_command": "/usr/bin/yes > /dev/null per worker",
            "settle_seconds": args.settle_seconds,
            "interleaved_bf16_q4": True,
            "quantization": quantization,
            "parameter_bytes": {"bf16": bf16_bytes, "q4": q4_bytes},
        },
        "observations": observations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
