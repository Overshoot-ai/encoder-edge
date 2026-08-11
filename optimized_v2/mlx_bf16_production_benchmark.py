"""Benchmark only the qualified BF16 pre-projector MLX vision encoder."""

import argparse
import importlib.metadata
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs
from PIL import Image, ImageOps

from .mlx_vision_quantization_ab import make_encoder, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default="mlx-community/gemma-4-e4b-it-4bit"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=Path)
    input_group.add_argument("--input", type=Path)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmups < 1 or args.rounds < 1:
        parser.error("warmups and rounds must be positive")

    wired_limit = 2 * 1024**3
    mx.set_wired_limit(wired_limit)
    quantization_placeholder = {
        "mode": "affine",
        "bits": 4,
        "group_size": 64,
        "quantize_input": False,
    }
    _, encoder, processor, config, parameter_bytes = make_encoder(
        args.checkpoint, False, quantization_placeholder
    )
    if args.input is not None:
        pixels = mx.load(str(args.input))["pixels"]
        input_source = str(args.input)
    else:
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
        input_source = str(args.image)

    for _ in range(args.warmups):
        mx.eval(encoder(pixels))
    mx.synchronize()
    mx.reset_peak_memory()
    load_before = list(os.getloadavg())
    samples = []
    process_cpu_samples = []
    output = None
    for _ in range(args.rounds):
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        output = encoder(pixels)
        mx.eval(output)
        mx.synchronize()
        samples.append((time.perf_counter() - wall_start) * 1000)
        process_cpu_samples.append((time.process_time() - cpu_start) * 1000)

    if output is None:
        raise RuntimeError("Benchmark produced no output")
    result = {
        "metadata": {
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": args.checkpoint,
            "mlx_version": importlib.metadata.version("mlx"),
            "mlx_vlm_version": importlib.metadata.version("mlx-vlm"),
            "device_info": mx.device_info(),
            "dtype": "bfloat16",
            "projector_included": False,
            "graph": {
                "gathered_positions": True,
                "fused_rope_layout": True,
                "segment_size": 3,
                "evaluate_segments": True,
            },
            "wired_limit_bytes": wired_limit,
            "warmups": args.warmups,
            "rounds": args.rounds,
            "input_source": input_source,
            "pixel_shape": list(pixels.shape),
            "output_shape": list(output.shape),
            "parameter_bytes": parameter_bytes,
            "load_average_before": load_before,
            "load_average_after": list(os.getloadavg()),
        },
        "wall_latency": {**summarize(samples), "raw": samples},
        "process_cpu_time": {
            **summarize(process_cpu_samples),
            "raw": process_cpu_samples,
        },
        "peak_memory_bytes": mx.get_peak_memory(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
