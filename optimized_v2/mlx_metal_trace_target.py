import argparse
import gc
import json
import time
from pathlib import Path

import mlx.core as mx
from PIL import Image, ImageOps
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

from .mlx_vision_optimizations import (
    encode_gemma4_unpadded_batch1,
    fuse_gemma4_rope_layout,
    optimize_gemma4_positions,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--trigger", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--variant",
        choices=("production-unmasked", "rope-layout"),
        default="production-unmasked",
    )
    args = parser.parse_args()

    image = ImageOps.fit(
        Image.open(args.image).convert("RGB"),
        (854, 480),
        method=Image.Resampling.LANCZOS,
    )
    model, processor = load("mlx-community/gemma-4-e4b-it-4bit")
    prompt = apply_chat_template(
        processor,
        model.config,
        "Describe this image.",
        num_images=1,
    )
    inputs = prepare_inputs(
        processor,
        images=[image],
        prompts=prompt,
        add_special_tokens=False,
    )
    pixels = inputs["pixel_values"]
    tower = model.vision_tower
    projector = model.embed_vision
    optimize_gemma4_positions(tower)
    if args.variant == "rope-layout":
        fuse_gemma4_rope_layout(tower)
    encode = mx.compile(
        lambda value: encode_gemma4_unpadded_batch1(tower, projector, value)
    )
    del model
    gc.collect()
    mx.clear_cache()
    for _ in range(5):
        output = encode(pixels)
        mx.eval(output)
    mx.synchronize()

    args.ready.write_text("ready\n")
    if args.trigger is not None:
        while not args.trigger.exists():
            time.sleep(0.01)
    elif args.delay:
        time.sleep(args.delay)

    elapsed_ms = []
    for _ in range(args.runs):
        started = time.perf_counter()
        output = encode(pixels)
        mx.eval(output)
        mx.synchronize()
        elapsed_ms.append((time.perf_counter() - started) * 1000)
    args.output.write_text(
        json.dumps(
            {
                "variant": args.variant,
                "elapsed_ms": elapsed_ms[0] if args.runs == 1 else elapsed_ms,
                "output_shape": list(output.shape),
                "output_dtype": str(output.dtype),
            },
            indent=2,
        )
        + "\n"
    )
    time.sleep(2)


if __name__ == "__main__":
    main()
