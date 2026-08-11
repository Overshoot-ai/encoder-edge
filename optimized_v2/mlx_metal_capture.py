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
    encode_gemma4_reshape_pool_batch1,
    encode_gemma4_unpadded_batch1,
    fuse_gemma4_qkv_epilogue,
    fuse_gemma4_rope_layout,
    optimize_gemma4_positions,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-4-e4b-it-4bit",
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument(
        "--variant",
        choices=(
            "baseline",
            "gathered-positions",
            "production-unmasked",
            "qkv-epilogue",
            "rope-layout",
            "reshape-pool",
        ),
        default="baseline",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = Image.open(args.image).convert("RGB")
    image = ImageOps.fit(source, (854, 480), method=Image.Resampling.LANCZOS)
    model, processor = load(args.model)
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
    if args.variant != "baseline":
        optimize_gemma4_positions(tower)
    if args.variant == "qkv-epilogue":
        fuse_gemma4_qkv_epilogue(tower)
    elif args.variant == "rope-layout":
        fuse_gemma4_rope_layout(tower)
    del model
    gc.collect()
    mx.clear_cache()
    mx.synchronize()

    def encode(value):
        if args.variant == "reshape-pool":
            return encode_gemma4_reshape_pool_batch1(tower, projector, value)
        if args.variant in ("production-unmasked", "qkv-epilogue", "rope-layout"):
            return encode_gemma4_unpadded_batch1(tower, projector, value)
        return projector(tower(value, None))

    compiled = mx.compile(encode)
    graph_output = compiled(pixels)
    mx.export_to_dot(str(args.output_dir / "compiled-vision.dot"), graph_output)
    for _ in range(args.warmups):
        output = compiled(pixels)
        mx.eval(output)
    mx.synchronize()

    mx.reset_peak_memory()
    trace_path = (args.output_dir / "compiled-vision.gputrace").resolve()
    started = time.perf_counter()
    mx.metal.start_capture(str(trace_path))
    try:
        output = compiled(pixels)
        mx.eval(output)
        mx.synchronize()
    finally:
        mx.metal.stop_capture()
    elapsed_ms = (time.perf_counter() - started) * 1000

    metadata = {
        "model": args.model,
        "variant": args.variant,
        "device": str(mx.default_device()),
        "device_info": mx.device_info(),
        "pixel_shape": list(pixels.shape),
        "pixel_dtype": str(pixels.dtype),
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "warmups": args.warmups,
        "captured_wall_ms": elapsed_ms,
        "active_memory_bytes": mx.get_active_memory(),
        "cache_memory_bytes": mx.get_cache_memory(),
        "peak_memory_bytes": mx.get_peak_memory(),
        "trace": str(trace_path),
        "graph": str((args.output_dir / "compiled-vision.dot").resolve()),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
