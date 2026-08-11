import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm import load

from .mlx_vision_optimizations import (
    encode_gemma4_unpadded_batch1,
    fuse_gemma4_rope_layout,
    make_segmented_gemma4_encoder,
    optimize_gemma4_positions,
)


def run_single(
    input_path: Path,
    layers: int,
    output: Path,
    segment_size: int | None,
    evaluate_segments: bool,
) -> None:
    mx.set_wired_limit(2 * 1024**3)
    model, _ = load("mlx-community/gemma-4-e4b-it-4bit")
    tower = model.vision_tower
    projector = model.embed_vision
    optimize_gemma4_positions(tower)
    if layers:
        fuse_gemma4_rope_layout(tower, layers)
    pixels = mx.load(str(input_path))["pixels"]
    del model
    gc.collect()
    mx.clear_cache()
    if segment_size is None:
        encode = mx.compile(
            lambda value: encode_gemma4_unpadded_batch1(tower, projector, value)
        )
    else:
        encode = make_segmented_gemma4_encoder(
            tower,
            projector,
            segment_size,
            evaluate_segments,
        )
    for _ in range(5):
        mx.eval(encode(pixels))
    mx.synchronize()
    active_before = mx.get_active_memory()
    cache_before = mx.get_cache_memory()
    mx.reset_peak_memory()
    started = time.perf_counter()
    value = encode(pixels)
    mx.eval(value)
    mx.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000
    result = {
        "fused_layers": layers,
        "segment_size": segment_size,
        "evaluate_segments": evaluate_segments,
        "elapsed_ms": elapsed_ms,
        "pixel_shape": list(pixels.shape),
        "active_before_bytes": active_before,
        "cache_before_bytes": cache_before,
        "active_after_bytes": mx.get_active_memory(),
        "cache_after_bytes": mx.get_cache_memory(),
        "peak_memory_bytes": mx.get_peak_memory(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--segment-size", type=int)
    parser.add_argument("--evaluate-segments", action="store_true")
    args = parser.parse_args()
    if args.layers is not None:
        run_single(
            args.input,
            args.layers,
            args.output,
            args.segment_size,
            args.evaluate_segments,
        )
        return

    results = []
    output_dir = args.output.parent / f"{args.output.stem}-parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    for layers in (0, 1, 2, 4, 8, 16):
        part = output_dir / f"layers-{layers:02d}.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "optimized_v2.mlx_fusion_memory_profile",
                "--input",
                str(args.input),
                "--output",
                str(part),
                "--layers",
                str(layers),
            ],
            check=True,
        )
        results.append(json.loads(part.read_text()))
    args.output.write_text(json.dumps({"results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
