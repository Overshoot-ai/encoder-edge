"""Export and validate a fixed-shape Gemma 4 E4B vision tower for WebGPU.

The exported graph stops before the server-side RMSNorm/projector. Position
embeddings, RoPE values, and pooling geometry are constants for one patch grid.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .coreml_gemma4_e4b_vision import (
    error_metrics,
    load_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact", type=Path, default=Path("artifacts/gemma-4-e4b/client")
    )
    parser.add_argument(
        "--image", type=Path, default=Path("artifacts/gemma-4-12b/sample.png")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=854)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--skip-runtime", action="store_true")
    return parser.parse_args()


def artifact_size(path: Path) -> int:
    prefix = path.name + "."
    return sum(
        item.stat().st_size
        for item in path.parent.iterdir()
        if item == path or item.name.startswith(prefix)
    )


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    (
        wrapper,
        pixels,
        positions,
        padded_pixels,
        padded_positions,
        tower,
        _,
    ) = load_experiment(args, stable_norms=False)
    wrapper.projector = torch.nn.Identity()

    with torch.inference_mode():
        fixed_reference = wrapper(pixels).numpy()
        source_error = None
        if args.layers is None:
            source = tower(padded_pixels, padded_positions).last_hidden_state.numpy()
            source_error = error_metrics(source[None, ...], fixed_reference)

    export_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    wrapper.to(dtype=export_dtype).eval()
    export_pixels = pixels.to(dtype=export_dtype)
    with torch.inference_mode():
        export_reference = wrapper(export_pixels).float().numpy()
    if not np.isfinite(export_reference).all():
        raise RuntimeError(f"{args.dtype} PyTorch output contains non-finite values")

    started = time.perf_counter()
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (export_pixels,),
            args.output,
            input_names=["pixel_values"],
            output_names=["image_features"],
            opset_version=args.opset,
            dynamo=False,
            do_constant_folding=True,
            external_data=True,
        )
    export_seconds = time.perf_counter() - started

    import onnx

    started = time.perf_counter()
    model = onnx.load(args.output, load_external_data=False)
    onnx.checker.check_model(model)
    check_seconds = time.perf_counter() - started

    result: dict[str, object] = {
        "artifact": str(args.artifact),
        "onnx": str(args.output),
        "opset": args.opset,
        "dtype": args.dtype,
        "requested_image_size": [args.width, args.height],
        "fixed_patch_shape": list(export_pixels.shape),
        "fixed_patch_grid": [
            int(positions[0, :, 0].max().item()) + 1,
            int(positions[0, :, 1].max().item()) + 1,
        ],
        "layers": len(wrapper.layers),
        "output_shape": list(export_reference.shape),
        "source_vs_fixed_float32": source_error,
        f"fixed_float32_vs_{args.dtype}": error_metrics(
            fixed_reference, export_reference
        ),
        "onnx_bytes": artifact_size(args.output),
        "export_seconds": export_seconds,
        "check_seconds": check_seconds,
    }

    if not args.skip_runtime:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        started = time.perf_counter()
        session = ort.InferenceSession(
            str(args.output), sess_options=options, providers=["CPUExecutionProvider"]
        )
        load_seconds = time.perf_counter() - started
        started = time.perf_counter()
        candidate = session.run(
            ["image_features"],
            {"pixel_values": export_pixels.numpy()},
        )[0]
        inference_seconds = time.perf_counter() - started
        if not np.isfinite(candidate).all():
            raise RuntimeError("ONNX Runtime output contains non-finite values")
        result["onnxruntime"] = {
            "providers": session.get_providers(),
            "load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "output_dtype": str(candidate.dtype),
            "output_shape": list(candidate.shape),
            "vs_pytorch_export_dtype": error_metrics(export_reference, candidate),
            "vs_fixed_float32": error_metrics(fixed_reference, candidate),
        }

    result_path = args.output.with_suffix(".results.json")
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
