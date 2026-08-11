import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image
from mlx.utils import tree_flatten
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

from .mlx_vision_optimizations import (
    exact_pool_gemma4_unpadded,
    fuse_gemma4_post_reduction_epilogue,
    fuse_gemma4_qkv_epilogue,
    fuse_gemma4_rope_layout,
    fuse_gemma4_rope_and_output_layout,
    optimize_gemma4_shapeless_rope,
    optimize_gemma4_positions,
)


MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
DATASET_ID = "HuggingFaceM4/ChartQA"
DATASET_SPLIT = "test"
VALIDATION_INDICES = (0, 4, 8, 14, 26, 46)
MAX_SOFT_TOKENS = 273
SCHEMA_VERSION = 1


def array_hash(value: mx.array) -> str:
    mx.eval(value)
    if value.dtype == mx.bfloat16:
        data = np.array(value.view(mx.uint16), copy=True).tobytes(order="C")
    else:
        data = np.ascontiguousarray(np.array(value, copy=True)).tobytes(order="C")
    return hashlib.sha256(data).hexdigest()


def image_hash(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(rgb.width.to_bytes(4, "little"))
    digest.update(rgb.height.to_bytes(4, "little"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def patch_positions(tower, pixels: mx.array) -> tuple[mx.array, mx.array]:
    batch, _, height, width = pixels.shape
    count = (height // tower.patch_size) * (width // tower.patch_size)
    positions, padding, _ = tower._patch_positions_single(
        height,
        width,
        max_patches=count,
    )
    return (
        mx.array(np.tile(positions[None], (batch, 1, 1))),
        mx.array(np.tile(padding[None], (batch, 1))),
    )


def attention_mask(padding: mx.array, dtype) -> mx.array:
    valid = ~padding
    mask = mx.expand_dims(valid, 1) * mx.expand_dims(valid, 2)
    mask = mx.where(mask, mx.array(0.0, dtype=dtype), mx.array(-1e4, dtype=dtype))
    return mx.expand_dims(mask, 1)


def raw_patches(pixels: mx.array, patch_size: int) -> mx.array:
    batch, channels, height, width = pixels.shape
    patch_height = height // patch_size
    patch_width = width // patch_size
    patches = pixels.reshape(
        batch,
        channels,
        patch_height,
        patch_size,
        patch_width,
        patch_size,
    )
    patches = patches.transpose(0, 2, 4, 3, 5, 1)
    return patches.reshape(
        batch,
        patch_height * patch_width,
        channels * patch_size * patch_size,
    )


def run_case(
    tower,
    projector,
    inputs: dict[str, mx.array],
    case: dict,
    candidate: str,
) -> dict[str, mx.array]:
    padded = case["padded"]
    if padded:
        patches = inputs["patches"]
        positions = inputs["positions"]
        padding = mx.all(positions == -1, axis=-1)
        hidden = tower.patch_embedder.embed_patches(patches, positions, padding)
        mask = attention_mask(padding, hidden.dtype)
        output_length = patches.shape[1] // (tower.pooling_kernel_size**2)
    else:
        pixels = inputs["pixels"]
        positions, padding = patch_positions(tower, pixels)
        hidden = tower.patch_embedder(pixels, positions, padding)
        mask = None
        output_length = hidden.shape[1] // (tower.pooling_kernel_size**2)

    results = {"patch_embed": hidden}
    for index, layer in enumerate(tower.encoder.layers):
        hidden = layer(hidden, positions, mask)
        results[f"block_{index:02d}_post"] = hidden

    if candidate == "exact_pool" and not padded:
        pooled = exact_pool_gemma4_unpadded(
            tower,
            hidden,
            case["patch_grid"][0],
            case["patch_grid"][1],
        )
    elif candidate == "reshape_pool" and not padded:
        pool = tower.pooling_kernel_size
        patch_height = case["patch_grid"][0]
        patch_width = case["patch_grid"][1]
        hidden_size = hidden.shape[-1]
        pooled = hidden.reshape(
            1,
            patch_height // pool,
            pool,
            patch_width // pool,
            pool,
            hidden_size,
        )
        pooled = pooled.astype(mx.float32).mean(axis=(2, 4))
        pooled = pooled.reshape(1, output_length, hidden_size).astype(hidden.dtype)
        pooled = pooled * tower.pooler.root_hidden_size
    else:
        pooled, pool_mask = tower.pooler(
            hidden,
            positions,
            padding,
            output_length=output_length,
        )
        if padded:
            valid_pool = (
                pool_mask if pool_mask.shape[1] == output_length else ~pool_mask
            )
            valid_count = int(valid_pool[0].astype(mx.int32).sum().item())
            pooled = pooled[:, :valid_count]
    results["pooled"] = pooled

    if tower.config.standardize:
        pooled = (pooled - tower.std_bias) * tower.std_scale
    final = projector(pooled)
    results["final"] = final
    mx.eval(*results.values())
    return results


def save_case_tensors(path: Path, values: dict[str, mx.array]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), values)


def load_model(dtype=None):
    model, processor = load(MODEL_ID)
    config = model.config
    tower = model.vision_tower
    projector = model.embed_vision
    optimize_gemma4_positions(tower)
    if dtype is not None:
        tower.set_dtype(dtype)
        projector.set_dtype(dtype)
        floating = []
        for _, value in tree_flatten(tower.parameters()):
            if mx.issubdtype(value.dtype, mx.floating):
                floating.append(value)
        for _, value in tree_flatten(projector.parameters()):
            if mx.issubdtype(value.dtype, mx.floating):
                floating.append(value)
        if any(value.dtype != dtype for value in floating):
            raise RuntimeError(f"Failed to cast every floating parameter to {dtype}")
        mx.eval(*floating)
    del model
    gc.collect()
    mx.clear_cache()
    return tower, projector, processor, config


def build_references(output: Path, force: bool) -> None:
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(f"Reference manifest already exists: {manifest_path}")
    output.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    tower, projector, processor, config = load_model()
    processor.image_processor.max_soft_tokens = MAX_SOFT_TOKENS
    prompt = apply_chat_template(processor, config, "x", num_images=1)
    cases = []
    prepared = {}
    for index in VALIDATION_INDICES:
        sample = dataset[index]
        image = sample["image"].convert("RGB")
        pixels = prepare_inputs(
            processor,
            images=[image],
            prompts=prompt,
            add_special_tokens=False,
        )["pixel_values"]
        patch_grid = [
            pixels.shape[-2] // tower.patch_size,
            pixels.shape[-1] // tower.patch_size,
        ]
        case_id = f"chartqa-{index:04d}"
        case_dir = output / "cases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        image.save(case_dir / "source.png", format="PNG")
        inputs = {"pixels": pixels}
        save_case_tensors(case_dir / "input.safetensors", inputs)
        case = {
            "case_id": case_id,
            "dataset_index": index,
            "query": sample["query"],
            "targets": sample["label"],
            "source_size": list(image.size),
            "image_sha256": image_hash(image),
            "pixel_shape": list(pixels.shape),
            "pixel_sha256": array_hash(pixels),
            "patch_grid": patch_grid,
            "patch_count": patch_grid[0] * patch_grid[1],
            "expected_tokens": patch_grid[0] * patch_grid[1] // 9,
            "padded": False,
        }
        cases.append(case)
        prepared[case_id] = inputs

    base = cases[0]
    pixels = prepared[base["case_id"]]["pixels"]
    patches = raw_patches(pixels, tower.patch_size)
    positions, _ = patch_positions(tower, pixels)
    pad_count = tower.pooling_kernel_size**2
    patches = mx.concatenate(
        [
            patches,
            mx.zeros((1, pad_count, patches.shape[-1]), dtype=patches.dtype),
        ],
        axis=1,
    )
    positions = mx.concatenate(
        [positions, mx.full((1, pad_count, 2), -1, dtype=mx.int32)],
        axis=1,
    )
    padded_id = f"{base['case_id']}-padded"
    padded_dir = output / "cases" / padded_id
    padded_inputs = {"patches": patches, "positions": positions}
    save_case_tensors(padded_dir / "input.safetensors", padded_inputs)
    padded_case = {
        **base,
        "case_id": padded_id,
        "pixel_sha256": None,
        "patch_count": int(patches.shape[1]),
        "expected_tokens": base["expected_tokens"],
        "padded": True,
        "padding_count": pad_count,
        "patches_sha256": array_hash(patches),
        "positions_sha256": array_hash(positions),
    }
    cases.append(padded_case)
    prepared[padded_id] = padded_inputs

    for case in cases:
        values = run_case(
            tower,
            projector,
            prepared[case["case_id"]],
            case,
            "production",
        )
        if values["final"].dtype != mx.bfloat16:
            raise RuntimeError("Production reference is not BF16")
        case["production_final_sha256"] = array_hash(values["final"])
        save_case_tensors(
            output / "cases" / case["case_id"] / "production-bf16.safetensors",
            values,
        )
        print(f"cached BF16 {case['case_id']}", flush=True)

    del tower, projector
    gc.collect()
    mx.clear_cache()
    tower, projector, _, _ = load_model(mx.float32)
    for case in cases:
        inputs = prepared[case["case_id"]]
        inputs = {name: value.astype(mx.float32) for name, value in inputs.items()}
        if "positions" in prepared[case["case_id"]]:
            inputs["positions"] = prepared[case["case_id"]]["positions"]
        values = run_case(tower, projector, inputs, case, "production")
        if any(value.dtype != mx.float32 for value in values.values()):
            raise RuntimeError("FP32 reference contains a non-FP32 activation")
        case["fp32_final_sha256"] = array_hash(values["final"])
        save_case_tensors(
            output / "cases" / case["case_id"] / "reference-fp32.safetensors",
            values,
        )
        print(f"cached FP32 {case['case_id']}", flush=True)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_ID,
        "model_weight_note": "FP32 compute reference uses BF16-origin vision weights",
        "dataset": DATASET_ID,
        "dataset_split": DATASET_SPLIT,
        "dataset_fingerprint": dataset._fingerprint,
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_vlm_version": importlib.metadata.version("mlx-vlm"),
        "max_soft_tokens": MAX_SOFT_TOKENS,
        "device_info": mx.device_info(),
        "cases": cases,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "cases": len(cases)}, indent=2))


def ordered_bf16(bits: np.ndarray) -> np.ndarray:
    negative = (bits & np.uint16(0x8000)) != 0
    return np.where(
        negative,
        (~bits).astype(np.uint16),
        bits | np.uint16(0x8000),
    ).astype(np.int32)


def ulp_metrics(reference: mx.array, candidate: mx.array) -> dict:
    reference_bits = np.array(reference.view(mx.uint16), copy=True)
    candidate_bits = np.array(candidate.view(mx.uint16), copy=True)
    reference_float = np.array(reference.astype(mx.float32), copy=True)
    candidate_float = np.array(candidate.astype(mx.float32), copy=True)
    reference_ulp = reference_bits.copy()
    candidate_ulp = candidate_bits.copy()
    reference_ulp[(reference_ulp & np.uint16(0x7FFF)) == 0] = 0
    candidate_ulp[(candidate_ulp & np.uint16(0x7FFF)) == 0] = 0
    distance = np.abs(
        ordered_bf16(reference_ulp) - ordered_bf16(candidate_ulp)
    )
    reference_nonzero = reference_float != 0
    candidate_nonzero = candidate_float != 0
    sign_flip = (
        ((reference_bits ^ candidate_bits) & np.uint16(0x8000)) != 0
    ) & reference_nonzero & candidate_nonzero
    ordered = np.sort(distance.reshape(-1))
    p999_index = max(0, math.ceil(0.999 * ordered.size) - 1)
    return {
        "elements": int(distance.size),
        "fraction_ulp_le_1": float(np.mean(distance <= 1)),
        "p999_ulp": int(ordered[p999_index]),
        "max_ulp": int(distance.max(initial=0)),
        "nan_count": int(np.count_nonzero(np.isnan(candidate_float))),
        "inf_count": int(np.count_nonzero(np.isinf(candidate_float))),
        "sign_flips_nonzero": int(np.count_nonzero(sign_flip)),
    }


def relative_l2(value: np.ndarray, truth: np.ndarray) -> float:
    value = value.astype(np.float64)
    truth = truth.astype(np.float64)
    denominator = np.linalg.norm(truth.reshape(-1))
    error = np.linalg.norm((value - truth).reshape(-1))
    return float(error / denominator) if denominator else float(error)


def numerical_metrics(
    baseline: mx.array,
    candidate: mx.array,
    truth: mx.array,
) -> dict:
    ulp = ulp_metrics(baseline, candidate)
    baseline_np = np.array(baseline.astype(mx.float32), copy=True)
    candidate_np = np.array(candidate.astype(mx.float32), copy=True)
    truth_np = np.array(truth, copy=True)
    baseline_l2 = relative_l2(baseline_np, truth_np)
    candidate_l2 = relative_l2(candidate_np, truth_np)
    l2_ratio = candidate_l2 / baseline_l2 if baseline_l2 else (
        1.0 if candidate_l2 == 0 else math.inf
    )

    dot = np.sum(candidate_np * baseline_np, axis=-1, dtype=np.float64)
    candidate_norm = np.sqrt(
        np.sum(candidate_np * candidate_np, axis=-1, dtype=np.float64)
    )
    baseline_norm = np.sqrt(
        np.sum(baseline_np * baseline_np, axis=-1, dtype=np.float64)
    )
    denominator = candidate_norm * baseline_norm
    cosine = np.where(
        denominator == 0,
        np.where((candidate_norm == 0) & (baseline_norm == 0), 1.0, -math.inf),
        dot / denominator,
    )

    baseline_mean = baseline_np.astype(np.float64).mean(axis=(0, 1))
    candidate_mean = candidate_np.astype(np.float64).mean(axis=(0, 1))
    baseline_std = baseline_np.astype(np.float64).std(axis=(0, 1))
    candidate_std = candidate_np.astype(np.float64).std(axis=(0, 1))

    def drift(current: np.ndarray, reference: np.ndarray) -> np.ndarray:
        result = np.full(reference.shape, math.inf, dtype=np.float64)
        nonzero = reference != 0
        result[nonzero] = np.abs(current[nonzero] - reference[nonzero]) / np.abs(
            reference[nonzero]
        )
        result[~nonzero & (current == 0)] = 0.0
        return result

    mean_drift = drift(candidate_mean, baseline_mean)
    std_drift = drift(candidate_std, baseline_std)
    metrics = {
        **ulp,
        "baseline_relative_l2_to_fp32": baseline_l2,
        "candidate_relative_l2_to_fp32": candidate_l2,
        "candidate_to_baseline_fp32_distance_ratio": l2_ratio,
        "min_token_cosine": float(np.min(cosine)),
        "max_channel_mean_relative_drift": float(np.max(mean_drift)),
        "max_channel_std_relative_drift": float(np.max(std_drift)),
    }
    metrics["gate_a1"] = bool(
        metrics["fraction_ulp_le_1"] >= 0.999
        and metrics["max_ulp"] <= 2
        and metrics["nan_count"] == 0
        and metrics["inf_count"] == 0
        and metrics["sign_flips_nonzero"] == 0
    )
    metrics["gate_a2"] = bool(l2_ratio <= 1.05)
    metrics["gate_a3"] = bool(metrics["min_token_cosine"] >= 0.99999)
    metrics["gate_a4"] = bool(
        metrics["max_channel_mean_relative_drift"] <= 0.001
        and metrics["max_channel_std_relative_drift"] <= 0.001
    )
    metrics["tier_a_pass"] = bool(
        metrics["gate_a1"]
        and metrics["gate_a2"]
        and metrics["gate_a3"]
        and metrics["gate_a4"]
    )
    return metrics


def validate_references(
    references: Path,
    candidate: str,
    output: Path,
    allow_production_drift: bool = False,
) -> None:
    manifest = json.loads((references / "manifest.json").read_text())
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise RuntimeError("Unsupported reference schema")
    tower, projector, _, _ = load_model()
    for case in manifest["cases"]:
        case_dir = references / "cases" / case["case_id"]
        inputs = mx.load(str(case_dir / "input.safetensors"))
        control = run_case(tower, projector, inputs, case, "production")
        if (
            not allow_production_drift
            and array_hash(control["final"]) != case["production_final_sha256"]
        ):
            raise RuntimeError(f"Production reference drifted for {case['case_id']}")
    if candidate == "qkv_epilogue":
        fuse_gemma4_qkv_epilogue(tower)
    elif candidate == "rope_layout":
        fuse_gemma4_rope_layout(tower)
    elif candidate == "rope_output_layout":
        fuse_gemma4_rope_and_output_layout(tower)
    elif candidate == "post_reduction_epilogue":
        fuse_gemma4_post_reduction_epilogue(tower)
    elif candidate == "shapeless_rope":
        optimize_gemma4_shapeless_rope(tower)

    case_results = []
    for case in manifest["cases"]:
        case_dir = references / "cases" / case["case_id"]
        inputs = mx.load(str(case_dir / "input.safetensors"))
        baseline = mx.load(str(case_dir / "production-bf16.safetensors"))
        truth = mx.load(str(case_dir / "reference-fp32.safetensors"))
        values = run_case(tower, projector, inputs, case, candidate)
        final_metrics = numerical_metrics(
            baseline["final"],
            values["final"],
            truth["final"],
        )
        layerwise = []
        for index in range(16):
            name = f"block_{index:02d}_post"
            layerwise.append({"block": index, **ulp_metrics(baseline[name], values[name])})
        case_result = {
            "case_id": case["case_id"],
            "padded": case["padded"],
            "final": final_metrics,
            "layerwise": layerwise,
        }
        case_results.append(case_result)
        print(
            f"{case['case_id']}: {'PASS' if final_metrics['tier_a_pass'] else 'FAIL'} "
            f"ulp<=1={final_metrics['fraction_ulp_le_1']:.6f} "
            f"max={final_metrics['max_ulp']} l2_ratio={final_metrics['candidate_to_baseline_fp32_distance_ratio']:.6f}",
            flush=True,
        )
    result = {
        "metadata": {
            "candidate": candidate,
            "references": str(references),
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "allowed_production_drift": allow_production_drift,
        },
        "tier_a_pass": all(case["final"]["tier_a_pass"] for case in case_results),
        "cases": case_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"tier_a_pass": result["tier_a_pass"], "output": str(output)}, indent=2))
    if not result["tier_a_pass"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--force", action="store_true")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--references", type=Path, required=True)
    validate.add_argument(
        "--candidate",
        choices=(
            "production",
            "reshape_pool",
            "exact_pool",
            "qkv_epilogue",
            "rope_layout",
            "rope_output_layout",
            "post_reduction_epilogue",
            "shapeless_rope",
        ),
        required=True,
    )
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument("--allow-production-drift", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        build_references(args.output, args.force)
    else:
        validate_references(
            args.references,
            args.candidate,
            args.output,
            args.allow_production_drift,
        )


if __name__ == "__main__":
    main()
