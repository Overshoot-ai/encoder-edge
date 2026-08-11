import argparse
import gc
import hashlib
import json
import math
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

from .mlx_vision_optimizations import (
    encode_gemma4_shapeless_batch1,
    encode_gemma4_shapeless_hidden,
    encode_gemma4_unpadded_batch1,
    fuse_gemma4_rope_layout,
    gemma4_rope_constants,
    gemma4_unpadded_inputs,
    make_segmented_gemma4_encoder,
    optimize_gemma4_positions,
    optimize_gemma4_shapeless_positions,
    optimize_gemma4_shapeless_rope,
)
from .mlx_tier_a import ulp_metrics


MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
DATASET_ID = "HuggingFaceM4/ChartQA"
MAX_SOFT_TOKENS = 273


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize(values: list[float]) -> dict:
    return {
        "mean_ms": statistics.mean(values),
        "p50_ms": percentile(values, 0.5),
        "p90_ms": percentile(values, 0.9),
        "p99_ms": percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def output_difference(reference: mx.array, candidate: mx.array) -> dict:
    metrics = ulp_metrics(reference, candidate)
    reference_np = np.array(reference.astype(mx.float32), copy=True)
    candidate_np = np.array(candidate.astype(mx.float32), copy=True)
    absolute = np.abs(reference_np - candidate_np)
    dot = np.sum(reference_np * candidate_np, axis=-1, dtype=np.float64)
    reference_norm = np.sqrt(
        np.sum(reference_np * reference_np, axis=-1, dtype=np.float64)
    )
    candidate_norm = np.sqrt(
        np.sum(candidate_np * candidate_np, axis=-1, dtype=np.float64)
    )
    denominator = reference_norm * candidate_norm
    cosine = np.where(
        denominator == 0,
        np.where((reference_norm == 0) & (candidate_norm == 0), 1.0, -math.inf),
        dot / denominator,
    )
    reference_l2 = np.linalg.norm(reference_np.reshape(-1).astype(np.float64))
    error_l2 = np.linalg.norm(
        (candidate_np - reference_np).reshape(-1).astype(np.float64)
    )
    return {
        **metrics,
        "differing_values": int(
            np.count_nonzero(
                np.array(reference.view(mx.uint16), copy=True)
                != np.array(candidate.view(mx.uint16), copy=True)
            )
        ),
        "mean_absolute_difference": float(absolute.mean()),
        "maximum_absolute_difference": float(absolute.max(initial=0)),
        "relative_l2_difference": float(error_l2 / reference_l2),
        "minimum_token_cosine": float(cosine.min(initial=1.0)),
    }


def image_hash(image) -> str:
    image = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(image.width.to_bytes(4, "little"))
    digest.update(image.height.to_bytes(4, "little"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def output_bits(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.array(value.view(mx.uint16), copy=True)


def thermal_state() -> str:
    result = subprocess.run(
        ["pmset", "-g", "therm"],
        check=False,
        capture_output=True,
        text=True,
    )
    return (result.stdout + result.stderr).strip()


def system_vm_state() -> dict:
    pressure = subprocess.run(
        ["memory_pressure", "-Q"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    swap = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    vm_stat = subprocess.run(
        ["vm_stat"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    free_match = re.search(r"memory free percentage:\s*(\d+)%", pressure)
    swap_match = re.search(r"used =\s*([0-9.]+)M", swap)
    counters = {}
    for name, value in re.findall(
        r"^(Pageins|Pageouts|Swapins|Swapouts):\s*(\d+)\.",
        vm_stat,
        flags=re.MULTILINE,
    ):
        counters[name.lower()] = int(value)
    return {
        "free_percent": int(free_match.group(1)) if free_match else None,
        "swap_used_mb": float(swap_match.group(1)) if swap_match else None,
        **counters,
    }


def prepare_corpus(output: Path, count: int, force: bool) -> None:
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(f"Mixed-shape manifest exists: {manifest_path}")
    output.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, split="test")
    model, processor = load(MODEL_ID)
    processor.image_processor.max_soft_tokens = MAX_SOFT_TOKENS
    prompt = apply_chat_template(processor, model.config, "x", num_images=1)
    tower = model.vision_tower
    cases = []
    hashes = set()
    for index in range(len(dataset)):
        sample = dataset[index]
        image = sample["image"].convert("RGB")
        digest = image_hash(image)
        if digest in hashes:
            continue
        pixels = prepare_inputs(
            processor,
            images=[image],
            prompts=prompt,
            add_special_tokens=False,
        )["pixel_values"]
        case_id = f"chartqa-{index:04d}"
        case_dir = output / "cases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        image.save(case_dir / "source.png", format="PNG")
        mx.save_safetensors(str(case_dir / "input.safetensors"), {"pixels": pixels})
        shape = list(pixels.shape)
        cases.append(
            {
                "case_id": case_id,
                "dataset_index": index,
                "query": sample["query"],
                "targets": sample["label"],
                "image_sha256": digest,
                "source_size": list(image.size),
                "pixel_shape": shape,
                "patch_grid": [
                    shape[-2] // tower.patch_size,
                    shape[-1] // tower.patch_size,
                ],
            }
        )
        hashes.add(digest)
        print(f"prepared {len(cases)}/{count}: {case_id} {shape}", flush=True)
        if len(cases) == count:
            break
    if len(cases) != count:
        raise RuntimeError(f"Only found {len(cases)} unique ChartQA images")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_ID,
        "dataset_split": "test",
        "dataset_fingerprint": dataset._fingerprint,
        "model": MODEL_ID,
        "max_soft_tokens": MAX_SOFT_TOKENS,
        "case_count": len(cases),
        "unique_pixel_shapes": len({tuple(case["pixel_shape"]) for case in cases}),
        "cases": cases,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("case_count", "unique_pixel_shapes")}, indent=2))


def run_benchmark(
    corpus: Path,
    variant: str,
    cooldown: int,
    output: Path,
    case_count: int | None = None,
    wired_limit_override: int | None = None,
    cache_limit: int | None = None,
    prewarm_all_shapes: bool = False,
    sample_system_vm: bool = False,
) -> None:
    manifest = json.loads((corpus / "manifest.json").read_text())
    cases = manifest["cases"][:case_count]
    wired_limit = (
        2 * 1024**3
        if variant
        in (
            "wired_status_quo",
            "wired_rope_layout",
            "wired_rope_layout_segment3",
            "wired_shapeless",
        )
        else 0
    )
    if wired_limit_override is not None:
        wired_limit = wired_limit_override
    if wired_limit:
        mx.set_wired_limit(wired_limit)
    model, _ = load(MODEL_ID)
    tower = model.vision_tower
    projector = model.embed_vision
    optimize_gemma4_positions(tower)
    if variant in ("wired_rope_layout", "wired_rope_layout_segment3"):
        fuse_gemma4_rope_layout(tower)
    elif variant in ("shapeless", "wired_shapeless"):
        optimize_gemma4_shapeless_positions(tower)
        optimize_gemma4_shapeless_rope(tower)
    del model
    gc.collect()
    mx.clear_cache()
    mx.synchronize()

    def make_compiled(shapeless=False):
        if shapeless:
            return mx.compile(
                lambda hidden, cosine, sine, pool_weights: (
                    encode_gemma4_shapeless_hidden(
                        tower,
                        projector,
                        hidden,
                        cosine,
                        sine,
                        pool_weights,
                    )
                ),
                shapeless=True,
            )
        return mx.compile(
            lambda value: encode_gemma4_unpadded_batch1(tower, projector, value),
            shapeless=shapeless,
        )

    if variant in (
        "status_quo",
        "wired_status_quo",
        "wired_rope_layout",
        "wired_rope_layout_segment3",
    ):
        if variant == "wired_rope_layout_segment3":
            shared = make_segmented_gemma4_encoder(
                tower,
                projector,
                3,
                evaluate_segments=True,
            )
        else:
            shared = make_compiled()
        get_function = lambda shape: shared
        function_count = 1
    elif variant in ("shapeless", "wired_shapeless"):
        shared = make_compiled(shapeless=True)
        auxiliaries = {}

        def get_function(shape):
            key = tuple(shape)
            if key not in auxiliaries:
                positions, padding, pool_weights = gemma4_unpadded_inputs(
                    tower,
                    shape[-2],
                    shape[-1],
                )
                cosine, sine = gemma4_rope_constants(tower, positions)
                patch = mx.compile(
                    lambda value, current_positions=positions,
                    current_padding=padding: tower.patch_embedder(
                        value,
                        current_positions,
                        current_padding,
                    )
                )
                auxiliaries[key] = (
                    cosine,
                    sine,
                    mx.squeeze(pool_weights, axis=0),
                    patch,
                )
            cosine, sine, pool_weights, patch = auxiliaries[key]
            return lambda value: shared(
                mx.squeeze(patch(value), axis=0),
                cosine,
                sine,
                pool_weights,
            )

        function_count = 1
    elif variant in ("shape_cache", "aspect_cache"):
        functions = {}

        def get_function(shape):
            if variant == "shape_cache":
                key = tuple(shape)
            else:
                ratio = shape[-1] / shape[-2]
                key = next(
                    index
                    for index, threshold in enumerate(
                        (0.65, 0.85, 1.15, 1.5, math.inf)
                    )
                    if ratio < threshold
                )
            if key not in functions:
                functions[key] = make_compiled()
            return functions[key]

        function_count = (
            manifest["unique_pixel_shapes"] if variant == "shape_cache" else 5
        )
    else:
        shared = lambda value: encode_gemma4_unpadded_batch1(
            tower, projector, value
        )
        get_function = lambda shape: shared
        function_count = 0

    representatives = {}
    prewarm_cases = manifest["cases"] if prewarm_all_shapes else cases
    for case in prewarm_cases:
        representatives.setdefault(tuple(case["pixel_shape"]), case)
    prewarm = []
    for shape, case in representatives.items():
        print(f"prewarm {list(shape)}", flush=True)
        pixels = mx.load(
            str(corpus / "cases" / case["case_id"] / "input.safetensors")
        )["pixels"]
        mx.eval(pixels)
        started = time.perf_counter()
        value = get_function(shape)(pixels)
        mx.eval(value)
        mx.synchronize()
        prewarm.append(
            {
                "pixel_shape": list(shape),
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            }
        )
        del pixels, value
        gc.collect()
    if cache_limit is not None:
        mx.set_cache_limit(cache_limit)
        mx.eval(mx.zeros((1,)))
        mx.synchronize()
    thermal_before_cooldown = thermal_state()
    memory_before_cooldown = {
        "active_bytes": mx.get_active_memory(),
        "cache_bytes": mx.get_cache_memory(),
    }
    vm_before_cooldown = system_vm_state() if sample_system_vm else None
    time.sleep(cooldown)
    thermal_before_timing = thermal_state()
    memory_before_timing = {
        "active_bytes": mx.get_active_memory(),
        "cache_bytes": mx.get_cache_memory(),
    }
    vm_before_timing = system_vm_state() if sample_system_vm else None

    records = []
    seen_shapes = set()
    mismatches = 0
    reference_dir = corpus / "production-outputs"
    if variant == "status_quo":
        reference_dir.mkdir(parents=True, exist_ok=True)
    for stream_index, case in enumerate(cases):
        input_path = corpus / "cases" / case["case_id"] / "input.safetensors"
        pixels = mx.load(str(input_path))["pixels"]
        mx.eval(pixels)
        shape_key = tuple(case["pixel_shape"])
        active_before_bytes = mx.get_active_memory()
        cache_before_bytes = mx.get_cache_memory()
        mx.reset_peak_memory()
        started = time.perf_counter()
        value = get_function(case["pixel_shape"])(pixels)
        mx.eval(value)
        mx.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        active_after_bytes = mx.get_active_memory()
        cache_after_bytes = mx.get_cache_memory()
        encode_peak_memory_bytes = mx.get_peak_memory()
        vm_after = system_vm_state() if sample_system_vm else None
        bits = output_bits(value)
        reference_path = reference_dir / f"{case['case_id']}.safetensors"
        if variant == "status_quo":
            mx.save_safetensors(str(reference_path), {"final": value})
            exact = True
        else:
            reference = mx.load(str(reference_path))["final"]
            exact = bool(np.array_equal(bits, output_bits(reference)))
            mismatches += int(not exact)
        difference = None if variant == "status_quo" else output_difference(reference, value)
        records.append(
            {
                "stream_index": stream_index,
                "case_id": case["case_id"],
                "pixel_shape": case["pixel_shape"],
                "first_seen_shape_in_timed_stream": shape_key not in seen_shapes,
                "elapsed_ms": elapsed_ms,
                "active_before_bytes": active_before_bytes,
                "cache_before_bytes": cache_before_bytes,
                "active_after_bytes": active_after_bytes,
                "cache_after_bytes": cache_after_bytes,
                "encode_peak_memory_bytes": encode_peak_memory_bytes,
                "system_vm_after": vm_after,
                "output_shape": list(value.shape),
                "bit_identical_to_status_quo": exact,
                "difference": difference,
            }
        )
        seen_shapes.add(shape_key)
        print(
            f"case {stream_index + 1}/{len(cases)} {case['case_id']} "
            f"{elapsed_ms:.1f} ms exact={exact}",
            flush=True,
        )
        del pixels, value
        gc.collect()
    thermal_after_timing = thermal_state()
    vm_after_timing = system_vm_state() if sample_system_vm else None
    latencies = [record["elapsed_ms"] for record in records]
    summary = summarize(latencies)
    result = {
        "metadata": {
            "variant": variant,
            "model": MODEL_ID,
            "device": str(mx.default_device()),
            "device_info": mx.device_info(),
            "case_count": len(records),
            "unique_pixel_shapes": manifest["unique_pixel_shapes"],
            "compiled_callable_count": function_count,
            "wired_limit_bytes": wired_limit,
            "cooldown_seconds": cooldown,
            "prewarmed_all_shapes": prewarm_all_shapes,
            "sampled_system_vm_per_encode": sample_system_vm,
            "cache_limit_bytes": cache_limit,
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "thermal_before_cooldown": thermal_before_cooldown,
            "thermal_before_timing": thermal_before_timing,
            "thermal_after_timing": thermal_after_timing,
            "memory_before_cooldown": memory_before_cooldown,
            "memory_before_timing": memory_before_timing,
            "vm_before_cooldown": vm_before_cooldown,
            "vm_before_timing": vm_before_timing,
            "vm_after_timing": vm_after_timing,
        },
        "summary": {
            **summary,
            "encodes_over_1s": sum(value > 1000 for value in latencies),
            "encodes_over_2s": sum(value > 2000 for value in latencies),
            "bit_mismatch_cases": mismatches,
            "peak_memory_bytes": max(
                record["encode_peak_memory_bytes"] for record in records
            ),
        },
        "prewarm": prewarm,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"metadata": result["metadata"], "summary": result["summary"]}, indent=2))
    if mismatches:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--count", type=int, default=100)
    prepare.add_argument("--force", action="store_true")
    run = subparsers.add_parser("run")
    run.add_argument("--corpus", type=Path, required=True)
    run.add_argument(
        "--variant",
        choices=(
            "status_quo",
            "shapeless",
            "shape_cache",
            "aspect_cache",
            "wired_status_quo",
            "wired_rope_layout",
            "wired_rope_layout_segment3",
            "wired_shapeless",
            "eager",
        ),
        required=True,
    )
    run.add_argument("--cooldown", type=int, default=90)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--case-count", type=int)
    run.add_argument("--wired-limit", type=int)
    run.add_argument("--cache-limit", type=int)
    run.add_argument("--prewarm-all-shapes", action="store_true")
    run.add_argument("--sample-system-vm", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_corpus(args.output, args.count, args.force)
    else:
        run_benchmark(
            args.corpus,
            args.variant,
            args.cooldown,
            args.output,
            args.case_count,
            args.wired_limit,
            args.cache_limit,
            args.prewarm_all_shapes,
            args.sample_system_vm,
        )


if __name__ == "__main__":
    main()
