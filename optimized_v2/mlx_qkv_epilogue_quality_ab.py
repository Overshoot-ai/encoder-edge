"""Paired ChartQA quality gate for the reassociated fused-QKV epilogue."""

import argparse
import gc
import http.client
import importlib.metadata
import json
import math
import resource
import statistics
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import mlx.core as mx
import numpy as np
from mlx_vlm import load

from .mlx_mixed_shape_benchmark import output_difference
from .mlx_vision_optimizations import (
    fuse_gemma4_qkv_epilogue,
    fuse_gemma4_rope_layout,
    make_segmented_gemma4_encoder,
    optimize_gemma4_positions,
)
from .overshoot_eval import CHARTQA_INSTRUCTIONS, chartqa_scores, percentile
from .protocol import CONTENT_TYPE, encode_raw_request


MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
DATASET_ID = "HuggingFaceM4/ChartQA"
DEFAULT_OUTPUT = Path("benchmark-results/mlx-roofline/qkv-quality-gate")
DEFAULT_CORPUS = Path("benchmark-results/mlx-roofline/mixed-shape-corpus")


def memory_snapshot() -> dict:
    return {
        "active_bytes": mx.get_active_memory(),
        "cache_bytes": mx.get_cache_memory(),
        "peak_bytes": mx.get_peak_memory(),
        "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def timing_summary(values: list[float]) -> dict:
    return {
        "rounds": len(values),
        "mean_ms": statistics.mean(values),
        "p50_ms": percentile(values, 0.5),
        "p90_ms": percentile(values, 0.9),
        "minimum_ms": min(values),
        "maximum_ms": max(values),
        "raw_ms": values,
    }


def time_encoder(encode, pixels: mx.array, warmups: int, rounds: int) -> dict:
    for _ in range(warmups):
        value = encode(pixels)
        mx.eval(value)
        mx.synchronize()
    mx.reset_peak_memory()
    before = memory_snapshot()
    elapsed = []
    for _ in range(rounds):
        started = time.perf_counter()
        value = encode(pixels)
        mx.eval(value)
        mx.synchronize()
        elapsed.append((time.perf_counter() - started) * 1000)
    return {
        **timing_summary(elapsed),
        "memory_before": before,
        "memory_after": memory_snapshot(),
    }


def load_encoder(variant: str):
    model, _ = load(MODEL_ID)
    tower = model.vision_tower
    optimize_gemma4_positions(tower)
    if variant == "baseline":
        fuse_gemma4_rope_layout(tower)
        encode = make_segmented_gemma4_encoder(
            tower,
            projector=None,
            segment_size=3,
            evaluate_segments=True,
        )
    elif variant == "candidate":
        fuse_gemma4_qkv_epilogue(tower)
        encode = make_qkv_segmented_encoder(tower, segment_size=3)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    del model
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    return tower, encode


def make_qkv_segmented_encoder(tower, segment_size: int):
    """Segment the candidate while moving its Python RoPE cache outside compile."""
    patch = mx.compile(
        lambda pixels, positions, padding: tower.patch_embedder(
            pixels, positions, padding
        )
    )
    attentions = [layer.self_attn for layer in tower.encoder.layers]
    original_rope = attentions[0]._rope

    def accept_constants(self, positions):
        if isinstance(positions, tuple):
            return positions
        return self._original_rope(positions)

    for attention in attentions:
        attention._original_rope = attention._rope
        attention._rope = types.MethodType(accept_constants, attention)

    segments = []
    for start in range(0, len(tower.encoder.layers), segment_size):
        layers = tuple(tower.encoder.layers[start : start + segment_size])

        def run_segment(hidden, cosine, sine, current_layers=layers):
            for layer in current_layers:
                hidden = layer(hidden, (cosine, sine), None)
            return hidden

        segments.append(mx.compile(run_segment))

    def finish(hidden, positions, padding):
        output_length = hidden.shape[1] // (tower.pooling_kernel_size**2)
        hidden, _ = tower.pooler(
            hidden, positions, padding, output_length=output_length
        )
        if tower.config.standardize:
            hidden = (hidden - tower.std_bias) * tower.std_scale
        return hidden

    finish = mx.compile(finish)
    rope_cache = {}

    def encode(pixels):
        batch, _, height, width = pixels.shape
        if batch != 1:
            raise ValueError("QKV epilogue quality gate requires batch size 1")
        patch_count = (height // tower.patch_size) * (width // tower.patch_size)
        positions, padding, _ = tower._patch_positions_single(
            height, width, max_patches=patch_count
        )
        positions = mx.array(np.expand_dims(positions, 0))
        padding = mx.array(np.expand_dims(padding, 0))
        key = (height, width)
        if key not in rope_cache:
            rope_cache[key] = original_rope(positions)
        cosine, sine = rope_cache[key]
        hidden = patch(pixels, positions, padding)
        for segment in segments:
            hidden = segment(hidden, cosine, sine)
            mx.eval(hidden)
            mx.synchronize()
        return finish(hidden, positions, padding)

    return encode


def encode_cases(encode, cases: list[dict], variant: str) -> tuple[list[mx.array], list[dict]]:
    features = []
    records = []
    for offset, case in enumerate(cases):
        mx.reset_peak_memory()
        before = memory_snapshot()
        started = time.perf_counter()
        value = encode(case["pixels"])
        mx.eval(value)
        mx.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        if value.ndim != 3 or value.shape[0] != 1 or value.dtype != mx.bfloat16:
            raise RuntimeError(f"Unexpected {variant} feature tensor: {value.shape} {value.dtype}")
        features.append(value)
        records.append(
            {
                "elapsed_ms": elapsed_ms,
                "shape": list(value.shape),
                "visual_tokens": value.shape[1],
                "memory_before": before,
                "memory_after": memory_snapshot(),
            }
        )
        print(
            f"{variant} feature {offset + 1}/{len(cases)}: "
            f"{elapsed_ms:.1f} ms, tokens={value.shape[1]}",
            flush=True,
        )
    return features, records


class Gateway:
    def __init__(self, server: str, model: str):
        parsed = urlsplit(server)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("--server must be an HTTP URL")
        self.host = parsed.hostname
        self.port = parsed.port or 80
        self.path = parsed.path.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.connection = http.client.HTTPConnection(self.host, self.port, timeout=300)

    def close(self) -> None:
        self.connection.close()

    def complete(self, features: mx.array, prompt: str, max_tokens: int) -> tuple[str, dict]:
        tensor = features[0]
        tensor_bytes = np.array(tensor.view(mx.uint16), copy=True).tobytes(order="C")
        payload = encode_raw_request(
            tensor_bytes,
            tuple(tensor.shape),
            prompt,
            self.model,
            max_tokens,
        )
        started = time.perf_counter()
        self.connection.request(
            "POST",
            self.path,
            body=payload,
            headers={"Content-Type": CONTENT_TYPE, "Accept": "text/event-stream"},
        )
        response = self.connection.getresponse()
        if response.status != 200:
            error = response.read().decode(errors="replace")
            raise RuntimeError(f"Gateway returned HTTP {response.status}: {error}")
        fragments = []
        usage = None
        first_token_at = None
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            usage = event.get("usage") or usage
            choices = event.get("choices", [])
            text = choices[0].get("delta", {}).get("content") if choices else None
            if text:
                first_token_at = first_token_at or time.perf_counter()
                fragments.append(text)
        response.read()
        finished = time.perf_counter()

        def header(name: str):
            value = response.getheader(name)
            return float(value) if value is not None else None

        return "".join(fragments), {
            "request_bytes": len(payload),
            "tensor_bytes": len(tensor_bytes),
            "visual_tokens": tensor.shape[0],
            "hidden_size": tensor.shape[1],
            "remote_ttft_ms": ((first_token_at or finished) - started) * 1000,
            "remote_e2e_ms": (finished - started) * 1000,
            "gateway_ttft_ms": header("X-Gateway-TTFT-Ms"),
            "gateway_prepare_ms": header("X-Gateway-Prepare-Ms"),
            "vllm_ttft_ms": header("X-vLLM-TTFT-Ms"),
            "usage": usage,
        }


def feature_summary(records: list[dict]) -> dict:
    return {
        "cases": len(records),
        "bit_identical_cases": sum(record["bit_identical"] for record in records),
        "mean_relative_l2_difference": statistics.mean(
            record["relative_l2_difference"] for record in records
        ),
        "maximum_relative_l2_difference": max(
            record["relative_l2_difference"] for record in records
        ),
        "minimum_token_cosine": min(record["minimum_token_cosine"] for record in records),
        "minimum_fraction_ulp_le_1": min(record["fraction_ulp_le_1"] for record in records),
        "maximum_absolute_difference": max(
            record["maximum_absolute_difference"] for record in records
        ),
        "nan_count": sum(record["nan_count"] for record in records),
        "inf_count": sum(record["inf_count"] for record in records),
    }


def quality_summary(records: list[dict]) -> dict:
    arms = {}
    for variant in ("baseline", "candidate"):
        arms[variant] = {
            "exact_match": statistics.mean(record[variant]["exact_match"] for record in records),
            "relaxed_accuracy": statistics.mean(
                record[variant]["relaxed_accuracy"] for record in records
            ),
            "anywhere_accuracy": statistics.mean(
                record[variant]["anywhere_accuracy"] for record in records
            ),
            "completion_tokens": sum(
                (record[variant]["gateway"].get("usage") or {}).get("completion_tokens", 0)
                for record in records
            ),
        }
    exact_agreement = statistics.mean(record["exact_generation_agreement"] for record in records)
    parsed_agreement = statistics.mean(record["parsed_answer_agreement"] for record in records)

    def paired_outcomes(metric: str) -> dict:
        outcomes = {
            "both_correct": 0,
            "baseline_only": 0,
            "candidate_only": 0,
            "both_incorrect": 0,
        }
        for record in records:
            baseline_correct = bool(record["baseline"][metric])
            candidate_correct = bool(record["candidate"][metric])
            if baseline_correct and candidate_correct:
                outcomes["both_correct"] += 1
            elif baseline_correct:
                outcomes["baseline_only"] += 1
            elif candidate_correct:
                outcomes["candidate_only"] += 1
            else:
                outcomes["both_incorrect"] += 1
        return outcomes

    gate_pass = bool(
        arms["candidate"]["relaxed_accuracy"] >= arms["baseline"]["relaxed_accuracy"]
        and paired_outcomes("relaxed_accuracy")["candidate_only"]
        >= paired_outcomes("relaxed_accuracy")["baseline_only"]
    )
    return {
        "arms": arms,
        "exact_generation_agreement": exact_agreement,
        "parsed_answer_agreement": parsed_agreement,
        "paired_relaxed_outcomes": paired_outcomes("relaxed_accuracy"),
        "paired_exact_outcomes": paired_outcomes("exact_match"),
        "relaxed_accuracy_delta": (
            arms["candidate"]["relaxed_accuracy"] - arms["baseline"]["relaxed_accuracy"]
        ),
        "gate_definition": "no relaxed-accuracy regression and nonnegative paired correctness",
        "quality_gate_pass": gate_pass,
        "promotion_decision": "promote" if gate_pass else "do_not_promote",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8002")
    parser.add_argument("--model", default="gemma-4-e4b-optimized")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--max-soft-tokens", type=int, default=273)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.limit < 1 or args.rounds < 10:
        raise ValueError("--limit must be positive and --rounds must be at least 10")
    args.output.mkdir(parents=True, exist_ok=True)

    corpus_manifest = json.loads((args.corpus / "manifest.json").read_text())
    if corpus_manifest["max_soft_tokens"] != args.max_soft_tokens:
        raise ValueError(
            "Cached corpus soft-token budget does not match --max-soft-tokens"
        )
    baseline_tower, baseline_encode = load_encoder("baseline")
    cases = []
    for cached_case in corpus_manifest["cases"][: args.limit]:
        prompt = CHARTQA_INSTRUCTIONS.format(question=cached_case["query"])
        pixels = mx.load(
            str(args.corpus / "cases" / cached_case["case_id"] / "input.safetensors")
        )["pixels"]
        mx.eval(pixels)
        cases.append(
            {
                "case_id": cached_case["case_id"],
                "index": cached_case["dataset_index"],
                "query": cached_case["query"],
                "targets": cached_case["targets"],
                "prompt": prompt,
                "pixels": pixels,
                "pixel_shape": list(pixels.shape),
            }
        )

    baseline_features, baseline_encodes = encode_cases(baseline_encode, cases, "baseline")
    baseline_timing = time_encoder(
        baseline_encode, cases[0]["pixels"], args.warmups, args.rounds
    )
    baseline_retained_memory = memory_snapshot()
    del baseline_encode, baseline_tower
    gc.collect()
    mx.clear_cache()
    mx.synchronize()

    candidate_tower, candidate_encode = load_encoder("candidate")
    candidate_features, candidate_encodes = encode_cases(candidate_encode, cases, "candidate")
    candidate_timing = time_encoder(
        candidate_encode, cases[0]["pixels"], args.warmups, args.rounds
    )

    differences = []
    for baseline, candidate in zip(baseline_features, candidate_features):
        metrics = output_difference(baseline, candidate)
        metrics["bit_identical"] = bool(metrics["differing_values"] == 0)
        differences.append(metrics)

    raw_path = args.output / "paired_results.jsonl"
    quality_records = (
        [json.loads(line) for line in raw_path.read_text().splitlines()]
        if raw_path.exists()
        else []
    )
    if len(quality_records) > len(cases):
        raise ValueError("Existing paired results exceed the requested case limit")
    for offset, record in enumerate(quality_records):
        if record.get("case_id") != cases[offset]["case_id"]:
            raise ValueError(
                f"Existing result {offset} is not the expected corpus prefix: "
                f"{record.get('case_id')} != {cases[offset]['case_id']}"
            )
    resumed_records = len(quality_records)
    if resumed_records:
        print(f"resuming after {resumed_records} paired quality records", flush=True)
    gateway = Gateway(args.server, args.model)
    try:
        for offset, case in enumerate(cases[resumed_records:], start=resumed_records):
            arm_results = {}
            for variant, features in (
                ("baseline", baseline_features[offset]),
                ("candidate", candidate_features[offset]),
            ):
                generation, gateway_metrics = gateway.complete(
                    features, case["prompt"], args.max_tokens
                )
                arm_results[variant] = {
                    "generation": generation,
                    **chartqa_scores(generation, case["targets"]),
                    "gateway": gateway_metrics,
                    "encoder": (
                        baseline_encodes[offset]
                        if variant == "baseline"
                        else candidate_encodes[offset]
                    ),
                }
            record = {
                "index": case["index"],
                "case_id": case["case_id"],
                "query": case["query"],
                "targets": case["targets"],
                "pixel_shape": case["pixel_shape"],
                "feature_difference": differences[offset],
                **arm_results,
                "exact_generation_agreement": bool(
                    arm_results["baseline"]["generation"]
                    == arm_results["candidate"]["generation"]
                ),
                "parsed_answer_agreement": bool(
                    arm_results["baseline"]["parsed_answer"].casefold()
                    == arm_results["candidate"]["parsed_answer"].casefold()
                ),
            }
            quality_records.append(record)
            with raw_path.open("a") as output:
                output.write(json.dumps(record) + "\n")
            print(
                f"quality {offset + 1}/{len(cases)}: "
                f"agreement={record['exact_generation_agreement']} "
                f"relaxed={arm_results['baseline']['relaxed_accuracy']:.0f}/"
                f"{arm_results['candidate']['relaxed_accuracy']:.0f}",
                flush=True,
            )
    finally:
        gateway.close()

    speedup = baseline_timing["p50_ms"] / candidate_timing["p50_ms"]
    result = {
        "metadata": {
            "benchmark": "fused_qkv_epilogue_chartqa_quality_ab",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_checkpoint": MODEL_ID,
            "decoder_model": args.model,
            "dataset": DATASET_ID,
            "dataset_split": "test",
            "dataset_fingerprint": corpus_manifest["dataset_fingerprint"],
            "corpus": str(args.corpus),
            "server": args.server,
            "cases": len(cases),
            "resumed_quality_records": resumed_records,
            "rounds_per_arm": args.rounds,
            "warmups_per_arm": args.warmups,
            "max_soft_tokens": args.max_soft_tokens,
            "max_generation_tokens": args.max_tokens,
            "temperature": 0,
            "segment_size": 3,
            "evaluate_segments": True,
            "projector_location": "H200",
            "baseline_graph": "optimized positions + exact RoPE/layout + segmented size3",
            "candidate_graph": "optimized positions + reassociated fused QKV epilogue + segmented size3",
            "model_state_strategy": "cache baseline features, release tower, reload identical weights for candidate",
            "mlx_version": importlib.metadata.version("mlx"),
            "mlx_vlm_version": importlib.metadata.version("mlx-vlm"),
            "device": str(mx.default_device()),
            "device_info": mx.device_info(),
        },
        "encoder_timing": {
            "input_pixel_shape": cases[0]["pixel_shape"],
            "baseline": baseline_timing,
            "candidate": candidate_timing,
            "candidate_p50_speedup": speedup,
            "candidate_p50_latency_change_percent": (1.0 / speedup - 1.0) * 100,
        },
        "memory": {
            "baseline_features_retained_before_candidate_load": baseline_retained_memory,
            "final": memory_snapshot(),
        },
        "numerical_features": feature_summary(differences),
        "quality": quality_summary(quality_records),
        "artifacts": {
            "summary": str(args.output / "summary.json"),
            "paired_results": str(raw_path),
        },
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
