import argparse
import gc
import json
import math
import statistics
import time
import uuid
from pathlib import Path

import mlx.core as mx
from PIL import Image, ImageOps
from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "p50": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "min": min(values),
        "max": max(values),
    }


def synchronize(array=None) -> None:
    if array is not None:
        mx.eval(array)
    mx.synchronize()


def make_prompt(processor, config, question: str) -> str:
    return apply_chat_template(
        processor,
        config,
        question,
        num_images=1,
    )


def prepare_image(processor, image: Image.Image, prompt: str) -> dict:
    return prepare_inputs(
        processor,
        images=[image],
        prompts=prompt,
        add_special_tokens=False,
    )


def run_full(
    model,
    processor,
    image: Image.Image,
    question: str,
    rounds: int,
    max_tokens: int,
) -> tuple[list[dict], str]:
    def generate(index: int) -> tuple[dict, str]:
        prompt = make_prompt(
            processor,
            model.config,
            f"{question} nonce={uuid.uuid4()} rep={index}",
        )
        started = time.perf_counter()
        first_token = None
        final = None
        text = []
        for result in stream_generate(
            model,
            processor,
            prompt,
            image=[image],
            max_tokens=max_tokens,
            temperature=0,
            verbose=False,
        ):
            now = time.perf_counter()
            if first_token is None:
                first_token = now
            final = result
            text.append(result.text)
        finished = time.perf_counter()
        if first_token is None or final is None:
            raise RuntimeError("MLX-VLM did not generate a token")
        return (
            {
                "rep": index,
                "pipeline_ttft_ms": (first_token - started) * 1000,
                "pipeline_e2e_ms": (finished - started) * 1000,
                "prompt_tokens": final.prompt_tokens,
                "generation_tokens": final.generation_tokens,
                "prompt_tps": final.prompt_tps,
                "generation_tps": final.generation_tps,
                "peak_memory_gb": final.peak_memory,
            },
            "".join(text),
        )

    generate(-1)
    records = []
    answer = ""
    for index in range(rounds):
        record, answer = generate(index)
        records.append(record)
        print(
            f"full {index + 1}/{rounds} "
            f"ttft={record['pipeline_ttft_ms']:.3f}ms "
            f"e2e={record['pipeline_e2e_ms']:.3f}ms",
            flush=True,
        )
    return records, answer


def run_split_vision(
    vision_tower,
    embed_vision,
    processor,
    image: Image.Image,
    config,
    question: str,
    rounds: int,
) -> tuple[list[dict], dict]:
    prompt = make_prompt(processor, config, question)

    def encode(index: int) -> tuple[dict, object]:
        started = time.perf_counter()
        inputs = prepare_image(processor, image, prompt)
        processed = time.perf_counter()
        features = embed_vision(vision_tower(inputs["pixel_values"], None))
        synchronize(features)
        encoded = time.perf_counter()
        record = {
            "rep": index,
            "preprocess_ms": (processed - started) * 1000,
            "vision_encode_ms": (encoded - processed) * 1000,
            "vision_pipeline_ms": (encoded - started) * 1000,
        }
        return record, features

    encode(-1)
    records = []
    features = None
    mx.reset_peak_memory()
    for index in range(rounds):
        record, features = encode(index)
        records.append(record)
        print(
            f"vision {index + 1}/{rounds} "
            f"preprocess={record['preprocess_ms']:.3f}ms "
            f"encode={record['vision_encode_ms']:.3f}ms",
            flush=True,
        )
    if features is None:
        raise RuntimeError("Vision benchmark produced no features")
    dtype_name = str(features.dtype).rsplit(".", 1)[-1]
    itemsize = 2 if dtype_name in {"bfloat16", "float16"} else 4
    details = {
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "tensor_bytes": features.size * itemsize,
        "active_memory_gb": mx.get_active_memory() / 1e9,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
    }
    return records, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-4-e4b-it-4bit",
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--question", default="Describe this image.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = Image.open(args.image).convert("RGB")
    image = ImageOps.fit(source, (854, 480), method=Image.Resampling.LANCZOS)

    mx.reset_peak_memory()
    load_started = time.perf_counter()
    model, processor = load(args.model)
    synchronize()
    load_finished = time.perf_counter()
    load_metrics = {
        "wall_ms": (load_finished - load_started) * 1000,
        "active_memory_gb": mx.get_active_memory() / 1e9,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
    }
    print(json.dumps({"load": load_metrics}, indent=2), flush=True)

    mx.reset_peak_memory()
    full_records, answer = run_full(
        model,
        processor,
        image,
        args.question,
        args.rounds,
        args.max_tokens,
    )
    full_summary = {
        "pipeline_ttft_ms": summarize(
            [record["pipeline_ttft_ms"] for record in full_records]
        ),
        "pipeline_e2e_ms": summarize(
            [record["pipeline_e2e_ms"] for record in full_records]
        ),
        "prompt_tps": summarize([record["prompt_tps"] for record in full_records]),
        "generation_tps": summarize(
            [record["generation_tps"] for record in full_records]
        ),
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "last_answer": answer,
    }

    config = model.config
    vision_tower = model.vision_tower
    embed_vision = model.embed_vision
    del model
    gc.collect()
    mx.clear_cache()
    synchronize()
    split_active_memory_gb = mx.get_active_memory() / 1e9

    vision_records, vision_details = run_split_vision(
        vision_tower,
        embed_vision,
        processor,
        image,
        config,
        args.question,
        args.rounds,
    )
    vision_details["active_memory_before_benchmark_gb"] = split_active_memory_gb
    vision_summary = {
        "preprocess_ms": summarize(
            [record["preprocess_ms"] for record in vision_records]
        ),
        "vision_encode_ms": summarize(
            [record["vision_encode_ms"] for record in vision_records]
        ),
        "vision_pipeline_ms": summarize(
            [record["vision_pipeline_ms"] for record in vision_records]
        ),
        **vision_details,
    }

    result = {
        "model": args.model,
        "image_resolution": "854x480",
        "rounds": args.rounds,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "load": load_metrics,
        "full_local": full_summary,
        "split_vision": vision_summary,
        "raw": {
            "full_local": full_records,
            "split_vision": vision_records,
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
