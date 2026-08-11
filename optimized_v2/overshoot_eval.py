import argparse
import base64
import io
import json
import math
import re
import statistics
import string
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageOps

CHARTQA_INSTRUCTIONS = """{question}
Analyze the image and question carefully, using step-by-step reasoning.
First, describe any image provided in detail. Then, present your reasoning. And finally your final answer in this format:
Final Answer: <answer>
where <answer> follows the following instructions:
- <answer> should should be a single phrase or number.
- <answer> should not paraphrase or reformat the text in the image.
- If <answer> is a ratio, it should be a decimal value like 0.25 instead of 1:4.
- If the question is a Yes/No question, <answer> should be Yes/No.
- If <answer> is a number, it should not contain any units.
- If <answer> is a percentage, it should include a % sign.
- If <answer> is an entity, it should include the full label from the graph.
IMPORTANT: Remember, to end your answer with Final Answer: <answer>."""


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def image_bytes(image: Image.Image, format: str, **kwargs: object) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=format, **kwargs)
    return buffer.getvalue()


def stream_full(
    server: str,
    model: str,
    encoded_image: bytes,
    mime: str,
    prompt: str,
    max_tokens: int,
) -> tuple[str, dict]:
    total_started = time.perf_counter()
    data_url = f"data:{mime};base64,{base64.b64encode(encoded_image).decode()}"
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        separators=(",", ":"),
    ).encode()
    pack_ms = (time.perf_counter() - total_started) * 1000
    remote_started = time.perf_counter()
    request = Request(
        f"{server.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    text = []
    first_token_at = None
    usage = None
    with urlopen(request, timeout=300) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            usage = event.get("usage") or usage
            choices = event.get("choices", [])
            fragment = (
                choices[0].get("delta", {}).get("content") if choices else None
            )
            if fragment:
                first_token_at = first_token_at or time.perf_counter()
                text.append(fragment)
    finished = time.perf_counter()
    first_token_at = first_token_at or finished
    return "".join(text), {
        "client_preprocess_ms": 0.0,
        "client_encode_ms": 0.0,
        "client_serialize_ms": pack_ms,
        "request_bytes": len(payload),
        "remote_ttft_ms": (first_token_at - remote_started) * 1000,
        "pipeline_ttft_ms": (first_token_at - total_started) * 1000,
        "remote_e2e_ms": (finished - remote_started) * 1000,
        "pipeline_e2e_ms": (finished - total_started) * 1000,
        "usage": usage,
    }


def stream_split(client, image: Image.Image, prompt: str, max_tokens: int) -> tuple[str, dict]:
    text = []
    done = None
    for event in client.stream(image, prompt, max_tokens):
        if event["type"] == "token":
            text.append(event["text"])
        else:
            done = event
    if done is None:
        raise RuntimeError("Split client did not return completion metrics")
    return "".join(text), done


def extract_final_answer(generation: str) -> str:
    generation = re.sub(r"([aA]nswer)\**:\**", r"\1:", generation)
    index = generation.lower().rfind("answer:")
    if index == -1:
        return ""
    lines = generation[index + len("answer:") :].splitlines()
    answer = next((line.strip() for line in lines if line.strip()), "")
    return re.sub(r"[*_\[\]\(\)]", "", answer)


def preprocess_answer(text: str) -> str:
    if not any(char.isdigit() for char in text):
        if (text.startswith('"') and text.endswith('"')) or (
            text.startswith("'") and text.endswith("'")
        ):
            return text[1:-1]
        return text
    while text and (text[-1] in string.punctuation or text[-1].isspace()) and text[-1] != "%":
        text = text[:-1]
    return text.replace(",", "").replace("$", "")


def relaxed_correct(prediction: str, targets: list[str]) -> bool:
    prediction = preprocess_answer(prediction)
    for target in targets:
        target = preprocess_answer(target)
        try:
            pred_percent = prediction.endswith("%")
            target_percent = target.endswith("%")
            pred_value = float(prediction.rstrip("%"))
            target_value = float(target.rstrip("%"))
            candidates = [(pred_value, target_value)]
            if pred_percent or target_percent:
                candidates.extend(
                    [(pred_value / 100, target_value), (pred_value, target_value / 100)]
                )
            if any(
                abs(pred - expected) / max(abs(expected), 1e-10) <= 0.05
                for pred, expected in candidates
            ):
                return True
        except ValueError:
            normalized_prediction = prediction.rstrip(string.punctuation).casefold()
            if normalized_prediction == target.casefold():
                return True
    return False


def chartqa_scores(generation: str, targets: list[str]) -> dict:
    parsed = extract_final_answer(generation)
    exact = bool(parsed) and any(
        parsed.casefold().removesuffix(".") == target.strip().casefold()
        for target in targets
    )
    relaxed = bool(parsed) and relaxed_correct(parsed, targets)
    anywhere = relaxed or any(target.casefold() in generation.casefold() for target in targets)
    return {
        "parsed_answer": parsed,
        "exact_match": float(exact),
        "relaxed_accuracy": float(relaxed),
        "anywhere_accuracy": float(anywhere),
    }


def write_outputs(output: Path, metadata: dict, records: list[dict], summary: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "raw_results.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def make_client(args: argparse.Namespace):
    if args.mode == "split":
        if args.artifact is None:
            raise ValueError("--artifact is required in split mode")
        from .client import BinaryStreamingImageClient

        return BinaryStreamingImageClient(args.artifact, args.server, args.model)
    return None


def run_performance(args: argparse.Namespace) -> None:
    source = Image.open(args.image).convert("RGB")
    image = ImageOps.fit(source, (854, 480), method=Image.Resampling.LANCZOS)
    jpeg = image_bytes(image, "JPEG", quality=90)
    client = make_client(args)
    prompts = [
        f"Describe this image. nonce={uuid.uuid4()} thread_id=0 rep={index}"
        for index in range(args.reps + 1)
    ]

    def run(prompt: str) -> tuple[str, dict]:
        if args.mode == "full":
            return stream_full(
                args.server, args.model, jpeg, "image/jpeg", prompt, 1
            )
        return stream_split(client, image, prompt, 1)

    run(prompts[0])
    records = []
    for index, prompt in enumerate(prompts[1:]):
        answer, metrics = run(prompt)
        records.append({"rep": index, "answer": answer, **metrics})

    summary = {}
    for field in (
        "client_preprocess_ms",
        "client_encode_ms",
        "client_serialize_ms",
        "request_bytes",
        "tensor_bytes",
        "remote_ttft_ms",
        "gateway_ttft_ms",
        "gateway_prepare_ms",
        "vllm_ttft_ms",
        "transport_ttft_ms",
        "pipeline_ttft_ms",
        "remote_e2e_ms",
        "pipeline_e2e_ms",
    ):
        values = [record.get(field) for record in records]
        if any(value is None for value in values):
            continue
        summary[field] = {
            "mean": statistics.mean(values),
            "p50": percentile(values, 0.5),
            "p90": percentile(values, 0.9),
            "min": min(values),
            "max": max(values),
        }
    metadata = {
        "benchmark": "overshoot_ttft_480p",
        "mode": args.mode,
        "model": args.model,
        "reps": args.reps,
        "requests_total": args.reps,
        "frame_resolution": "854x480 JPEG quality=90",
        "max_tokens": 1,
        "stream": True,
        "temperature": 0,
        "unique_nonce": True,
        "sequential": True,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    write_outputs(args.output, metadata, records, summary)
    print(json.dumps(summary, indent=2))


def run_chartqa(args: argparse.Namespace) -> None:
    from datasets import load_dataset

    dataset = load_dataset("HuggingFaceM4/ChartQA", split="test")
    client = make_client(args)
    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "raw_results.jsonl"
    records = (
        [json.loads(line) for line in raw_path.read_text().splitlines()]
        if raw_path.exists()
        else []
    )
    if records:
        print(f"Resuming from sample {len(records) + 1}", flush=True)
    for index in range(len(records), min(args.limit, len(dataset))):
        sample = dataset[index]
        image = sample["image"].convert("RGB")
        prompt = CHARTQA_INSTRUCTIONS.format(question=sample["query"])
        for attempt in range(3):
            try:
                if args.mode == "full":
                    png = image_bytes(image, "PNG")
                    answer, metrics = stream_full(
                        args.server,
                        args.model,
                        png,
                        "image/png",
                        prompt,
                        512,
                    )
                else:
                    answer, metrics = stream_split(client, image, prompt, 512)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
        scores = chartqa_scores(answer, sample["label"])
        record = {
            "index": index,
            "query": sample["query"],
            "targets": sample["label"],
            "generation": answer,
            **scores,
            **metrics,
        }
        records.append(record)
        with raw_path.open("a") as output:
            output.write(json.dumps(record) + "\n")
        print(
            f"{index + 1}/{args.limit} relaxed={scores['relaxed_accuracy']:.0f} "
            f"answer={scores['parsed_answer']!r} target={sample['label']!r}",
            flush=True,
        )

    summary = {
        "samples": len(records),
        "exact_match": statistics.mean(record["exact_match"] for record in records),
        "relaxed_accuracy": statistics.mean(
            record["relaxed_accuracy"] for record in records
        ),
        "anywhere_accuracy": statistics.mean(
            record["anywhere_accuracy"] for record in records
        ),
        "pipeline_ttft_ms_p50": percentile(
            [record["pipeline_ttft_ms"] for record in records], 0.5
        ),
        "pipeline_ttft_ms_p90": percentile(
            [record["pipeline_ttft_ms"] for record in records], 0.9
        ),
    }
    metadata = {
        "benchmark": "lm_eval_chartqa",
        "dataset": "HuggingFaceM4/ChartQA",
        "split": "test",
        "mode": args.mode,
        "model": args.model,
        "limit": args.limit,
        "max_gen_toks": 512,
        "temperature": 0,
        "apply_chat_template": True,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    write_outputs(args.output, metadata, records, summary)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="benchmark", required=True)
    for name in ("performance", "chartqa"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("mode", choices=["full", "split"])
        subparser.add_argument("--server", required=True)
        subparser.add_argument("--model", required=True)
        subparser.add_argument("--artifact", type=Path)
        subparser.add_argument("--output", type=Path, required=True)
    performance = subparsers.choices["performance"]
    performance.add_argument("--image", type=Path, required=True)
    performance.add_argument("--reps", type=int, default=20)
    chartqa = subparsers.choices["chartqa"]
    chartqa.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    if args.benchmark == "performance":
        run_performance(args)
    else:
        run_chartqa(args)


if __name__ == "__main__":
    main()
