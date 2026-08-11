import argparse
import base64
import json
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image

from .client import BinaryStreamingImageClient
from .quality_benchmark import score


def stream_full_h200(
    server: str,
    model: str,
    image_path: Path,
    question: str,
    max_tokens: int,
) -> tuple[str, dict]:
    total_started = time.perf_counter()
    image_bytes = image_path.read_bytes()
    suffix = image_path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": question},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
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
        "pipeline_ttft_ms": (first_token_at - total_started) * 1000,
        "remote_ttft_ms": (first_token_at - remote_started) * 1000,
        "pipeline_e2e_ms": (finished - total_started) * 1000,
        "remote_e2e_ms": (finished - remote_started) * 1000,
        "usage": usage,
    }


def stream_split(
    client: BinaryStreamingImageClient,
    image_path: Path,
    question: str,
    max_tokens: int,
) -> tuple[str, dict]:
    text = []
    done = None
    for event in client.stream(Image.open(image_path), question, max_tokens):
        if event["type"] == "token":
            text.append(event["text"])
        else:
            done = event
    if done is None:
        raise RuntimeError("Split client did not produce completion metrics")
    return "".join(text), done


def median(results: list[dict], field: str) -> float:
    return statistics.median(result[field] for result in results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["full", "split"])
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "split" and args.artifact is None:
        parser.error("--artifact is required in split mode")

    cases = json.loads(args.cases.read_text())
    split_client = (
        BinaryStreamingImageClient(args.artifact, args.server, args.model)
        if args.mode == "split"
        else None
    )
    case_results = []
    all_metrics = []
    for case in cases:
        image_path = Path(case["image"]).expanduser()
        if not image_path.is_absolute():
            image_path = args.cases.parent / image_path
        runs = []
        for _ in range(args.rounds):
            if args.mode == "full":
                answer, metrics = stream_full_h200(
                    args.server,
                    args.model,
                    image_path,
                    case["question"],
                    args.max_tokens,
                )
            else:
                answer, metrics = stream_split(
                    split_client,
                    image_path,
                    case["question"],
                    args.max_tokens,
                )
            scored = score(answer, case)
            runs.append({"answer": answer, **scored, "metrics": metrics})
            all_metrics.append(metrics)
        case_results.append(
            {
                "name": case["name"],
                "image": str(image_path),
                "question": case["question"],
                "runs": runs,
            }
        )

    quality_runs = [run for case in case_results for run in case["runs"]]
    report = {
        "mode": args.mode,
        "rounds": args.rounds,
        "summary": {
            "mean_required_score": sum(
                run["required_score"] for run in quality_runs
            )
            / len(quality_runs),
            "strict_passes": sum(run["strict_pass"] for run in quality_runs),
            "quality_runs": len(quality_runs),
            "forbidden_hallucinations": sum(
                bool(run["forbidden_hits"]) for run in quality_runs
            ),
            "median_request_bytes": median(all_metrics, "request_bytes"),
            "median_client_preprocess_ms": median(
                all_metrics, "client_preprocess_ms"
            ),
            "median_client_encode_ms": median(all_metrics, "client_encode_ms"),
            "median_client_serialize_ms": median(
                all_metrics, "client_serialize_ms"
            ),
            "median_remote_ttft_ms": median(all_metrics, "remote_ttft_ms"),
            "median_pipeline_ttft_ms": median(all_metrics, "pipeline_ttft_ms"),
            "median_remote_e2e_ms": median(all_metrics, "remote_e2e_ms"),
            "median_pipeline_e2e_ms": median(all_metrics, "pipeline_e2e_ms"),
        },
        "cases": case_results,
    }
    rendered = json.dumps(report, indent=2)
    args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
