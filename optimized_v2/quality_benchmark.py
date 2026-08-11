import argparse
import base64
import http.client
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

import torch
from PIL import Image

from optimized.client import StreamingImageClient

from .protocol import CONTENT_TYPE, encode_request


def make_chat_payload(
    features: torch.Tensor,
    question: str,
    model: str,
    max_tokens: int,
    stream: bool,
) -> bytes:
    buffer = io.BytesIO()
    torch.save(features.detach().cpu().contiguous(), buffer)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_embeds",
                        "image_embeds": base64.b64encode(buffer.getvalue()).decode(),
                    },
                    {"type": "text", "text": question},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if stream:
        payload.update(stream=True, stream_options={"include_usage": True})
    return json.dumps(payload).encode()


def ask_minimal(
    server: str,
    features: torch.Tensor,
    question: str,
    model: str,
    max_tokens: int,
) -> str:
    request = Request(
        f"{server.rstrip('/')}/v1/chat/completions",
        data=make_chat_payload(features, question, model, max_tokens, False),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        result = json.loads(response.read())
    return result["choices"][0]["message"]["content"]


def ask_vllm_base64(
    server: str,
    features: torch.Tensor,
    question: str,
    model: str,
    max_tokens: int,
) -> str:
    request = Request(
        f"{server.rstrip('/')}/v1/chat/completions",
        data=make_chat_payload(features, question, model, max_tokens, True),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    text = []
    with urlopen(request, timeout=300) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            choices = event.get("choices", [])
            fragment = (
                choices[0].get("delta", {}).get("content") if choices else None
            )
            if fragment:
                text.append(fragment)
    return "".join(text)


def ask_vllm_binary(
    server: str,
    features: torch.Tensor,
    question: str,
    model: str,
    max_tokens: int,
) -> str:
    from urllib.parse import urlsplit

    parsed = urlsplit(server)
    connection = http.client.HTTPConnection(
        parsed.hostname,
        parsed.port or 80,
        timeout=300,
    )
    connection.request(
        "POST",
        parsed.path.rstrip("/") + "/v1/chat/completions",
        body=encode_request(features, question, model, max_tokens),
        headers={"Content-Type": CONTENT_TYPE, "Accept": "text/event-stream"},
    )
    response = connection.getresponse()
    if response.status != 200:
        error = response.read().decode(errors="replace")
        connection.close()
        raise RuntimeError(f"Gateway returned HTTP {response.status}: {error}")
    text = []
    try:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            choices = event.get("choices", [])
            fragment = (
                choices[0].get("delta", {}).get("content") if choices else None
            )
            if fragment:
                text.append(fragment)
    finally:
        response.read()
        connection.close()
    return "".join(text)


def score(answer: str, case: dict) -> dict:
    normalized = answer.casefold()
    required = case["required"]
    group_hits = [
        any(term.casefold() in normalized for term in group) for group in required
    ]
    forbidden_hits = [
        term for term in case.get("forbidden", []) if term.casefold() in normalized
    ]
    return {
        "required_groups_hit": sum(group_hits),
        "required_groups_total": len(group_hits),
        "required_score": sum(group_hits) / len(group_hits),
        "forbidden_hits": forbidden_hits,
        "strict_pass": all(group_hits) and not forbidden_hits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--minimal-server", default="http://127.0.0.1:8000")
    parser.add_argument("--base64-server", default="http://127.0.0.1:8001")
    parser.add_argument("--binary-server", default="http://127.0.0.1:8002")
    parser.add_argument("--model", default="gemma-4-12b-optimized")
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text())
    encoder = StreamingImageClient(args.artifact, args.base64_server, args.model)
    backends = {
        "transformers": lambda features, question: ask_minimal(
            args.minimal_server,
            features,
            question,
            args.model,
            args.max_tokens,
        ),
        "vllm_base64": lambda features, question: ask_vllm_base64(
            args.base64_server,
            features,
            question,
            args.model,
            args.max_tokens,
        ),
        "vllm_binary": lambda features, question: ask_vllm_binary(
            args.binary_server,
            features,
            question,
            args.model,
            args.max_tokens,
        ),
    }
    results = []
    for case in cases:
        image_path = Path(case["image"]).expanduser()
        if not image_path.is_absolute():
            image_path = args.cases.parent / image_path
        image = Image.open(image_path).convert("RGB")
        features, _, _ = encoder.encode_image(image)
        backend_results = {}
        for name, ask in backends.items():
            answer = ask(features, case["question"])
            backend_results[name] = {
                "answer": answer,
                **score(answer, case),
            }
        results.append(
            {
                "name": case["name"],
                "image": str(image_path),
                "question": case["question"],
                "visual_tokens": features.shape[0],
                "transport_outputs_equal": (
                    backend_results["vllm_base64"]["answer"]
                    == backend_results["vllm_binary"]["answer"]
                ),
                "backends": backend_results,
            }
        )

    summary = {}
    for name in backends:
        backend_results = [result["backends"][name] for result in results]
        summary[name] = {
            "mean_required_score": sum(
                result["required_score"] for result in backend_results
            )
            / len(backend_results),
            "strict_passes": sum(result["strict_pass"] for result in backend_results),
            "cases": len(backend_results),
            "forbidden_hallucinations": sum(
                bool(result["forbidden_hits"]) for result in backend_results
            ),
        }
    report = {
        "summary": summary,
        "all_transport_outputs_equal": all(
            result["transport_outputs_equal"] for result in results
        ),
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
