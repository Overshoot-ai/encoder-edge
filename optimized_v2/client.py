import argparse
import http.client
import json
import time
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image

from optimized.client import StreamingImageClient

from .protocol import CONTENT_TYPE, encode_request


class BinaryStreamingImageClient(StreamingImageClient):
    def __init__(self, artifact: Path, server: str, model: str):
        super().__init__(artifact, server, model)
        parsed = urlsplit(server)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("Server must be an HTTP URL")
        self.path = parsed.path.rstrip("/") + "/v1/chat/completions"
        self.host = parsed.hostname
        self.port = parsed.port or 80

    def stream(self, image: Image.Image, question: str, max_tokens: int = 128):
        total_started = time.perf_counter()
        features, preprocess_ms, encode_ms = self.encode_image(image)

        started = time.perf_counter()
        payload = encode_request(features, question, self.model, max_tokens)
        pack_ms = (time.perf_counter() - started) * 1000
        remote_started = time.perf_counter()
        connection = http.client.HTTPConnection(self.host, self.port, timeout=300)
        connection.request(
            "POST",
            self.path,
            body=payload,
            headers={
                "Content-Type": CONTENT_TYPE,
                "Accept": "text/event-stream",
            },
        )
        response = connection.getresponse()
        if response.status != 200:
            error = response.read().decode(errors="replace")
            connection.close()
            raise RuntimeError(f"Gateway returned HTTP {response.status}: {error}")
        gateway_ttft_header = response.getheader("X-Gateway-TTFT-Ms")
        gateway_ttft_ms = (
            float(gateway_ttft_header) if gateway_ttft_header is not None else None
        )
        gateway_prepare_header = response.getheader("X-Gateway-Prepare-Ms")
        gateway_prepare_ms = (
            float(gateway_prepare_header)
            if gateway_prepare_header is not None
            else None
        )
        vllm_ttft_header = response.getheader("X-vLLM-TTFT-Ms")
        vllm_ttft_ms = (
            float(vllm_ttft_header) if vllm_ttft_header is not None else None
        )

        first_token_at = None
        usage = None
        try:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    break
                event = json.loads(data)
                usage = event.get("usage") or usage
                choices = event.get("choices", [])
                text = (
                    choices[0].get("delta", {}).get("content") if choices else None
                )
                if text:
                    first_token_at = first_token_at or time.perf_counter()
                    yield {"type": "token", "text": text}
        finally:
            response.read()
            connection.close()

        finished = time.perf_counter()
        if first_token_at is None:
            first_token_at = finished
        tensor_bytes = features.numel() * features.element_size()
        yield {
            "type": "done",
            "client_preprocess_ms": preprocess_ms,
            "client_encode_ms": encode_ms,
            "client_serialize_ms": pack_ms,
            "request_bytes": len(payload),
            "tensor_bytes": tensor_bytes,
            "base64_tensor_bytes": 4 * ((tensor_bytes + 2) // 3),
            "visual_tokens": features.shape[0],
            "pipeline_ttft_ms": (first_token_at - total_started) * 1000,
            "remote_ttft_ms": (first_token_at - remote_started) * 1000,
            "gateway_ttft_ms": gateway_ttft_ms,
            "gateway_prepare_ms": gateway_prepare_ms,
            "vllm_ttft_ms": vllm_ttft_ms,
            "transport_ttft_ms": (
                (first_token_at - remote_started) * 1000 - gateway_ttft_ms
                if gateway_ttft_ms is not None
                else None
            ),
            "pipeline_e2e_ms": (finished - total_started) * 1000,
            "remote_e2e_ms": (finished - remote_started) * 1000,
            "usage": usage,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", default="gemma-4-12b-optimized")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", required=True)
    args = parser.parse_args()
    client = BinaryStreamingImageClient(args.artifact, args.server, args.model)
    for event in client.stream(Image.open(args.image), args.question):
        print(event.get("text", ""), end="", flush=True)
        if event["type"] == "done":
            print("\n" + json.dumps(event, indent=2))


if __name__ == "__main__":
    main()
