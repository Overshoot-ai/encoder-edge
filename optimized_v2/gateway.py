import argparse
import base64
import http.client
import io
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import torch

from .protocol import CONTENT_TYPE, HEADER, MAX_METADATA_BYTES, MAX_TENSOR_BYTES
from .protocol import BinaryRequest, decode_request

MAX_REQUEST_BYTES = HEADER.size + MAX_METADATA_BYTES + MAX_TENSOR_BYTES
IMAGE_EMBED_WIDTHS = frozenset((768, 2560, 3840))


def build_vllm_payload(request: BinaryRequest) -> bytes:
    buffer = io.BytesIO()
    torch.save(request.tensor.contiguous(), buffer)
    return json.dumps(
        {
            "model": request.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_embeds",
                            "image_embeds": base64.b64encode(
                                buffer.getvalue()
                            ).decode(),
                        },
                        {"type": "text", "text": request.question},
                    ],
                }
            ],
            "max_tokens": request.max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        separators=(",", ":"),
    ).encode()


def create_handler(upstream_url: str):
    parsed = urlsplit(upstream_url)
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError("Upstream must be an HTTP URL")
    upstream_path = parsed.path.rstrip(":/") + "/v1/chat/completions"

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self) -> None:
            super().setup()
            self.upstream = http.client.HTTPConnection(
                parsed.hostname,
                parsed.port or 80,
                timeout=300,
            )

        def finish(self) -> None:
            self.upstream.close()
            super().finish()

        def send_bytes(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self.send_bytes(404, "application/json", b'{"error":"Not found"}')
                return
            try:
                content_type = self.headers.get("Content-Type", "").split(";", 1)[0]
                if content_type != CONTENT_TYPE:
                    raise ValueError(f"Expected Content-Type {CONTENT_TYPE}")
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError("Invalid request size")
                payload = self.rfile.read(length)
                gateway_started = time.perf_counter()
                request = decode_request(payload)
                if request.tensor.shape[1] not in IMAGE_EMBED_WIDTHS:
                    raise ValueError(
                        "Visual tensor width must be 768, 2560, or 3840, got "
                        f"{request.tensor.shape[1]}"
                    )
                body = build_vllm_payload(request)
                gateway_prepare_ms = (time.perf_counter() - gateway_started) * 1000
                vllm_started = time.perf_counter()
                self.upstream.request(
                    "POST",
                    upstream_path,
                    body=body,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream",
                    },
                )
                response = self.upstream.getresponse()
                try:
                    if response.status != 200:
                        error = response.read()
                        self.send_bytes(
                            response.status,
                            response.getheader("Content-Type", "application/json"),
                            error,
                        )
                        return

                    buffered_lines = []
                    for line in response:
                        buffered_lines.append(line)
                        if not line.startswith(b"data: "):
                            continue
                        data = line[6:].strip()
                        if data == b"[DONE]":
                            break
                        event = json.loads(data)
                        choices = event.get("choices", [])
                        text = (
                            choices[0].get("delta", {}).get("content")
                            if choices
                            else None
                        )
                        if text:
                            break
                    gateway_ttft_ms = (time.perf_counter() - gateway_started) * 1000
                    vllm_ttft_ms = (time.perf_counter() - vllm_started) * 1000

                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header(
                        "X-Gateway-TTFT-Ms",
                        f"{gateway_ttft_ms:.6f}",
                    )
                    self.send_header(
                        "X-Gateway-Prepare-Ms",
                        f"{gateway_prepare_ms:.6f}",
                    )
                    self.send_header("X-vLLM-TTFT-Ms", f"{vllm_ttft_ms:.6f}")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    for line in buffered_lines:
                        self.wfile.write(f"{len(line):x}\r\n".encode())
                        self.wfile.write(line + b"\r\n")
                    for line in response:
                        self.wfile.write(f"{len(line):x}\r\n".encode())
                        self.wfile.write(line + b"\r\n")
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                finally:
                    response.close()
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                body = json.dumps({"error": str(error)}).encode()
                self.send_bytes(400, "application/json", body)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate bit-identical binary visual tensors to vLLM"
    )
    parser.add_argument("--upstream", default="http://127.0.0.1:8001")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    server = ThreadingHTTPServer(
        (args.host, args.port),
        create_handler(args.upstream),
    )
    print(f"Binary gateway listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
