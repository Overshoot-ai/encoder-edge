from __future__ import annotations

import base64
import hmac
import http.client
import io
import json
import os
import ssl
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import torch

from .chat_protocol import (
    CONTENT_TYPE,
    HEADER,
    MAX_METADATA_BYTES,
    MAX_TENSOR_BYTES,
    ChatRequest,
    SERVER_REVISION,
    decode_chat_request,
)


MAX_REQUEST_BYTES = HEADER.size + MAX_METADATA_BYTES + MAX_TENSOR_BYTES


def build_vllm_payload(request: ChatRequest) -> bytes:
    buffer = io.BytesIO()
    torch.save(request.tensor.contiguous(), buffer)
    return json.dumps(
        {
            "model": request.served_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_embeds",
                            "image_embeds": base64.b64encode(buffer.getvalue()).decode(),
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


def create_handler(
    upstream_url: str,
    api_key: str | None = None,
    upstream_api_key: str | None = None,
    server_revision: str = SERVER_REVISION,
):
    parsed = urlsplit(upstream_url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Upstream must be an HTTP or HTTPS URL")
    upstream_path = parsed.path.rstrip("/") + "/v1/chat/completions"

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _new_upstream(self):
            connection_type = (
                http.client.HTTPSConnection
                if parsed.scheme == "https"
                else http.client.HTTPConnection
            )
            options = {}
            if parsed.scheme == "https":
                options["context"] = ssl.create_default_context()
            return connection_type(
                parsed.hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                timeout=300,
                **options,
            )

        def setup(self) -> None:
            super().setup()
            self.upstream = self._new_upstream()

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
            response_started = False
            if self.path != "/v1/chat/completions":
                self.send_bytes(404, "application/json", b'{"error":"Not found"}')
                return
            if api_key:
                expected = f"Bearer {api_key}"
                supplied = self.headers.get("Authorization", "")
                if not hmac.compare_digest(supplied, expected):
                    self.send_bytes(401, "application/json", b'{"error":"Unauthorized"}')
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
                request = decode_chat_request(
                    payload, expected_server_revision=server_revision
                )
                body = build_vllm_payload(request)
                gateway_prepare_ms = (time.perf_counter() - gateway_started) * 1000

                upstream_headers = {
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                }
                if upstream_api_key:
                    upstream_headers["Authorization"] = f"Bearer {upstream_api_key}"
                vllm_started = time.perf_counter()
                self.upstream.request(
                    "POST", upstream_path, body=body, headers=upstream_headers
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

                    buffered = bytearray()
                    while b"\n\n" not in buffered and b"\r\n\r\n" not in buffered:
                        chunk = response.read(1)
                        if not chunk:
                            break
                        buffered.extend(chunk)
                    gateway_ttft_ms = (time.perf_counter() - gateway_started) * 1000
                    vllm_ttft_ms = (time.perf_counter() - vllm_started) * 1000

                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("X-Gateway-TTFT-Ms", f"{gateway_ttft_ms:.6f}")
                    self.send_header("X-Gateway-Prepare-Ms", f"{gateway_prepare_ms:.6f}")
                    self.send_header("X-vLLM-TTFT-Ms", f"{vllm_ttft_ms:.6f}")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    response_started = True

                    def write_chunk(chunk: bytes) -> None:
                        if not chunk:
                            return
                        self.wfile.write(f"{len(chunk):x}\r\n".encode())
                        self.wfile.write(chunk + b"\r\n")
                        self.wfile.flush()

                    write_chunk(bytes(buffered))
                    for line in response:
                        write_chunk(line)
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                finally:
                    response.close()
            except (ValueError, TypeError) as error:
                self.send_bytes(400, "application/json", json.dumps({"error": str(error)}).encode())
            except (OSError, http.client.HTTPException) as error:
                if not response_started:
                    self.send_bytes(
                        502,
                        "application/json",
                        json.dumps(
                            {"error": f"Upstream request failed: {error}"}
                        ).encode(),
                    )
                self.upstream.close()
                self.upstream = self._new_upstream()

        def log_message(self, format: str, *args: object) -> None:
            pass

    return Handler


def serve_gateway(upstream: str, host: str, port: int) -> None:
    server = ThreadingHTTPServer(
        (host, port),
        create_handler(
            upstream,
            api_key=os.environ.get("EDGE_ENCODER_GATEWAY_API_KEY"),
            upstream_api_key=os.environ.get("EDGE_ENCODER_UPSTREAM_API_KEY"),
            server_revision=os.environ.get(
                "EDGE_ENCODER_SERVER_REVISION", SERVER_REVISION
            ),
        ),
    )
    print(f"Edge Encoder gateway listening on http://{host}:{port}")
    server.serve_forever()
