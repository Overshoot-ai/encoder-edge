from __future__ import annotations

import http.client
import json
import queue
import ssl
import threading
import time
from urllib.parse import urlsplit

from .chat_protocol import CONTENT_TYPE


def _parse_server(server: str):
    parsed = urlsplit(server)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Server must be an HTTP or HTTPS URL without credentials or query parameters")
    path = parsed.path.rstrip("/") + "/v1/chat/completions"
    return parsed, path


def iter_sse(response):
    data_lines = []
    saw_done = False
    while True:
        line = response.readline()
        if not line:
            break
        line = line.rstrip(b"\r\n")
        if not line:
            if not data_lines:
                continue
            data = b"\n".join(data_lines).decode(errors="replace")
            data_lines = []
            if data.strip() == "[DONE]":
                saw_done = True
                break
            try:
                yield json.loads(data)
            except json.JSONDecodeError as error:
                raise RuntimeError("Gateway returned invalid SSE JSON") from error
            continue
        if line.startswith(b"data:"):
            data_lines.append(line[5:].lstrip(b" "))
    if not saw_done:
        raise RuntimeError("Gateway stream ended before [DONE]")


class ChatClient:
    def __init__(
        self,
        server: str,
        api_key: str | None = None,
        timeout: float = 300,
        max_connections: int = 1,
    ):
        if max_connections < 1:
            raise ValueError("max_connections must be positive")
        self.parsed, self.path = _parse_server(server)
        self.api_key = api_key
        self.timeout = timeout
        self.max_connections = max_connections
        self.pool = queue.LifoQueue()
        self.condition = threading.Condition()
        self.connection_count = 0

    def _new_connection(self):
        connection_type = (
            http.client.HTTPSConnection
            if self.parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        options = {}
        if self.parsed.scheme == "https":
            options["context"] = ssl.create_default_context()
        return connection_type(
            self.parsed.hostname,
            self.parsed.port or (443 if self.parsed.scheme == "https" else 80),
            timeout=self.timeout,
            **options,
        )

    def _acquire(self):
        with self.condition:
            while True:
                try:
                    return self.pool.get_nowait()
                except queue.Empty:
                    if self.connection_count < self.max_connections:
                        self.connection_count += 1
                        return self._new_connection()
                    self.condition.wait()

    def _release(self, connection, reusable: bool) -> None:
        with self.condition:
            if reusable:
                self.pool.put(connection)
            else:
                connection.close()
                self.connection_count -= 1
            self.condition.notify()

    def stream(self, payload: bytes):
        connection = self._acquire()
        headers = {"Content-Type": CONTENT_TYPE, "Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        remote_started = time.perf_counter()
        response = None
        reusable = False
        try:
            connection.request("POST", self.path, body=payload, headers=headers)
            response = connection.getresponse()
            if response.status != 200:
                detail = response.read().decode(errors="replace")
                raise RuntimeError(f"Gateway returned HTTP {response.status}: {detail}")
            first_token_at = None
            usage = None
            for event in iter_sse(response):
                if event.get("error"):
                    raise RuntimeError(f"Gateway stream error: {event['error']}")
                usage = event.get("usage") or usage
                choices = event.get("choices") or []
                text = choices[0].get("delta", {}).get("content") if choices else None
                if text:
                    first_token_at = first_token_at or time.perf_counter()
                    yield {"type": "token", "text": text}
            response.read()
            reusable = True
            finished = time.perf_counter()
            first_token_at = first_token_at or finished
            yield {
                "type": "done",
                "remote_ttft_ms": (first_token_at - remote_started) * 1000,
                "remote_e2e_ms": (finished - remote_started) * 1000,
                "gateway_ttft_ms": _float_header(response, "X-Gateway-TTFT-Ms"),
                "gateway_prepare_ms": _float_header(response, "X-Gateway-Prepare-Ms"),
                "vllm_ttft_ms": _float_header(response, "X-vLLM-TTFT-Ms"),
                "usage": usage,
            }
        finally:
            if response is not None and not reusable:
                response.close()
            self._release(connection, reusable)

    def close(self) -> None:
        with self.condition:
            while True:
                try:
                    self.pool.get_nowait().close()
                    self.connection_count -= 1
                except queue.Empty:
                    break


def stream_chat(
    server: str,
    payload: bytes,
    api_key: str | None = None,
    timeout: float = 300,
):
    client = ChatClient(server, api_key=api_key, timeout=timeout)
    try:
        yield from client.stream(payload)
    finally:
        client.close()


def _float_header(response, name: str) -> float | None:
    value = response.getheader(name)
    return float(value) if value is not None else None
