from __future__ import annotations

import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from cross_device_gemma.chat_client import ChatClient, stream_chat
from cross_device_gemma.chat_protocol import CONTENT_TYPE


SSE_BODY = (
    b'data: {"choices":[{"delta":{"role":"assistant"}}]}\r\n\r\n'
    b'data:{"choices":[{"delta":{"content":"hello "}}]}\n\n'
    b'data: {"choices":[{"delta":{"content":"world"}}]}\n\n'
    b'data: {"choices":[],"usage":{"completion_tokens":2}}\n\n'
    b"data: [DONE]\n\n"
)


class GatewayHandler(BaseHTTPRequestHandler):
    headers_seen = None
    payload_seen = None

    def do_POST(self):
        self.__class__.headers_seen = self.headers
        self.__class__.payload_seen = self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(SSE_BODY)))
        self.send_header("X-Gateway-TTFT-Ms", "12.5")
        self.end_headers()
        self.wfile.write(SSE_BODY)

    def log_message(self, format, *args):
        pass


class ChatClientTests(unittest.TestCase):
    def test_streams_content_and_sends_bearer_auth(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), GatewayHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            events = list(
                stream_chat(
                    f"http://127.0.0.1:{server.server_port}",
                    b"payload",
                    api_key="secret",
                )
            )
        finally:
            server.shutdown()
            server.server_close()

        self.assertEqual("".join(e["text"] for e in events[:-1]), "hello world")
        self.assertEqual(events[-1]["usage"]["completion_tokens"], 2)
        self.assertEqual(events[-1]["gateway_ttft_ms"], 12.5)
        self.assertEqual(GatewayHandler.headers_seen["Authorization"], "Bearer secret")
        self.assertEqual(GatewayHandler.headers_seen["Content-Type"], CONTENT_TYPE)
        self.assertEqual(GatewayHandler.payload_seen, b"payload")

    def test_rejects_early_end_of_stream(self):
        class EarlyHandler(GatewayHandler):
            def do_POST(self):
                body = b'data: {"choices":[]}\n\n'
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), EarlyHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with self.assertRaisesRegex(RuntimeError, "before \[DONE\]"):
                list(stream_chat(f"http://127.0.0.1:{server.server_port}", b"x"))
        finally:
            server.shutdown()
            server.server_close()

    def test_reuses_persistent_connection(self):
        class PersistentHandler(GatewayHandler):
            protocol_version = "HTTP/1.1"
            connections = 0

            def setup(self):
                super().setup()
                self.__class__.connections += 1

        PersistentHandler.connections = 0
        server = ThreadingHTTPServer(("127.0.0.1", 0), PersistentHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        client = ChatClient(f"http://127.0.0.1:{server.server_port}")
        try:
            list(client.stream(b"one"))
            list(client.stream(b"two"))
        finally:
            client.close()
            server.shutdown()
            server.server_close()

        self.assertEqual(PersistentHandler.connections, 1)


if __name__ == "__main__":
    unittest.main()
