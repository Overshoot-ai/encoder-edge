from __future__ import annotations

import base64
import http.client
import io
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from cross_device_gemma.chat_gateway import create_handler
from cross_device_gemma.chat_protocol import CONTENT_TYPE, encode_chat_request


SSE_BODY = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'


class UpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    requests = []
    connections = 0

    def setup(self):
        super().setup()
        self.__class__.connections += 1

    def do_POST(self):
        self.__class__.requests.append(
            (
                self.headers,
                self.rfile.read(int(self.headers["Content-Length"])),
            )
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(SSE_BODY)))
        self.end_headers()
        self.wfile.write(SSE_BODY)

    def log_message(self, format, *args):
        pass


class ChatGatewayTests(unittest.TestCase):
    def test_authenticates_preserves_bits_and_binds_served_model(self):
        original = torch.randn(8, 768, dtype=torch.bfloat16)
        payload = encode_chat_request(original, "question")
        UpstreamHandler.requests = []
        UpstreamHandler.connections = 0
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
        gateway = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            create_handler(
                f"http://127.0.0.1:{upstream.server_port}",
                api_key="client-secret",
                upstream_api_key="upstream-secret",
            ),
        )
        for server in (upstream, gateway):
            threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", gateway.server_port)
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=payload,
                headers={
                    "Content-Type": CONTENT_TYPE,
                    "Authorization": "Bearer client-secret",
                },
            )
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), SSE_BODY)
            connection.close()

            self.assertEqual(len(UpstreamHandler.requests), 1)
            headers, body = UpstreamHandler.requests[0]
            self.assertEqual(headers["Authorization"], "Bearer upstream-secret")
            request = json.loads(body)
            self.assertEqual(request["model"], "gemma-4-e4b-optimized")
            encoded = request["messages"][0]["content"][0]["image_embeds"]
            reconstructed = torch.load(
                io.BytesIO(base64.b64decode(encoded)),
                map_location="cpu",
                weights_only=True,
            )
            self.assertTrue(
                torch.equal(original.view(torch.int16), reconstructed.view(torch.int16))
            )

            connection = http.client.HTTPConnection("127.0.0.1", gateway.server_port)
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=payload,
                headers={"Content-Type": CONTENT_TYPE},
            )
            response = connection.getresponse()
            self.assertEqual(response.status, 401)
            response.read()
            connection.close()
            self.assertEqual(len(UpstreamHandler.requests), 1)
            self.assertEqual(UpstreamHandler.connections, 1)
        finally:
            gateway.shutdown()
            upstream.shutdown()
            gateway.server_close()
            upstream.server_close()

    def test_rejects_server_revision_mismatch_before_upstream(self):
        payload = encode_chat_request(
            torch.zeros(2, 768, dtype=torch.bfloat16), "question"
        )
        UpstreamHandler.requests = []
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
        gateway = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            create_handler(
                f"http://127.0.0.1:{upstream.server_port}",
                server_revision="different-release",
            ),
        )
        for server in (upstream, gateway):
            threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", gateway.server_port)
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=payload,
                headers={"Content-Type": CONTENT_TYPE},
            )
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            self.assertIn(b"server_revision", response.read())
            connection.close()
            self.assertEqual(UpstreamHandler.requests, [])
        finally:
            gateway.shutdown()
            upstream.shutdown()
            gateway.server_close()
            upstream.server_close()


if __name__ == "__main__":
    unittest.main()
