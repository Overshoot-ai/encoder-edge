import base64
import http.client
import io
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from .gateway import create_handler
from .protocol import CONTENT_TYPE, encode_request

SSE_BODY = (
    b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
    b"data: [DONE]\n\n"
)


class UpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    requests: list[bytes] = []

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.requests.append(body)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(SSE_BODY)))
        self.end_headers()
        self.wfile.write(SSE_BODY)

    def log_message(self, format: str, *args: object) -> None:
        pass


class BinaryGatewayTest(unittest.TestCase):
    def test_preserves_bits_and_reuses_client_connection(self) -> None:
        original = torch.randn(8, 768, dtype=torch.bfloat16)
        payload = encode_request(original, "question", "model")
        UpstreamHandler.requests = []
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
        gateway = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            create_handler(f"http://127.0.0.1:{upstream.server_port}"),
        )
        threads = [
            threading.Thread(target=upstream.serve_forever, daemon=True),
            threading.Thread(target=gateway.serve_forever, daemon=True),
        ]
        for thread in threads:
            thread.start()

        try:
            connection = http.client.HTTPConnection(
                "127.0.0.1", gateway.server_port, timeout=10
            )
            for _ in range(2):
                connection.request(
                    "POST",
                    "/v1/chat/completions",
                    body=payload,
                    headers={"Content-Type": CONTENT_TYPE},
                )
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(response.read(), SSE_BODY)
            connection.close()

            unified = encode_request(
                torch.randn(8, 3840, dtype=torch.bfloat16),
                "question",
                "model",
            )
            connection = http.client.HTTPConnection(
                "127.0.0.1", gateway.server_port, timeout=10
            )
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=unified,
                headers={"Content-Type": CONTENT_TYPE},
            )
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), SSE_BODY)
            connection.close()

            self.assertEqual(len(UpstreamHandler.requests), 3)
            request = json.loads(UpstreamHandler.requests[0])
            encoded = request["messages"][0]["content"][0]["image_embeds"]
            reconstructed = torch.load(
                io.BytesIO(base64.b64decode(encoded)),
                map_location="cpu",
                weights_only=True,
            )
            self.assertTrue(
                torch.equal(
                    original.view(torch.int16),
                    reconstructed.view(torch.int16),
                )
            )

            invalid = encode_request(
                torch.randn(8, 999, dtype=torch.bfloat16),
                "question",
                "model",
            )
            connection = http.client.HTTPConnection(
                "127.0.0.1", gateway.server_port, timeout=10
            )
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=invalid,
                headers={"Content-Type": CONTENT_TYPE},
            )
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            self.assertIn(b"width must be 768, 2560, or 3840", response.read())
            connection.close()
            self.assertEqual(len(UpstreamHandler.requests), 3)
        finally:
            gateway.shutdown()
            upstream.shutdown()
            gateway.server_close()
            upstream.server_close()


if __name__ == "__main__":
    unittest.main()
