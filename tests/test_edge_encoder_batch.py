from __future__ import annotations

import json
import tempfile
import time
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from cross_device_gemma.chat_protocol import decode_chat_request
from cross_device_gemma.edge_encoder_batch import load_requests, run_batch


class FakeEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, image):
        return torch.zeros(2, 768, dtype=torch.bfloat16), {
            "preprocess_ms": 1.0,
            "encode_ms": 2.0,
        }


class FakeClient:
    def __init__(self, *args, **kwargs):
        pass

    def stream(self, payload):
        request = decode_chat_request(payload)
        if request.question == "first":
            time.sleep(0.03)
        yield {"type": "token", "text": request.question.upper()}
        yield {"type": "done", "remote_ttft_ms": 1.0, "remote_e2e_ms": 2.0}

    def close(self):
        pass


class EdgeEncoderBatchTests(unittest.TestCase):
    def test_loads_relative_images_and_rejects_empty_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "requests.jsonl"
            manifest.write_text('{"image":"image.jpg","prompt":"question"}\n')
            requests = load_requests(manifest, 128)
            self.assertEqual(requests[0]["image"], (root / "image.jpg").resolve())

            manifest.write_text("\n")
            with self.assertRaisesRegex(ValueError, "no requests"):
                load_requests(manifest, 128)

    @patch("cross_device_gemma.edge_encoder_batch.ChatClient", FakeClient)
    @patch("cross_device_gemma.edge_encoder_batch.EdgeEncoder", FakeEncoder)
    def test_writes_results_in_input_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("one.png", "two.png"):
                Image.new("RGB", (2, 2)).save(root / name)
            manifest = root / "requests.jsonl"
            manifest.write_text(
                '{"id":"a","image":"one.png","prompt":"first"}\n'
                '{"id":"b","image":"two.png","prompt":"second"}\n'
            )
            output = root / "responses.jsonl"
            args = Namespace(
                input=manifest,
                output=output,
                server="http://gateway",
                max_tokens=128,
                max_in_flight=2,
                cache_dir=None,
                zstd=False,
            )

            summary = run_batch(args, token=None, api_key=None)
            results = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual([result["id"] for result in results], ["a", "b"])
        self.assertEqual([result["response"] for result in results], ["FIRST", "SECOND"])
        self.assertEqual(summary["requests"], 2)
        self.assertIn("pipeline_ttft_ms", results[0]["metrics"])


if __name__ == "__main__":
    unittest.main()
