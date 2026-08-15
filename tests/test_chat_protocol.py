from __future__ import annotations

import json
import unittest

import torch

from cross_device_gemma.chat_protocol import (
    HEADER,
    decode_chat_request,
    encode_chat_request,
)


class ChatProtocolTests(unittest.TestCase):
    def test_preserves_bfloat16_bits_and_pairing(self):
        tensor = torch.randn(8, 768, dtype=torch.bfloat16)
        request = decode_chat_request(
            encode_chat_request(tensor, "question", compression="zstd")
        )

        self.assertTrue(
            torch.equal(tensor.view(torch.int16), request.tensor.view(torch.int16))
        )
        self.assertEqual(request.split_point, "vision_pre_projector")
        self.assertEqual(request.encoder_model, "google/gemma-4-E4B-it")

    def test_rejects_wrong_width(self):
        with self.assertRaisesRegex(ValueError, "768-wide"):
            encode_chat_request(torch.zeros(8, 2560, dtype=torch.bfloat16), "q")

    def test_rejects_non_finite_values(self):
        tensor = torch.zeros(2, 768, dtype=torch.bfloat16)
        tensor[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            decode_chat_request(encode_chat_request(tensor, "q"))

    def test_rejects_pairing_tampering(self):
        payload = bytearray(
            encode_chat_request(torch.zeros(2, 768, dtype=torch.bfloat16), "q")
        )
        _, metadata_size, tensor_size = HEADER.unpack_from(payload)
        start = HEADER.size
        metadata = json.loads(payload[start : start + metadata_size])
        metadata["server_revision"] = "wrong"
        encoded = json.dumps(metadata, separators=(",", ":")).encode()
        tampered = HEADER.pack(b"CDG3", len(encoded), tensor_size)
        tampered += encoded + payload[start + metadata_size :]

        with self.assertRaisesRegex(ValueError, "server_revision"):
            decode_chat_request(bytes(tampered))

    def test_gateway_can_pin_a_deployed_server_revision(self):
        payload = encode_chat_request(torch.zeros(2, 768, dtype=torch.bfloat16), "q")
        with self.assertRaisesRegex(ValueError, "server_revision"):
            decode_chat_request(payload, expected_server_revision="different-release")

    def test_rejects_non_object_metadata(self):
        metadata = b"[]"
        payload = HEADER.pack(b"CDG3", len(metadata), 0) + metadata
        with self.assertRaisesRegex(ValueError, "must be an object"):
            decode_chat_request(payload)


if __name__ == "__main__":
    unittest.main()
