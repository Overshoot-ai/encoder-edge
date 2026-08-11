import struct

import pytest
import torch

from .protocol import HEADER, decode_request, encode_request


@pytest.mark.parametrize("compression", (None, "zstd"))
def test_protocol_preserves_bfloat16_bits(compression):
    tensor = torch.randn(32, 768, dtype=torch.bfloat16)
    payload = encode_request(
        tensor,
        "question",
        "model",
        compression=compression,
    )

    request = decode_request(payload)

    assert torch.equal(tensor.view(torch.int16), request.tensor.view(torch.int16))


def test_zstd_rejects_invalid_compressed_payload():
    tensor = torch.randn(32, 768, dtype=torch.bfloat16)
    payload = bytearray(
        encode_request(tensor, "question", "model", compression="zstd")
    )
    _, metadata_size, _ = HEADER.unpack_from(payload)
    payload[HEADER.size + metadata_size] ^= 0xFF

    with pytest.raises(ValueError, match="zstd"):
        decode_request(bytes(payload))


def test_protocol_rejects_unsupported_compression():
    tensor = torch.randn(4, 768, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="Unsupported"):
        encode_request(tensor, "question", "model", compression="gzip")
