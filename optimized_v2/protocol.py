from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

MAGIC = b"CDG2"
CONTENT_TYPE = "application/x-cross-device-gemma"
HEADER = struct.Struct("!4sIQ")
MAX_METADATA_BYTES = 64 * 1024
MAX_TENSOR_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class BinaryRequest:
    model: str
    question: str
    tensor: torch.Tensor
    max_tokens: int


def encode_raw_request(
    tensor_bytes: bytes,
    shape: tuple[int, int],
    question: str,
    model: str,
    max_tokens: int = 128,
    compression: str | None = None,
) -> bytes:
    if len(shape) != 2 or not all(isinstance(size, int) and size > 0 for size in shape):
        raise ValueError("Expected a two-dimensional visual tensor shape")
    if math.prod(shape) * 2 != len(tensor_bytes):
        raise ValueError("Visual tensor shape does not match its BF16 byte length")
    if not question or not model:
        raise ValueError("Question and model are required")
    if not 1 <= max_tokens <= 1024:
        raise ValueError("max_tokens must be between 1 and 1024")

    if len(tensor_bytes) > MAX_TENSOR_BYTES:
        raise ValueError("Visual tensor is too large")
    wire_bytes = tensor_bytes
    compression_metadata = {}
    if compression is not None:
        if compression != "zstd":
            raise ValueError("Unsupported visual tensor compression")
        import zstandard

        compressed = zstandard.ZstdCompressor(
            level=1, write_checksum=True
        ).compress(tensor_bytes)
        if len(compressed) < len(tensor_bytes):
            wire_bytes = compressed
            compression_metadata = {
                "compression": "zstd",
                "uncompressed_size": len(tensor_bytes),
            }

    metadata = json.dumps(
        {
            "model": model,
            "question": question,
            "shape": list(shape),
            "dtype": "bfloat16",
            "max_tokens": max_tokens,
            **compression_metadata,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    if len(metadata) > MAX_METADATA_BYTES:
        raise ValueError("Request metadata is too large")
    return HEADER.pack(MAGIC, len(metadata), len(wire_bytes)) + metadata + wire_bytes


def encode_request(
    tensor: torch.Tensor,
    question: str,
    model: str,
    max_tokens: int = 128,
    compression: str | None = None,
) -> bytes:
    import torch

    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype != torch.bfloat16 or tensor.ndim != 2:
        raise ValueError("Expected a two-dimensional BF16 visual tensor")
    if not question or not model:
        raise ValueError("Question and model are required")
    if not 1 <= max_tokens <= 1024:
        raise ValueError("max_tokens must be between 1 and 1024")

    tensor_bytes = tensor.view(torch.uint8).numpy().tobytes()
    return encode_raw_request(
        tensor_bytes,
        tuple(tensor.shape),
        question,
        model,
        max_tokens,
        compression,
    )


def decode_request(payload: bytes) -> BinaryRequest:
    import torch

    if len(payload) < HEADER.size:
        raise ValueError("Request is missing its binary header")
    magic, metadata_size, tensor_size = HEADER.unpack_from(payload)
    if magic != MAGIC:
        raise ValueError("Invalid binary request magic")
    if metadata_size > MAX_METADATA_BYTES or tensor_size > MAX_TENSOR_BYTES:
        raise ValueError("Binary request exceeds configured limits")
    expected_size = HEADER.size + metadata_size + tensor_size
    if len(payload) != expected_size:
        raise ValueError("Binary request length does not match its header")

    metadata_end = HEADER.size + metadata_size
    metadata = json.loads(payload[HEADER.size:metadata_end])
    if metadata.get("dtype") != "bfloat16":
        raise ValueError("Only BF16 visual tensors are accepted")
    shape = metadata.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or not all(isinstance(size, int) and size > 0 for size in shape)
    ):
        raise ValueError("Invalid visual tensor shape")
    if shape[0] > 1120 or shape[1] > 16384:
        raise ValueError("Visual tensor shape exceeds configured limits")
    expected_tensor_size = math.prod(shape) * torch.bfloat16.itemsize
    if expected_tensor_size > MAX_TENSOR_BYTES:
        raise ValueError("Visual tensor is too large")

    model = metadata.get("model")
    question = metadata.get("question")
    max_tokens = metadata.get("max_tokens")
    if not isinstance(model, str) or not model:
        raise ValueError("Invalid model name")
    if not isinstance(question, str) or not question:
        raise ValueError("Invalid question")
    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or not 1 <= max_tokens <= 1024
    ):
        raise ValueError("Invalid max_tokens")

    wire_bytes = payload[metadata_end:]
    compression = metadata.get("compression")
    if compression is None:
        if tensor_size != expected_tensor_size:
            raise ValueError("Visual tensor shape does not match its byte length")
        tensor_bytes = wire_bytes
    elif compression == "zstd":
        if metadata.get("uncompressed_size") != expected_tensor_size:
            raise ValueError("Compressed visual tensor size does not match its shape")
        try:
            import zstandard

            tensor_bytes = zstandard.ZstdDecompressor().decompress(
                wire_bytes,
                max_output_size=expected_tensor_size,
            )
        except zstandard.ZstdError as error:
            raise ValueError("Invalid zstd visual tensor") from error
        if len(tensor_bytes) != expected_tensor_size:
            raise ValueError("Decompressed visual tensor has the wrong size")
    else:
        raise ValueError("Unsupported visual tensor compression")

    storage = bytearray(tensor_bytes)
    tensor = torch.frombuffer(storage, dtype=torch.bfloat16).reshape(shape)
    if not torch.isfinite(tensor).all().item():
        raise ValueError("Visual tensor contains non-finite values")
    return BinaryRequest(model, question, tensor, max_tokens)
