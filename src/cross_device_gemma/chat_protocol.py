from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass

import torch


MAGIC = b"CDG3"
CONTENT_TYPE = "application/x-cross-device-gemma"
HEADER = struct.Struct("!4sIQ")
MAX_METADATA_BYTES = 64 * 1024
MAX_TENSOR_BYTES = 32 * 1024 * 1024

ENCODER_MODEL = "google/gemma-4-E4B-it"
ENCODER_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
PROCESSOR_REVISION = ENCODER_REVISION
SERVED_MODEL = "gemma-4-e4b-optimized"
SERVER_REVISION = "gemma-4-e4b-h200-projector-r1"
SPLIT_POINT = "vision_pre_projector"
FEATURE_WIDTH = 768


@dataclass(frozen=True)
class ChatRequest:
    served_model: str
    encoder_model: str
    encoder_revision: str
    processor_revision: str
    server_revision: str
    split_point: str
    question: str
    tensor: torch.Tensor
    max_tokens: int


def pairing_metadata() -> dict[str, str]:
    return {
        "served_model": SERVED_MODEL,
        "encoder_model": ENCODER_MODEL,
        "encoder_revision": ENCODER_REVISION,
        "processor_revision": PROCESSOR_REVISION,
        "server_revision": SERVER_REVISION,
        "split_point": SPLIT_POINT,
    }


def encode_chat_request(
    tensor: torch.Tensor,
    question: str,
    max_tokens: int = 128,
    compression: str | None = None,
) -> bytes:
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype != torch.bfloat16 or tensor.ndim != 2:
        raise ValueError("Expected a two-dimensional BF16 visual tensor")
    if tensor.shape[1] != FEATURE_WIDTH:
        raise ValueError(f"Expected {FEATURE_WIDTH}-wide pre-projector features")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question is required")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 1024:
        raise ValueError("max_tokens must be between 1 and 1024")

    tensor_bytes = tensor.view(torch.uint8).numpy().tobytes()
    if len(tensor_bytes) > MAX_TENSOR_BYTES:
        raise ValueError("Visual tensor is too large")
    wire_bytes = tensor_bytes
    compression_metadata = {}
    if compression is not None:
        if compression != "zstd":
            raise ValueError("Unsupported visual tensor compression")
        import zstandard

        compressed = zstandard.ZstdCompressor(level=1, write_checksum=True).compress(
            tensor_bytes
        )
        if len(compressed) < len(tensor_bytes):
            wire_bytes = compressed
            compression_metadata = {
                "compression": "zstd",
                "uncompressed_size": len(tensor_bytes),
            }

    metadata = json.dumps(
        {
            **pairing_metadata(),
            "question": question,
            "shape": list(tensor.shape),
            "dtype": "bfloat16",
            "byte_order": "little",
            "layout": "row_major",
            "max_tokens": max_tokens,
            **compression_metadata,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    if len(metadata) > MAX_METADATA_BYTES:
        raise ValueError("Request metadata is too large")
    return HEADER.pack(MAGIC, len(metadata), len(wire_bytes)) + metadata + wire_bytes


def decode_chat_request(
    payload: bytes, expected_server_revision: str = SERVER_REVISION
) -> ChatRequest:
    if len(payload) < HEADER.size:
        raise ValueError("Request is missing its binary header")
    magic, metadata_size, tensor_size = HEADER.unpack_from(payload)
    if magic != MAGIC:
        raise ValueError("Invalid binary request magic")
    if metadata_size > MAX_METADATA_BYTES or tensor_size > MAX_TENSOR_BYTES:
        raise ValueError("Binary request exceeds configured limits")
    if len(payload) != HEADER.size + metadata_size + tensor_size:
        raise ValueError("Binary request length does not match its header")

    metadata_end = HEADER.size + metadata_size
    try:
        metadata = json.loads(payload[HEADER.size:metadata_end])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid request metadata") from error
    if not isinstance(metadata, dict):
        raise ValueError("Request metadata must be an object")
    expected_pairing = pairing_metadata()
    expected_pairing["server_revision"] = expected_server_revision
    for key, expected in expected_pairing.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Incompatible {key}")
    if metadata.get("dtype") != "bfloat16":
        raise ValueError("Only BF16 visual tensors are accepted")
    if metadata.get("byte_order") != "little" or metadata.get("layout") != "row_major":
        raise ValueError("Unsupported tensor byte order or layout")

    shape = metadata.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape)
        or shape[0] > 1120
        or shape[1] != FEATURE_WIDTH
    ):
        raise ValueError(f"Visual tensor shape must be [1..1120, {FEATURE_WIDTH}]")
    expected_tensor_size = math.prod(shape) * torch.bfloat16.itemsize
    if expected_tensor_size > MAX_TENSOR_BYTES:
        raise ValueError("Visual tensor is too large")

    question = metadata.get("question")
    max_tokens = metadata.get("max_tokens")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Invalid question")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
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
                wire_bytes, max_output_size=expected_tensor_size
            )
        except zstandard.ZstdError as error:
            raise ValueError("Invalid zstd visual tensor") from error
        if len(tensor_bytes) != expected_tensor_size:
            raise ValueError("Decompressed visual tensor has the wrong size")
    else:
        raise ValueError("Unsupported visual tensor compression")

    tensor = torch.frombuffer(bytearray(tensor_bytes), dtype=torch.bfloat16).reshape(shape)
    if not torch.isfinite(tensor).all().item():
        raise ValueError("Visual tensor contains non-finite values")
    return ChatRequest(
        served_model=metadata["served_model"],
        encoder_model=metadata["encoder_model"],
        encoder_revision=metadata["encoder_revision"],
        processor_revision=metadata["processor_revision"],
        server_revision=metadata["server_revision"],
        split_point=metadata["split_point"],
        question=question,
        tensor=tensor,
        max_tokens=max_tokens,
    )
