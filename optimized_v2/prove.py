import base64
import io
import json

import torch

from .gateway import build_vllm_payload
from .protocol import decode_request, encode_request


def main() -> None:
    torch.manual_seed(7)
    original = torch.randn(264, 3840, dtype=torch.bfloat16)
    binary_payload = encode_request(
        original,
        "What is shown in this image?",
        "gemma-4-12b-optimized",
    )
    decoded = decode_request(binary_payload)
    bits_equal = torch.equal(
        original.view(torch.int16),
        decoded.tensor.view(torch.int16),
    )

    vllm_payload = json.loads(build_vllm_payload(decoded))
    encoded = vllm_payload["messages"][0]["content"][0]["image_embeds"]
    reconstructed = torch.load(
        io.BytesIO(base64.b64decode(encoded)),
        map_location="cpu",
        weights_only=True,
    )
    gateway_bits_equal = torch.equal(
        original.view(torch.int16),
        reconstructed.view(torch.int16),
    )
    raw_bytes = original.numel() * original.element_size()
    base64_bytes = 4 * ((raw_bytes + 2) // 3)
    result = {
        "shape": list(original.shape),
        "dtype": str(original.dtype),
        "raw_tensor_bytes": raw_bytes,
        "binary_request_bytes": len(binary_payload),
        "base64_tensor_bytes": base64_bytes,
        "mac_to_gateway_bits_equal": bits_equal,
        "gateway_to_vllm_bits_equal": gateway_bits_equal,
        "proof": "PASS" if bits_equal and gateway_bits_equal else "FAIL",
    }
    print(json.dumps(result, indent=2))
    if result["proof"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
