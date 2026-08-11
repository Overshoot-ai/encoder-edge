import argparse
import base64
import io
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoConfig, AutoImageProcessor
from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedVisionEmbedder,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Gemma vision locally and send only its features"
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", default="gemma-4-12b")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16
    config = AutoConfig.from_pretrained(args.artifact)
    image_processor = AutoImageProcessor.from_pretrained(args.artifact)
    embedder = Gemma4UnifiedVisionEmbedder(config.vision_config, config.text_config)
    embedder.load_state_dict(load_file(args.artifact / "vision.safetensors"))
    embedder.to(device=device, dtype=dtype).eval()

    started = time.perf_counter()
    inputs = image_processor(
        images=Image.open(args.image).convert("RGB"),
        return_tensors="pt",
    )
    preprocess_ms = (time.perf_counter() - started) * 1000
    positions = inputs["image_position_ids"].to(device)
    if device.type == "mps":
        torch.mps.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        features = embedder(inputs["pixel_values"].to(device), positions)
    if device.type == "mps":
        torch.mps.synchronize()
    encode_ms = (time.perf_counter() - started) * 1000
    features = features[~positions.eq(-1).all(dim=-1)]
    started = time.perf_counter()
    buffer = io.BytesIO()
    torch.save(features.cpu().contiguous(), buffer)
    payload = json.dumps(
        {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_embeds",
                            "image_embeds": base64.b64encode(
                                buffer.getvalue()
                            ).decode(),
                        },
                        {"type": "text", "text": args.question},
                    ],
                }
            ],
            "max_tokens": 128,
            "temperature": 0,
        }
    ).encode()
    serialize_ms = (time.perf_counter() - started) * 1000
    request = Request(
        f"{args.server.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urlopen(request, timeout=300) as response:
        result = json.loads(response.read())
    print(result["choices"][0]["message"]["content"])
    print(
        json.dumps(
            {
                "client_preprocess_ms": preprocess_ms,
                "client_encode_ms": encode_ms,
                "client_serialize_ms": serialize_ms,
                "request_bytes": len(payload),
                "request_e2e_ms": (time.perf_counter() - started) * 1000,
                **result["metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
