import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

import torch
from PIL import Image
from safetensors.torch import load_file
from safetensors.torch import save as save_tensors
from transformers import AutoConfig, AutoProcessor
from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedVisionEmbedder


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemma vision locally and send only its features")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "mps" else torch.float32
    config = AutoConfig.from_pretrained(args.artifact)
    processor = AutoProcessor.from_pretrained(args.artifact)
    embedder = Gemma4UnifiedVisionEmbedder(config.vision_config, config.text_config)
    embedder.load_state_dict(load_file(args.artifact / "vision.safetensors"))
    embedder.to(device=device, dtype=dtype).eval()

    inputs = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image", "image": Image.open(args.image).convert("RGB")},
            {"type": "text", "text": args.question},
        ]}],
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    )
    positions = inputs["image_position_ids"].to(device)
    with torch.inference_mode():
        features = embedder(inputs["pixel_values"].to(device), positions)
    features = features[~positions.eq(-1).all(dim=-1)]
    payload = save_tensors({
        "image_features": features.cpu().contiguous(),
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "mm_token_type_ids": inputs["mm_token_type_ids"],
    })
    request = Request(
        f"{args.server.rstrip('/')}/generate",
        data=payload,
        headers={"Content-Type": "application/x-safetensors"},
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        print(json.loads(response.read())["answer"])
