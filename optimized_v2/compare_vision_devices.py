import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoConfig, AutoImageProcessor
from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedVisionEmbedder,
)


def tensor_hash(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", choices=["mps", "cuda"], required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    config = AutoConfig.from_pretrained(args.artifact)
    processor = AutoImageProcessor.from_pretrained(args.artifact)
    embedder = Gemma4UnifiedVisionEmbedder(
        config.vision_config,
        config.text_config,
    )
    embedder.load_state_dict(load_file(args.artifact / "vision.safetensors"))
    embedder.to(device=device, dtype=torch.bfloat16).eval()
    inputs = processor(
        images=Image.open(args.image).convert("RGB"),
        return_tensors="pt",
    )
    pixel_hash = tensor_hash(inputs["pixel_values"])
    position_hash = tensor_hash(inputs["image_position_ids"])
    positions = inputs["image_position_ids"].to(device)
    with torch.inference_mode():
        features = embedder(inputs["pixel_values"].to(device), positions)
    if device.type == "mps":
        torch.mps.synchronize()
    else:
        torch.cuda.synchronize()
    features = features[~positions.eq(-1).all(dim=-1)].cpu().contiguous()

    if args.output:
        torch.save(features, args.output)

    result = {
        "device": args.device,
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "pixel_values_sha256": pixel_hash,
        "position_ids_sha256": position_hash,
    }
    if args.reference:
        reference = torch.load(
            args.reference,
            map_location="cpu",
            weights_only=True,
        )
        difference = (features.float() - reference.float()).abs()
        result.update(
            bits_equal=torch.equal(
                features.view(torch.int16), reference.view(torch.int16)
            ),
            values_equal=torch.equal(features, reference),
            differing_values=(features != reference).sum().item(),
            total_values=features.numel(),
            max_absolute_difference=difference.max().item(),
            mean_absolute_difference=difference.mean().item(),
            cosine_similarity=torch.nn.functional.cosine_similarity(
                features.float().flatten(),
                reference.float().flatten(),
                dim=0,
            ).item(),
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
