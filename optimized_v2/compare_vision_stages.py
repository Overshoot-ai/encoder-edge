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


def run_stages(
    embedder: Gemma4UnifiedVisionEmbedder,
    pixel_values: torch.Tensor,
    positions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    stages = {"input": pixel_values}
    hidden = embedder.patch_ln1(pixel_values.to(embedder.patch_dense.weight.dtype))
    stages["patch_ln1"] = hidden
    hidden = embedder.patch_dense(hidden)
    stages["patch_dense"] = hidden
    hidden = embedder.patch_ln2(hidden)
    stages["patch_ln2"] = hidden

    clamped = positions.clamp(min=0).long()
    valid = (positions != -1).to(embedder.pos_embedding.dtype).unsqueeze(-1)
    axes = torch.arange(2, device=positions.device)
    positional = (embedder.pos_embedding[clamped, axes] * valid).sum(-2)
    stages["positional_embedding"] = positional
    hidden = hidden + positional
    stages["position_add"] = hidden
    hidden = embedder.pos_norm(hidden)
    stages["pos_norm"] = hidden

    projector = embedder.multimodal_embedder
    hidden = projector.embedding_pre_projection_norm(hidden)
    stages["projection_rms_norm"] = hidden
    hidden = projector.embedding_projection(hidden)
    stages["projection"] = hidden
    return {name: tensor.detach().cpu().contiguous() for name, tensor in stages.items()}


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    difference = (actual.float() - expected.float()).abs()
    return {
        "bits_equal": torch.equal(
            actual.view(torch.int16),
            expected.view(torch.int16),
        ),
        "differing_values": (actual != expected).sum().item(),
        "total_values": actual.numel(),
        "max_absolute_difference": difference.max().item(),
        "mean_absolute_difference": difference.mean().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", choices=["mps", "cuda"], required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()

    torch.use_deterministic_algorithms(True)
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
    with torch.inference_mode():
        stages = run_stages(
            embedder,
            inputs["pixel_values"].to(device),
            inputs["image_position_ids"].to(device),
        )
    if device.type == "mps":
        torch.mps.synchronize()
    else:
        torch.cuda.synchronize()

    if args.output:
        torch.save(stages, args.output)

    result = {
        "device": args.device,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "stages": {
            name: {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": tensor_hash(tensor),
            }
            for name, tensor in stages.items()
        },
    }
    if args.reference:
        reference = torch.load(
            args.reference,
            map_location="cpu",
            weights_only=True,
        )
        result["comparison"] = {
            name: compare(tensor, reference[name])
            for name, tensor in stages.items()
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
