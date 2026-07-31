from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForMultimodalLM, AutoProcessor

from . import MODEL_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export client-only and server-only Gemma artifacts"
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client_dir = args.output / "client"
    server_dir = args.output / "server"
    client_dir.mkdir(parents=True, exist_ok=True)
    server_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
    )
    model.eval()

    vision = model.model.embed_vision
    if vision is None:
        raise RuntimeError("The selected model has no unified image embedder")

    vision_state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in vision.state_dict().items()
    }
    save_file(vision_state, client_dir / "vision.safetensors")
    model.config.save_pretrained(client_dir)
    processor.image_processor.save_pretrained(client_dir)

    vision_parameters = sum(parameter.numel() for parameter in vision.parameters())

    # The server keeps multimodal masking logic but has no pixel/audio embedding weights.
    model.model.embed_vision = None
    model.model.embed_audio = None
    model.config.vision_config = None
    model.config.audio_config = None
    model.model.config.vision_config = None
    model.model.config.audio_config = None
    model.save_pretrained(server_dir, max_shard_size=args.max_shard_size)
    processor.tokenizer.save_pretrained(server_dir)

    print(f"Client artifact: {client_dir}")
    print(f"Server artifact: {server_dir}")
    print(f"Client vision parameters: {vision_parameters:,}")
