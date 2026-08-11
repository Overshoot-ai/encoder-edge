from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
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

    vision_projector = model.model.embed_vision
    vision_tower = getattr(model.model, "vision_tower", None)
    if vision_projector is None:
        raise RuntimeError("The selected model has no unified image embedder")

    if vision_tower is None:
        vision_state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in vision_projector.state_dict().items()
        }
    else:
        vision_state = {
            **{
                f"vision_tower.{name}": tensor.detach().cpu().contiguous()
                for name, tensor in vision_tower.state_dict().items()
            },
            **{
                f"embed_vision.{name}": tensor.detach().cpu().contiguous()
                for name, tensor in vision_projector.state_dict().items()
            },
        }
    save_file(vision_state, client_dir / "vision.safetensors")
    model.config.save_pretrained(client_dir)
    processor.image_processor.save_pretrained(client_dir)

    vision_parameters = sum(
        parameter.numel() for parameter in vision_projector.parameters()
    )
    if vision_tower is not None:
        vision_parameters += sum(
            parameter.numel() for parameter in vision_tower.parameters()
        )

    # The server keeps multimodal masking logic but has no pixel/audio embedding weights.
    model.model.embed_vision = None
    if vision_tower is not None:
        model.model.vision_tower = None
    model.model.embed_audio = None
    if getattr(model.model, "audio_tower", None) is not None:
        model.model.audio_tower = None
    model.config.vision_config = None
    model.config.audio_config = None
    model.model.config.vision_config = None
    model.model.config.audio_config = None
    model.save_pretrained(server_dir, max_shard_size=args.max_shard_size)

    index_path = server_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map = index["weight_map"]
    else:
        model_path = server_dir / "model.safetensors"
        with safe_open(model_path, framework="pt") as saved:
            weight_map = {name: model_path.name for name in saved.keys()}
        index = {"metadata": {}, "weight_map": weight_map}

    # save_pretrained omits aliased tensors, while serving runtimes may expect
    # every checkpoint key explicitly. Preserve omitted aliases in a small shard.
    missing_state = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in model.state_dict().items()
        if name not in weight_map
    }
    source_dir = Path(args.model)
    if not source_dir.exists():
        source_dir = Path(
            snapshot_download(
                args.model,
                revision=args.revision,
                allow_patterns=["*.safetensors", "*.safetensors.index.json"],
            )
        )
    multimodal_prefixes = (
        "model.vision_tower.",
        "model.embed_vision.",
        "model.audio_tower.",
        "model.embed_audio.",
    )
    for source_path in source_dir.glob("*.safetensors"):
        with safe_open(source_path, framework="pt") as source:
            for name in source.keys():
                if (
                    name not in weight_map
                    and name not in missing_state
                    and not name.startswith(multimodal_prefixes)
                ):
                    missing_state[name] = source.get_tensor(name).clone()
    if missing_state:
        extra_name = "model-extra.safetensors"
        save_file(missing_state, server_dir / extra_name)
        weight_map.update({name: extra_name for name in missing_state})
        index["metadata"]["total_size"] = index["metadata"].get(
            "total_size", 0
        ) + sum(tensor.numel() * tensor.element_size() for tensor in missing_state.values())
        index_path.write_text(json.dumps(index, indent=2) + "\n")

    processor.tokenizer.save_pretrained(server_dir)

    print(f"Client artifact: {client_dir}")
    print(f"Server artifact: {server_dir}")
    print(f"Client vision parameters: {vision_parameters:,}")
