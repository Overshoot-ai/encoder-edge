import argparse
import json
import os
from pathlib import Path

from safetensors.torch import load_file, save_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct a full raw-image Gemma artifact for stock vLLM"
    )
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    excluded = {"config.json", "model.safetensors.index.json"}
    for source in args.server.iterdir():
        if source.name not in excluded:
            target = args.output / source.name
            if not target.exists():
                target.symlink_to(source.resolve())

    client_config = json.loads((args.client / "config.json").read_text())
    server_config = json.loads((args.server / "config.json").read_text())
    server_config["architectures"] = client_config["architectures"]
    server_config["model_type"] = client_config["model_type"]
    server_config["vision_config"] = client_config["vision_config"]
    server_config["audio_config"] = None
    (args.output / "config.json").write_text(json.dumps(server_config, indent=2))

    source_vision = load_file(args.client / "vision.safetensors")
    vision_weights = {}
    for name, tensor in source_vision.items():
        if client_config["model_type"] == "gemma4":
            target_name = name
        elif name.startswith("multimodal_embedder."):
            target_name = "model.embed_vision." + name.removeprefix(
                "multimodal_embedder."
            )
        else:
            target_name = f"model.vision_embedder.{name}"
        vision_weights[target_name] = tensor.contiguous()
    vision_filename = "vision-vllm.safetensors"
    save_file(vision_weights, args.output / vision_filename)

    index = json.loads((args.server / "model.safetensors.index.json").read_text())
    index["weight_map"].update(
        {name: vision_filename for name in vision_weights}
    )
    vision_size = sum(
        tensor.numel() * tensor.element_size() for tensor in vision_weights.values()
    )
    if "total_size" in index.get("metadata", {}):
        index["metadata"]["total_size"] += vision_size
    (args.output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2)
    )

    for name in ("processor_config.json", "preprocessor_config.json"):
        source = args.client / name
        target = args.output / name
        if source.exists() and not target.exists():
            os.symlink(source.resolve(), target)

    print(f"Full vLLM artifact: {args.output}")
    print(f"Vision tensors: {len(vision_weights)}")
    print(f"Vision bytes: {vision_size}")


if __name__ == "__main__":
    main()
