import argparse
import json
import os
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-vision-projector",
        action="store_true",
        help="Add the vision projector for 768D pre-projector inputs",
    )
    args = parser.parse_args()

    server_config = json.loads((args.server / "config.json").read_text())
    client_config = json.loads((args.client / "config.json").read_text())
    is_gemma4 = client_config.get("model_type") == "gemma4"
    if is_gemma4 and not args.include_vision_projector:
        parser.error("standard Gemma 4 requires --include-vision-projector")
    if args.include_vision_projector and not is_gemma4:
        parser.error("--include-vision-projector is supported only for Gemma 4")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be empty")
    args.output.mkdir(parents=True, exist_ok=True)

    excluded = {"config.json"}
    if args.include_vision_projector:
        excluded.add("model.safetensors.index.json")
    for source in args.server.iterdir():
        if source.name not in excluded:
            target = args.output / source.name
            if not target.exists():
                target.symlink_to(source.resolve())

    architecture = (
        "CrossDeviceGemma4ForConditionalGeneration"
        if is_gemma4
        else "CrossDeviceGemma4UnifiedForConditionalGeneration"
    )
    server_config["architectures"] = [architecture]
    server_config["vision_config"] = client_config["vision_config"]
    (args.output / "config.json").write_text(json.dumps(server_config, indent=2))

    if args.include_vision_projector:
        projector_weights = {}
        with safe_open(args.client / "vision.safetensors", framework="pt") as source:
            for name in source.keys():
                if name.startswith("embed_vision."):
                    projector_weights[name] = source.get_tensor(name).contiguous()
        if not projector_weights:
            raise RuntimeError("Client artifact has no embed_vision projector weights")

        projector_filename = "vision-projector.safetensors"
        save_file(projector_weights, args.output / projector_filename)
        index_path = args.server / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
        else:
            model_path = args.server / "model.safetensors"
            if not model_path.exists():
                raise RuntimeError("Server artifact has no Safetensors weights")
            with safe_open(model_path, framework="pt") as source:
                index = {
                    "metadata": {},
                    "weight_map": {
                        name: model_path.name for name in source.keys()
                    },
                }
        duplicates = projector_weights.keys() & index["weight_map"].keys()
        if duplicates:
            raise RuntimeError(
                f"Server artifact already contains projector weights: {sorted(duplicates)}"
            )
        index["weight_map"].update(
            {name: projector_filename for name in projector_weights}
        )
        projector_size = sum(
            tensor.numel() * tensor.element_size()
            for tensor in projector_weights.values()
        )
        if "total_size" in index.get("metadata", {}):
            index["metadata"]["total_size"] += projector_size
        output_index = args.output / "model.safetensors.index.json"
        if output_index.is_symlink():
            output_index.unlink()
        output_index.write_text(json.dumps(index, indent=2) + "\n")

    for name in (
        "processor_config.json",
        "preprocessor_config.json",
        "chat_template.jinja",
    ):
        source = args.client / name
        target = args.output / name
        if source.exists() and not target.exists():
            os.symlink(source.resolve(), target)


if __name__ == "__main__":
    main()
