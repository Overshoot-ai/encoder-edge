import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    for source in args.server.iterdir():
        if source.name != "config.json":
            target = args.output / source.name
            if not target.exists():
                target.symlink_to(source.resolve())

    server_config = json.loads((args.server / "config.json").read_text())
    client_config = json.loads((args.client / "config.json").read_text())
    server_config["architectures"] = [
        "CrossDeviceGemma4UnifiedForConditionalGeneration"
    ]
    server_config["vision_config"] = client_config["vision_config"]
    (args.output / "config.json").write_text(json.dumps(server_config, indent=2))

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
