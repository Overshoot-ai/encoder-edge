import argparse
import json
import platform
from pathlib import Path

import torch
from safetensors.torch import load_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prove client/server artifact placement"
    )
    parser.add_argument("role", choices=["client", "server"])
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()

    result = {
        "role": args.role,
        "hostname": platform.node(),
        "artifact": str(args.artifact.resolve()),
    }
    if args.role == "client":
        state = load_file(args.artifact / "vision.safetensors")
        result.update(
            device="mps" if torch.backends.mps.is_available() else "cpu",
            vision_tensors=len(state),
            vision_parameters=sum(tensor.numel() for tensor in state.values()),
        )
        passed = result["vision_parameters"] == 49_922_304
    else:
        index = next(args.artifact.glob("*.safetensors.index.json"))
        keys = list(json.loads(index.read_text())["weight_map"])
        vision = [
            key for key in keys if "vision" in key.lower() or "image" in key.lower()
        ]
        audio = [key for key in keys if "audio" in key.lower()]
        result.update(
            device=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            checkpoint_tensors=len(keys),
            vision_tensors=len(vision),
            audio_tensors=len(audio),
        )
        passed = not vision and not audio and "H200" in result["device"]

    result["proof"] = "PASS" if passed else "FAIL"
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
