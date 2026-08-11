import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from .prepare_artifact import main


class PrepareArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.client = root / "client"
        self.server = root / "server"
        self.output = root / "output"
        self.client.mkdir()
        self.server.mkdir()
        (self.client / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "gemma4",
                    "vision_config": {"hidden_size": 768},
                }
            )
        )
        (self.server / "config.json").write_text(
            json.dumps({"text_config": {"hidden_size": 2560}})
        )
        save_file(
            {"embed_vision.weight": torch.zeros((2, 2))},
            self.client / "vision.safetensors",
        )
        save_file(
            {"language_model.weight": torch.zeros((2, 2))},
            self.server / "model.safetensors",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def run_main(self, *extra: str) -> None:
        arguments = [
            "prepare_artifact.py",
            "--client",
            str(self.client),
            "--server",
            str(self.server),
            "--output",
            str(self.output),
            *extra,
        ]
        with patch.object(sys, "argv", arguments):
            main()

    def test_requires_projector_for_standard_gemma4(self) -> None:
        with self.assertRaises(SystemExit):
            self.run_main()
        self.assertFalse(self.output.exists())

    def test_builds_index_for_unsharded_checkpoint(self) -> None:
        self.run_main("--include-vision-projector")

        index = json.loads(
            (self.output / "model.safetensors.index.json").read_text()
        )
        self.assertEqual(
            index["weight_map"],
            {
                "language_model.weight": "model.safetensors",
                "embed_vision.weight": "vision-projector.safetensors",
            },
        )
        config = json.loads((self.output / "config.json").read_text())
        self.assertEqual(
            config["architectures"],
            ["CrossDeviceGemma4ForConditionalGeneration"],
        )

        with self.assertRaises(SystemExit):
            self.run_main("--include-vision-projector")


if __name__ == "__main__":
    unittest.main()
