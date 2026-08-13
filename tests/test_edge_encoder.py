from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from huggingface_hub import get_local_safetensors_metadata
from safetensors.torch import save_file
from transformers import AutoConfig

from cross_device_gemma.edge_encoder import (
    Gemma4Adapter,
    Gemma4UnifiedAdapter,
    ModelSource,
    UnsupportedModelError,
    materialize_weights,
    plan_weights,
    resolve_adapter,
)


def save_config(path: Path, model_type: str) -> None:
    (path / "config.json").write_text(json.dumps({"model_type": model_type}))


class EdgeEncoderTests(unittest.TestCase):
    def test_gemma4_preserves_all_projected_tokens(self):
        expected = torch.ones(12, 2816)

        def module(pixel_values, image_position_ids):
            return expected

        actual = Gemma4Adapter().encode(
            module,
            {
                "pixel_values": torch.empty(1),
                "image_position_ids": torch.empty(1),
            },
        )

        self.assertIs(actual, expected)

    def test_separate_encoder_selects_only_vision_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_config(root, "gemma4")
            save_file(
                {
                    "model.language_model.weight": torch.ones(32, 32),
                    "model.vision_tower.layer.weight": torch.arange(16).reshape(4, 4),
                    "model.embed_vision.embedding_projection.weight": torch.ones(4, 4),
                },
                root / "model.safetensors",
            )
            source = ModelSource(str(root), root, None, "local")
            adapter = Gemma4Adapter()
            plan = plan_weights(source, adapter)
            self.assertEqual(len(plan.selected_keys), 2)
            self.assertLess(plan.selected_bytes, plan.checkpoint_bytes)

            extracted = materialize_weights(source, adapter, root / "cache")
            from safetensors.torch import load_file

            state = load_file(extracted)
            self.assertEqual(
                set(state),
                {
                    "vision_tower.layer.weight",
                    "embed_vision.embedding_projection.weight",
                },
            )

    def test_unified_model_separates_image_embedding_layer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_config(root, "gemma4_unified")
            save_file(
                {
                    "model.language_model.weight": torch.ones(64, 64),
                    "model.vision_embedder.patch_dense.weight": torch.arange(24).reshape(4, 6),
                    "model.embed_vision.embedding_projection.weight": torch.ones(4, 4),
                },
                root / "model.safetensors",
            )
            source = ModelSource(str(root), root, None, "local")
            adapter = Gemma4UnifiedAdapter()
            extracted = materialize_weights(source, adapter, root / "cache")
            from safetensors.torch import load_file

            self.assertEqual(
                set(load_file(extracted)),
                {
                    "patch_dense.weight",
                    "multimodal_embedder.embedding_projection.weight",
                },
            )

    def test_model_without_vision_is_rejected_before_weight_planning(self):
        config = AutoConfig.for_model("gpt2")
        with patch(
            "cross_device_gemma.edge_encoder.plan_weights",
            side_effect=AssertionError("weights must not be inspected"),
        ):
            with self.assertRaisesRegex(UnsupportedModelError, "no vision encoder"):
                resolve_adapter(config)

    def test_full_download_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_config(root, "gemma4_unified")
            save_file(
                {"model.vision_embedder.patch_dense.weight": torch.ones(2, 2)},
                root / "model.safetensors",
            )
            source = ModelSource("org/model", None, None, "revision")
            adapter = Gemma4UnifiedAdapter()
            with (
                patch(
                    "cross_device_gemma.edge_encoder._safetensors_metadata",
                    return_value=get_local_safetensors_metadata(root),
                ),
                patch(
                    "cross_device_gemma.edge_encoder._read_header_size_remote",
                    return_value=8,
                ),
                patch(
                    "cross_device_gemma.edge_encoder._copy_remote_range",
                    side_effect=RuntimeError("range unsupported"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "--allow-full-download"):
                    materialize_weights(source, adapter, root / "cache")

    def test_failed_range_does_not_leave_a_partial_cache_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_file(
                {"model.vision_embedder.patch_dense.weight": torch.ones(2, 2)},
                root / "model.safetensors",
            )
            source = ModelSource("org/model", None, None, "revision")
            adapter = Gemma4UnifiedAdapter()

            def fail_after_write(*args):
                output = args[4]
                output.write(b"partial")
                raise RuntimeError("range unsupported")

            with (
                patch(
                    "cross_device_gemma.edge_encoder._safetensors_metadata",
                    return_value=get_local_safetensors_metadata(root),
                ),
                patch(
                    "cross_device_gemma.edge_encoder._read_header_size_remote",
                    return_value=8,
                ),
                patch(
                    "cross_device_gemma.edge_encoder._copy_remote_range",
                    side_effect=fail_after_write,
                ),
            ):
                with self.assertRaises(RuntimeError):
                    materialize_weights(source, adapter, root / "cache")
            self.assertFalse(any((root / "cache").rglob("vision.safetensors")))


if __name__ == "__main__":
    unittest.main()
