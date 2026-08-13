from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image

from cross_device_gemma.edge_encoder import EdgeEncoder, ModelSource, WeightPlan
from cross_device_gemma.mlx_edge_encoder import (
    QUALIFIED_REVISIONS,
    qualification_reason,
)


def config(**overrides):
    values = {
        "model_type": "gemma4",
        "vision_config": SimpleNamespace(
            hidden_size=768,
            num_hidden_layers=16,
            num_attention_heads=12,
            head_dim=64,
            patch_size=16,
            pooling_kernel_size=3,
        ),
        "text_config": SimpleNamespace(hidden_size=2560),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class MLXQualificationTests(unittest.TestCase):
    def setUp(self):
        revision = next(iter(QUALIFIED_REVISIONS))
        self.source = ModelSource("google/gemma-4-E4B-it", None, None, revision)

    @patch("cross_device_gemma.mlx_edge_encoder.platform.machine", return_value="arm64")
    @patch("cross_device_gemma.mlx_edge_encoder.sys.platform", "darwin")
    def test_exact_profile_is_qualified(self, _machine):
        self.assertIsNone(qualification_reason(self.source, config(), "auto", "auto"))

    @patch("cross_device_gemma.mlx_edge_encoder.platform.machine", return_value="x86_64")
    @patch("cross_device_gemma.mlx_edge_encoder.sys.platform", "linux")
    def test_non_apple_host_uses_compatibility_backend(self, _machine):
        self.assertIn(
            "Apple Silicon",
            qualification_reason(self.source, config(), "auto", "auto"),
        )

    @patch("cross_device_gemma.mlx_edge_encoder.platform.machine", return_value="arm64")
    @patch("cross_device_gemma.mlx_edge_encoder.sys.platform", "darwin")
    def test_other_model_or_revision_is_not_qualified(self, _machine):
        other_model = ModelSource("google/gemma-4-26B-A4B-it", None, None, "sha")
        other_revision = ModelSource(
            "google/gemma-4-E4B-it", None, None, "unqualified"
        )
        self.assertIn(
            "no qualified MLX profile",
            qualification_reason(other_model, config(), "auto", "auto"),
        )
        self.assertIn(
            "revision",
            qualification_reason(other_revision, config(), "auto", "auto"),
        )

    @patch("cross_device_gemma.mlx_edge_encoder.platform.machine", return_value="arm64")
    @patch("cross_device_gemma.mlx_edge_encoder.sys.platform", "darwin")
    def test_architecture_drift_fails_closed(self, _machine):
        changed = config()
        changed.vision_config.num_hidden_layers = 17
        self.assertIn(
            "architecture",
            qualification_reason(self.source, changed, "auto", "auto"),
        )

    @patch("cross_device_gemma.edge_encoder.materialize_weights")
    @patch("cross_device_gemma.edge_encoder.plan_weights")
    @patch("cross_device_gemma.edge_encoder.resolve_adapter")
    @patch("cross_device_gemma.edge_encoder.AutoConfig.from_pretrained")
    @patch("cross_device_gemma.edge_encoder.resolve_source")
    @patch("cross_device_gemma.mlx_edge_encoder.MLXGemma4E4BEncoder")
    @patch("cross_device_gemma.mlx_edge_encoder.qualification_reason", return_value=None)
    def test_runtime_failure_switches_permanently_to_pytorch(
        self,
        _qualified,
        mlx_type,
        resolve_source,
        auto_config,
        resolve_adapter,
        plan_weights,
        materialize_weights,
    ):
        mlx = Mock()
        mlx.encode.side_effect = RuntimeError("metal failure")
        mlx_type.return_value = mlx
        resolve_source.return_value = self.source
        auto_config.return_value = config()
        adapter = Mock()
        adapter.architecture = "separate"
        resolve_adapter.return_value = adapter
        plan_weights.return_value = WeightPlan(
            "gemma4", "separate", (), 1, 2, ("model.safetensors",)
        )
        materialize_weights.return_value = "vision.safetensors"

        encoder = EdgeEncoder("google/gemma-4-E4B-it")

        def initialize_pytorch():
            encoder.backend = "pytorch"
            encoder.device = torch.device("cpu")
            encoder.processor = lambda **_kwargs: {}

        encoder._initialize_pytorch = Mock(side_effect=initialize_pytorch)
        encoder._encode_inputs = Mock(return_value=torch.ones(2, 3))
        output, _ = encoder.encode(Image.new("RGB", (16, 16)))

        self.assertEqual(tuple(output.shape), (2, 3))
        self.assertIsNone(encoder._mlx)
        mlx.close.assert_called_once_with()
        encoder._initialize_pytorch.assert_called_once_with()
        encoder.encode(Image.new("RGB", (16, 16)))
        self.assertEqual(mlx.encode.call_count, 1)


if __name__ == "__main__":
    unittest.main()
