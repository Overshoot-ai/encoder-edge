from __future__ import annotations

import os
import unittest

from transformers import AutoConfig

from cross_device_gemma.edge_encoder import (
    UnsupportedModelError,
    plan_weights,
    resolve_adapter,
    resolve_source,
)


@unittest.skipUnless(
    os.environ.get("EDGE_ENCODER_HF_TESTS") == "1",
    "set EDGE_ENCODER_HF_TESTS=1 to query live Hugging Face metadata",
)
class HuggingFaceDiscoveryTests(unittest.TestCase):
    def plan(self, model: str):
        source = resolve_source(model)
        config = AutoConfig.from_pretrained(model, revision=source.resolved_revision)
        adapter = resolve_adapter(config)
        return source, adapter, plan_weights(source, adapter)

    def test_conventional_encoder_selects_only_vision_ranges(self):
        _, adapter, plan = self.plan("HuggingFaceTB/SmolVLM-256M-Instruct")
        self.assertEqual(adapter.architecture, "separate")
        self.assertEqual(plan.selected_bytes, 187_021_824)
        self.assertLess(plan.selected_bytes, plan.checkpoint_bytes)

    def test_unified_gemma_selects_embedding_path_only(self):
        _, adapter, plan = self.plan("google/gemma-4-12B-it")
        self.assertEqual(adapter.architecture, "unified_embedding")
        self.assertEqual(plan.selected_bytes, 99_844_608)
        self.assertEqual(plan.checkpoint_bytes, 23_919_549_408)

    def test_text_only_model_is_rejected(self):
        source = resolve_source("openai-community/gpt2")
        config = AutoConfig.from_pretrained(
            source.identifier, revision=source.resolved_revision
        )
        with self.assertRaisesRegex(UnsupportedModelError, "no vision encoder"):
            resolve_adapter(config)
