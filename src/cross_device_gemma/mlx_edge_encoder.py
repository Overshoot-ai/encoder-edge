from __future__ import annotations

import gc
import platform
import sys
import time
from importlib.metadata import version

import numpy as np
import torch


QUALIFIED_MODEL = "google/gemma-4-e4b-it"
QUALIFIED_REVISIONS = {"ee0ef6023621cff504d758262d4e04895a5af4a2"}
OPTIMIZATION_PROFILE = "gemma4-e4b-m4-qkv-r1"
QUALIFIED_MLX_VERSION = "0.32.0"
QUALIFIED_MLX_VLM_VERSION = "0.6.9"


def qualification_reason(source, config, device: str, dtype: str) -> str | None:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return "optimized MLX execution requires Apple Silicon"
    if source.path is not None:
        return "local model revisions are not optimization-qualified"
    if source.identifier.lower() != QUALIFIED_MODEL:
        return "model has no qualified MLX profile"
    if source.resolved_revision not in QUALIFIED_REVISIONS:
        return "model revision has not passed the MLX qualification gate"
    if device != "auto":
        return "an internal device override requested the compatibility backend"
    if dtype not in ("auto", "bfloat16"):
        return "the qualified MLX profile requires bfloat16 output"

    vision = config.vision_config
    expected = {
        "model_type": "gemma4",
        "vision_hidden_size": 768,
        "vision_layers": 16,
        "vision_heads": 12,
        "head_dim": 64,
        "patch_size": 16,
        "pooling_kernel_size": 3,
        "text_hidden_size": 2560,
    }
    actual = {
        "model_type": config.model_type,
        "vision_hidden_size": vision.hidden_size,
        "vision_layers": vision.num_hidden_layers,
        "vision_heads": vision.num_attention_heads,
        "head_dim": vision.head_dim,
        "patch_size": vision.patch_size,
        "pooling_kernel_size": vision.pooling_kernel_size,
        "text_hidden_size": config.text_config.hidden_size,
    }
    if actual != expected:
        return "model architecture does not match the qualified MLX profile"
    return None


class MLXGemma4E4BEncoder:
    def __init__(self, config, weights, split_point: str = "vision_post_projector"):
        mlx_version = version("mlx")
        mlx_vlm_version = version("mlx-vlm")
        if (mlx_version, mlx_vlm_version) != (
            QUALIFIED_MLX_VERSION,
            QUALIFIED_MLX_VLM_VERSION,
        ):
            raise RuntimeError(
                "qualified MLX profile requires "
                f"mlx=={QUALIFIED_MLX_VERSION} and "
                f"mlx-vlm=={QUALIFIED_MLX_VLM_VERSION}"
            )
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_vlm.models.gemma4.config import VisionConfig
        from mlx_vlm.models.gemma4.gemma4 import MultimodalEmbedder, VisionModel
        from mlx_vlm.models.gemma4.processing_gemma4 import Gemma4ImageProcessor
        from optimized_v2.mlx_vision_optimizations import (
            fuse_gemma4_qkv_epilogue,
            make_segmented_gemma4_encoder,
            optimize_gemma4_positions,
        )

        self.mx = mx
        self.mlx_version = mlx_version
        self.mlx_vlm_version = mlx_vlm_version
        mx.set_wired_limit(2 * 1024**3)

        module = nn.Module()
        module.vision_tower = VisionModel(
            VisionConfig.from_dict(config.vision_config.to_dict())
        )
        module.embed_vision = MultimodalEmbedder(
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
            config.vision_config.rms_norm_eps,
        )
        module.load_weights(str(weights), strict=True)
        mx.eval(module.parameters())

        optimize_gemma4_positions(module.vision_tower)
        fuse_gemma4_qkv_epilogue(module.vision_tower)
        if split_point not in ("vision_pre_projector", "vision_post_projector"):
            raise ValueError(f"Unsupported split point: {split_point}")
        self.split_point = split_point
        self.output_width = (
            config.vision_config.hidden_size
            if split_point == "vision_pre_projector"
            else config.text_config.hidden_size
        )
        self.encode_vision = make_segmented_gemma4_encoder(
            module.vision_tower,
            None if split_point == "vision_pre_projector" else module.embed_vision,
            segment_size=3,
            evaluate_segments=True,
        )
        self.module = module
        self.processor = Gemma4ImageProcessor(
            do_resize=True,
            do_rescale=True,
            rescale_factor=1 / 255,
            do_normalize=False,
            image_mean=[0.0, 0.0, 0.0],
            image_std=[1.0, 1.0, 1.0],
            do_convert_rgb=True,
            patch_size=config.vision_config.patch_size,
            max_soft_tokens=config.vision_config.default_output_length,
            pooling_kernel_size=config.vision_config.pooling_kernel_size,
        )

    def encode(self, image):
        started = time.perf_counter()
        inputs, _ = self.processor(images=[image.convert("RGB")])
        pixels = self.mx.array(inputs["pixel_values"])
        processed = time.perf_counter()
        features = self.encode_vision(pixels)
        self.mx.eval(features)
        self.mx.synchronize()
        encoded = time.perf_counter()

        if (
            features.ndim != 3
            or features.shape[0] != 1
            or features.shape[2] != self.output_width
        ):
            raise RuntimeError(f"Unexpected MLX vision output shape: {features.shape}")
        if features.dtype != self.mx.bfloat16:
            raise RuntimeError(f"Expected BF16 MLX vision output, got {features.dtype}")
        words = np.array(features[0].view(self.mx.uint16), copy=True)
        tensor = torch.from_numpy(words).view(torch.bfloat16)
        if not torch.isfinite(tensor.float()).all():
            raise RuntimeError("Encoder output contains non-finite values")
        return tensor, {
            "preprocess_ms": (processed - started) * 1000,
            "encode_ms": (encoded - processed) * 1000,
        }

    def metadata(self) -> dict:
        return {
            "backend": "mlx",
            "device": "mlx:gpu",
            "optimization_profile": OPTIMIZATION_PROFILE,
            "optimizations": [
                "gathered_positions",
                "fused_qkv_rope_layout",
                "segmented_layers_3_3_3_3_3_1",
                "wired_memory_2gib",
            ],
            "mlx_version": self.mlx_version,
            "mlx_vlm_version": self.mlx_vlm_version,
            "numerical_profile": "accuracy_qualified",
            "split_point": self.split_point,
        }

    def close(self) -> None:
        del self.encode_vision
        del self.module
        gc.collect()
        self.mx.clear_cache()
        self.mx.synchronize()
