from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from huggingface_hub import (
    HfApi,
    get_local_safetensors_metadata,
    get_safetensors_metadata,
    hf_hub_download,
    hf_hub_url,
)
from huggingface_hub.utils import build_hf_headers, get_session
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoConfig, AutoImageProcessor


class UnsupportedModelError(ValueError):
    pass


@dataclass(frozen=True)
class ModelSource:
    identifier: str
    path: Path | None
    revision: str | None
    resolved_revision: str


@dataclass(frozen=True)
class WeightPlan:
    model_type: str
    architecture: str
    selected_keys: tuple[str, ...]
    selected_bytes: int
    checkpoint_bytes: int
    files: tuple[str, ...]


class EncoderAdapter:
    model_types: tuple[str, ...] = ()
    architecture = "separate"
    prefixes: tuple[str, ...] = ()

    def select(self, key: str) -> bool:
        return key.startswith(self.prefixes)

    def map_key(self, key: str) -> str:
        raise NotImplementedError

    def build(self, config):
        raise NotImplementedError

    def encode(self, module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class Gemma4UnifiedAdapter(EncoderAdapter):
    model_types = ("gemma4_unified",)
    architecture = "unified_embedding"
    prefixes = ("model.vision_embedder.", "model.embed_vision.")

    def map_key(self, key: str) -> str:
        if key.startswith("model.vision_embedder."):
            return key.removeprefix("model.vision_embedder.")
        return "multimodal_embedder." + key.removeprefix("model.embed_vision.")

    def build(self, config):
        from transformers.models.gemma4_unified.modeling_gemma4_unified import (
            Gemma4UnifiedVisionEmbedder,
        )

        return Gemma4UnifiedVisionEmbedder(config.vision_config, config.text_config)

    def encode(self, module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        positions = inputs["image_position_ids"]
        features = module(inputs["pixel_values"], positions)
        return features[~positions.eq(-1).all(dim=-1)]


class Gemma4Adapter(EncoderAdapter):
    model_types = ("gemma4",)
    prefixes = ("model.vision_tower.", "model.embed_vision.")

    def map_key(self, key: str) -> str:
        return key.removeprefix("model.")

    def build(self, config):
        from torch import nn
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4MultimodalEmbedder,
            Gemma4VisionModel,
        )

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_tower = Gemma4VisionModel(config.vision_config)
                self.embed_vision = Gemma4MultimodalEmbedder(
                    config.vision_config, config.text_config
                )

            def forward(self, pixel_values, image_position_ids):
                hidden = self.vision_tower(
                    pixel_values=pixel_values,
                    pixel_position_ids=image_position_ids,
                ).last_hidden_state
                return self.embed_vision(inputs_embeds=hidden)

        return Encoder()

    def encode(self, module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return module(inputs["pixel_values"], inputs["image_position_ids"])


class Qwen25VLAdapter(EncoderAdapter):
    model_types = ("qwen2_5_vl",)
    prefixes = ("visual.",)

    def map_key(self, key: str) -> str:
        return key.removeprefix("visual.")

    def build(self, config):
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VisionTransformerPretrainedModel,
        )

        return Qwen2_5_VisionTransformerPretrainedModel(config.vision_config)

    def encode(self, module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return module(
            inputs["pixel_values"], grid_thw=inputs["image_grid_thw"]
        ).pooler_output


class Idefics3Adapter(EncoderAdapter):
    model_types = ("idefics3",)
    prefixes = ("model.vision_model.", "model.connector.")

    def map_key(self, key: str) -> str:
        return key.removeprefix("model.")

    def build(self, config):
        from torch import nn
        from transformers.models.idefics3.modeling_idefics3 import (
            Idefics3Connector,
            Idefics3VisionTransformer,
        )

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_model = Idefics3VisionTransformer(config.vision_config)
                self.connector = Idefics3Connector(config)

            def forward(self, pixel_values, pixel_attention_mask):
                batch, images, channels, height, width = pixel_values.shape
                pixels = pixel_values.view(batch * images, channels, height, width)
                real = (pixels == 0).sum(dim=(-1, -2, -3)) != pixels[0].numel()
                pixels = pixels[real].contiguous()
                mask = pixel_attention_mask.view(batch * images, height, width)[real]
                patch = config.vision_config.patch_size
                mask = mask.unfold(1, patch, patch).unfold(2, patch, patch)
                patch_mask = mask.sum(dim=(-1, -2)) > 0
                hidden = self.vision_model(
                    pixel_values=pixels,
                    patch_attention_mask=patch_mask,
                ).last_hidden_state
                return self.connector(hidden)

        return Encoder()

    def encode(self, module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return module(inputs["pixel_values"], inputs["pixel_attention_mask"])


ADAPTERS = tuple(
    adapter()
    for adapter in (
        Gemma4UnifiedAdapter,
        Gemma4Adapter,
        Qwen25VLAdapter,
        Idefics3Adapter,
    )
)


def resolve_adapter(config) -> EncoderAdapter:
    if getattr(config, "vision_config", None) is None:
        raise UnsupportedModelError(
            f"{config.model_type!r} has no vision encoder; refusing to download weights"
        )
    for adapter in ADAPTERS:
        if config.model_type in adapter.model_types:
            return adapter
    raise UnsupportedModelError(
        f"Vision model type {config.model_type!r} is not supported yet"
    )


def parse_model_reference(value: str, revision: str | None = None) -> tuple[str, str | None]:
    value = value.removeprefix("hf://")
    path = Path(value).expanduser()
    if path.exists():
        return str(path.resolve()), revision
    if revision is None and "@" in value:
        value, revision = value.rsplit("@", 1)
    return value, revision


def resolve_source(value: str, revision: str | None = None, token=None) -> ModelSource:
    identifier, revision = parse_model_reference(value, revision)
    path = Path(identifier)
    if path.exists():
        return ModelSource(identifier, path, revision, revision or "local")
    info = HfApi(token=token).model_info(identifier, revision=revision)
    return ModelSource(identifier, None, revision, info.sha)


def _safetensors_metadata(source: ModelSource, token=None):
    if source.path:
        return get_local_safetensors_metadata(source.path)
    return get_safetensors_metadata(
        source.identifier,
        revision=source.resolved_revision,
        token=token,
        timeout=120,
    )


def plan_weights(source: ModelSource, adapter: EncoderAdapter, token=None) -> WeightPlan:
    metadata = _safetensors_metadata(source, token)
    selected = tuple(sorted(key for key in metadata.weight_map if adapter.select(key)))
    if not selected:
        raise UnsupportedModelError(
            f"{source.identifier!r} declares vision support but no recognized vision weights were found"
        )
    selected_bytes = 0
    for key in selected:
        filename = metadata.weight_map[key]
        selected_bytes += metadata.files_metadata[filename].tensors[key].data_offsets[1]
        selected_bytes -= metadata.files_metadata[filename].tensors[key].data_offsets[0]
    if source.path:
        checkpoint_bytes = sum((source.path / name).stat().st_size for name in metadata.files_metadata)
    else:
        info = HfApi(token=token).model_info(
            source.identifier, revision=source.resolved_revision, files_metadata=True
        )
        sizes = {item.rfilename: item.size for item in info.siblings}
        checkpoint_bytes = sum(sizes.get(name, 0) or 0 for name in metadata.files_metadata)
    return WeightPlan(
        model_type=adapter.model_types[0],
        architecture=adapter.architecture,
        selected_keys=selected,
        selected_bytes=selected_bytes,
        checkpoint_bytes=checkpoint_bytes,
        files=tuple(sorted({metadata.weight_map[key] for key in selected})),
    )


def default_cache_dir() -> Path:
    root = os.environ.get("XDG_CACHE_HOME")
    return Path(root).expanduser() / "edge-encoder" if root else Path.home() / ".cache/edge-encoder"


def _cache_path(cache_dir: Path, source: ModelSource, adapter: EncoderAdapter) -> Path:
    identity = hashlib.sha256(
        f"{source.identifier}@{source.resolved_revision}:{adapter.model_types[0]}".encode()
    ).hexdigest()[:20]
    return cache_dir / "models" / identity / "vision.safetensors"


def _merge_intervals(items):
    groups = []
    for item in sorted(items, key=lambda value: value[1]):
        start, end = item[1:3]
        if groups and start == groups[-1][-1][2]:
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups


def _read_header_size_local(path: Path) -> int:
    with path.open("rb") as handle:
        return struct.unpack("<Q", handle.read(8))[0]


def _read_header_size_remote(source: ModelSource, filename: str, token=None) -> int:
    response = get_session().get(
        hf_hub_url(source.identifier, filename, revision=source.resolved_revision),
        headers={**build_hf_headers(token=token), "Range": "bytes=0-7"},
        follow_redirects=True,
        timeout=120,
    )
    response.raise_for_status()
    if len(response.content) != 8:
        raise RuntimeError(f"{filename} did not honor the Safetensors header range request")
    return struct.unpack("<Q", response.content)[0]


def _copy_local_range(source: Path, start: int, end: int, output) -> None:
    with source.open("rb") as handle:
        handle.seek(start)
        remaining = end - start
        while remaining:
            chunk = handle.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise EOFError(f"Unexpected end of {source}")
            output.write(chunk)
            remaining -= len(chunk)


def _copy_remote_range(
    source: ModelSource, filename: str, start: int, end: int, output, token=None
) -> None:
    headers = {
        **build_hf_headers(token=token),
        "Range": f"bytes={start}-{end - 1}",
    }
    with get_session().stream(
        "GET",
        hf_hub_url(source.identifier, filename, revision=source.resolved_revision),
        headers=headers,
        follow_redirects=True,
        timeout=300,
    ) as response:
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError(f"{filename} does not support selective byte-range downloads")
        written = 0
        for chunk in response.iter_bytes(8 * 1024 * 1024):
            output.write(chunk)
            written += len(chunk)
        if written != end - start:
            raise RuntimeError(f"Short byte-range response from {filename}: {written} bytes")


def materialize_weights(
    source: ModelSource,
    adapter: EncoderAdapter,
    cache_dir: Path | None = None,
    token=None,
    progress: Callable[[str], None] | None = None,
    allow_full_download: bool = False,
) -> Path:
    cache_dir = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    destination = _cache_path(cache_dir, source, adapter)
    if destination.exists():
        if progress:
            progress(f"Using cached vision encoder: {destination}")
        return destination
    metadata = _safetensors_metadata(source, token)
    selected = [key for key in metadata.weight_map if adapter.select(key)]
    entries = []
    offset = 0
    for filename in sorted({metadata.weight_map[key] for key in selected}):
        file_metadata = metadata.files_metadata[filename]
        file_keys = sorted(
            (key for key in selected if metadata.weight_map[key] == filename),
            key=lambda key: file_metadata.tensors[key].data_offsets[0],
        )
        for key in file_keys:
            tensor = file_metadata.tensors[key]
            size = tensor.data_offsets[1] - tensor.data_offsets[0]
            entries.append(
                (
                    filename,
                    tensor.data_offsets[0],
                    tensor.data_offsets[1],
                    adapter.map_key(key),
                    tensor.dtype,
                    list(tensor.shape),
                    offset,
                    offset + size,
                )
            )
            offset += size
    header = {
        "__metadata__": {
            "source": source.identifier,
            "revision": source.resolved_revision,
        }
    }
    for _, _, _, key, dtype, shape, start, end in entries:
        if key in header:
            raise RuntimeError(f"Duplicate mapped tensor key: {key}")
        header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [start, end]}
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    header_bytes += b" " * (-len(header_bytes) % 8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(struct.pack("<Q", len(header_bytes)))
            temporary.write(header_bytes)
            for filename in sorted({entry[0] for entry in entries}):
                file_entries = [entry for entry in entries if entry[0] == filename]
                header_size = (
                    _read_header_size_local(source.path / filename)
                    if source.path
                    else _read_header_size_remote(source, filename, token)
                )
                for group in _merge_intervals(file_entries):
                    start = 8 + header_size + group[0][1]
                    end = 8 + header_size + group[-1][2]
                    if progress:
                        progress(f"Fetching {filename} bytes {start:,}-{end - 1:,}")
                    if source.path:
                        _copy_local_range(source.path / filename, start, end, temporary)
                    else:
                        output_offset = temporary.tell()
                        try:
                            _copy_remote_range(source, filename, start, end, temporary, token)
                        except RuntimeError as error:
                            temporary.seek(output_offset)
                            temporary.truncate()
                            if not allow_full_download:
                                raise RuntimeError(
                                    f"Selective download failed for {filename}. Re-run with "
                                    "--allow-full-download to download the containing checkpoint shard."
                                ) from error
                            if progress:
                                progress(f"Downloading full fallback shard {filename}")
                            full_path = Path(
                                hf_hub_download(
                                    source.identifier,
                                    filename,
                                    revision=source.resolved_revision,
                                    token=token,
                                )
                            )
                            _copy_local_range(full_path, start, end, temporary)
        temporary_path.replace(destination)
    finally:
        if temporary_path:
            temporary_path.unlink(missing_ok=True)
    return destination


def select_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_dtype(value: str, device: torch.device) -> torch.dtype:
    if value == "float32":
        return torch.float32
    if value == "bfloat16":
        return torch.bfloat16
    return torch.float32 if device.type == "cpu" else torch.bfloat16


def tensor_bytes(tensor: torch.Tensor) -> tuple[bytes, str]:
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.uint16).numpy().tobytes(), "bfloat16"
    if tensor.dtype == torch.float16:
        return tensor.view(torch.uint16).numpy().tobytes(), "float16"
    if tensor.dtype == torch.float32:
        return tensor.numpy().tobytes(), "float32"
    raise TypeError(f"Unsupported output dtype: {tensor.dtype}")


class EdgeEncoder:
    def __init__(
        self,
        model: str,
        revision: str | None = None,
        cache_dir: Path | None = None,
        device: str = "auto",
        dtype: str = "auto",
        token=None,
        progress: Callable[[str], None] | None = None,
        allow_full_download: bool = False,
        optimize: bool = True,
        split_point: str = "vision_post_projector",
        require_optimized: bool = False,
    ):
        if progress:
            progress(f"Resolving model: {model}")
        self.source = resolve_source(model, revision, token)
        if progress:
            progress(
                f"Model found: {self.source.identifier}@{self.source.resolved_revision}"
            )
        self.config = AutoConfig.from_pretrained(
            self.source.identifier,
            revision=self.source.resolved_revision if not self.source.path else None,
            token=token,
        )
        if progress:
            progress("Finding vision encoder")
        self.adapter = resolve_adapter(self.config)
        self.plan = plan_weights(self.source, self.adapter, token)
        if progress:
            progress(
                f"Vision encoder found: {self.config.model_type} "
                f"({self.plan.selected_bytes / 1_000_000:.1f} MB)"
            )
        weights = materialize_weights(
            self.source,
            self.adapter,
            cache_dir,
            token,
            progress,
            allow_full_download,
        )
        self._weights = weights
        self._progress = progress
        self._device_request = device
        self._dtype_request = dtype
        self._token = token
        self._split_point = split_point
        self._require_optimized = require_optimized
        self._mlx = None
        try:
            from .mlx_edge_encoder import MLXGemma4E4BEncoder, qualification_reason

            reason = (
                qualification_reason(self.source, self.config, device, dtype)
                if optimize
                else "optimized execution disabled for benchmark control"
            )
            if reason is None:
                if progress:
                    progress("Qualified optimization profile found: Gemma 4 E4B on MLX")
                    progress("Loading optimized MLX vision encoder")
                self._mlx = MLXGemma4E4BEncoder(self.config, weights, split_point)
                self.backend = "mlx"
                self.device = "mlx:gpu"
                if progress:
                    progress("Using optimized MLX backend")
                return
            if progress and self.source.identifier.lower() == "google/gemma-4-e4b-it":
                progress(f"MLX optimization unavailable: {reason}; using PyTorch")
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            if require_optimized:
                raise RuntimeError(f"Required optimized backend failed: {error}") from error
            if progress:
                progress(f"MLX initialization failed: {error}; using PyTorch")
        if require_optimized:
            raise UnsupportedModelError(
                f"Chat requires the qualified optimized backend: {reason}"
            )
        if split_point != "vision_post_projector":
            raise UnsupportedModelError(
                "The compatibility backend does not support this split point"
            )
        self._initialize_pytorch()

    def _initialize_pytorch(self) -> None:
        self.processor = AutoImageProcessor.from_pretrained(
            self.source.identifier,
            revision=self.source.resolved_revision if not self.source.path else None,
            token=self._token,
        )
        self.device = select_device(self._device_request)
        if self._progress:
            self._progress(f"Using PyTorch backend on: {self.device}")
        self.backend = "pytorch"
        self._auto_device = self._device_request == "auto"
        self.dtype = select_dtype(self._dtype_request, self.device)
        self.module = self.adapter.build(self.config)
        self.module.load_state_dict(load_file(self._weights), strict=True)
        self.module.to(device=self.device, dtype=self.dtype).eval()

    def encode(self, image: Image.Image) -> tuple[torch.Tensor, dict]:
        if self._mlx is not None:
            try:
                return self._mlx.encode(image)
            except RuntimeError as error:
                if self._require_optimized:
                    self._mlx.close()
                    self._mlx = None
                    raise RuntimeError(f"Required optimized backend failed: {error}") from error
                if self._progress:
                    self._progress(f"MLX encoding failed: {error}; retrying with PyTorch")
                self._mlx.close()
                self._mlx = None
                self._initialize_pytorch()
        started = time.perf_counter()
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        processed = time.perf_counter()
        try:
            features = self._encode_inputs(inputs)
        except RuntimeError:
            if not self._auto_device or self.device.type == "cpu":
                raise
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.module.to(device=self.device, dtype=self.dtype)
            features = self._encode_inputs(inputs)
        encoded = time.perf_counter()
        if not torch.isfinite(features.float()).all():
            raise RuntimeError("Encoder output contains non-finite values")
        return features, {
            "preprocess_ms": (processed - started) * 1000,
            "encode_ms": (encoded - processed) * 1000,
        }

    def _encode_inputs(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        device_inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }
        if self.device.type == "mps":
            torch.mps.synchronize()
        with torch.inference_mode():
            features = self.adapter.encode(self.module, device_inputs)
        if self.device.type == "mps":
            torch.mps.synchronize()
        return features

    def metadata(self, tensor: torch.Tensor, dtype_name: str, metrics: dict) -> dict:
        runtime = (
            self._mlx.metadata()
            if self._mlx is not None
            else {"backend": "pytorch", "device": str(self.device)}
        )
        return {
            "model": self.source.identifier,
            "revision": self.source.resolved_revision,
            "model_type": self.config.model_type,
            "architecture": self.adapter.architecture,
            **runtime,
            "tensor": {
                "shape": list(tensor.shape),
                "dtype": dtype_name,
                "byte_order": "little",
                "byte_length": tensor.numel() * tensor.element_size(),
            },
            "weights": {
                "selected_bytes": self.plan.selected_bytes,
                "checkpoint_bytes": self.plan.checkpoint_bytes,
                "files": list(self.plan.files),
            },
            "metrics": metrics,
        }
