from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

import torch
from torch import nn
from transformers import BatchFeature
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM
from vllm.model_executor.models.gemma4_mm import (
    Gemma4DummyInputsBuilder,
    Gemma4ForConditionalGeneration,
    Gemma4ImageInputs,
    Gemma4MultimodalEmbedder,
    Gemma4MultiModalProcessor,
    Gemma4ProcessingInfo,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.model_executor.models.gemma4_unified import (
    Gemma4UnifiedForConditionalGeneration,
    Gemma4UnifiedProcessingInfo,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import ImageEmbeddingItems, MultiModalDataItems
from vllm.multimodal.processing.processor import (
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.utils.tensor_schema import TensorSchema, TensorShape

logger = init_logger(__name__)


class Gemma4ImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"] = "image_embeds"
    data: Annotated[torch.Tensor, TensorShape("bn", "ifs", "hs")]


class CrossDeviceGemmaProcessor(Gemma4MultiModalProcessor):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if not mm_data:
            prompt_ids = self._apply_hf_processor_text_only(prompt, tok_kwargs)
            return BatchFeature({"input_ids": [prompt_ids]})
        return super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        fields = dict(super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs))
        fields["image_embeds"] = MultiModalFieldConfig.batched("image")
        return fields

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        images = mm_items.get("image")
        if not isinstance(images, ImageEmbeddingItems):
            return super()._get_prompt_updates(
                mm_items, hf_processor_mm_kwargs, out_mm_kwargs
            )

        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        config = self.info.get_hf_config()

        def replacement(item_idx: int):
            token_ids = (
                [config.boi_token_id]
                + [processor.image_token_id] * images.get_feature_size(item_idx)
                + [config.eoi_token_id]
            )
            return PromptUpdateDetails.select_token_id(
                token_ids, processor.image_token_id
            )

        return [
            PromptReplacement(
                modality="image",
                target=processor.image_token,
                replacement=replacement,
            )
        ]


class CrossDeviceGemma4Processor(CrossDeviceGemmaProcessor):
    pass


@MULTIMODAL_REGISTRY.register_processor(
    CrossDeviceGemma4Processor,
    info=Gemma4ProcessingInfo,
    dummy_inputs=Gemma4DummyInputsBuilder,
)
class CrossDeviceGemma4ForConditionalGeneration(Gemma4ForConditionalGeneration):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image_embeds":
            modality = "image"
        return super().get_placeholder_str(modality, i)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        self.model_dtype = vllm_config.model_config.dtype
        self.vision_tower = None
        self.audio_tower = None
        self.embed_audio = None

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.embed_vision = Gemma4MultimodalEmbedder(
                config.vision_config,
                config.text_config,
                quant_config=None,
                prefix=maybe_prefix(prefix, "embed_vision"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model: Gemma4ForCausalLM = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Gemma4ForCausalLM"],
            )
            ple_dim = config.text_config.hidden_size_per_layer_input
            if ple_dim is not None and ple_dim > 0:
                embed = self.language_model.model.embed_tokens
                self.per_layer_embeddings = torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.text_config.num_hidden_layers,
                    ple_dim,
                    device=next(embed.parameters()).device,
                    dtype=vllm_config.model_config.dtype,
                )
            else:
                self.per_layer_embeddings = None

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        layer_types = getattr(config.text_config, "layer_types", None)
        if getattr(config.text_config, "use_bidirectional_attention", None) == "vision":
            self._full_attn_layer_idxs = frozenset(
                index
                for index, layer_type in enumerate(layer_types or [])
                if layer_type != "sliding_attention"
            )
        else:
            self._full_attn_layer_idxs = frozenset()

        for name in (
            "moe_layers",
            "num_moe_layers",
            "num_logical_experts",
            "num_physical_experts",
            "num_local_physical_experts",
            "num_routed_experts",
            "num_expert_groups",
            "num_shared_experts",
            "num_redundant_experts",
        ):
            setattr(self, name, getattr(self.language_model, name))

        generation_config = vllm_config.model_config.try_get_generation_config()
        self._suppress_token_ids = (
            generation_config.get("suppress_tokens") if generation_config else None
        )
        logger.info(
            "Cross-device Gemma 4 initialized without multimodal towers and "
            "with the server-side vision projector"
        )

    def _parse_and_validate_image_input(self, **kwargs: object):
        image_embeds = kwargs.pop("image_embeds", None)
        if image_embeds is not None:
            return Gemma4ImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )
        return super()._parse_and_validate_image_input(**kwargs)

    def _process_image_input(
        self,
        image_input: Gemma4ImageInputs | Gemma4ImageEmbeddingInputs,
    ) -> list[torch.Tensor]:
        if image_input["type"] == "image_embeds":
            embeddings = list(image_input["data"])
            vision_size = self.config.vision_config.hidden_size
            text_size = self.config.text_config.hidden_size
            projected: list[torch.Tensor] = []
            for embedding in embeddings:
                if embedding.ndim != 2:
                    raise ValueError(
                        "Each image embedding must be a 2D token matrix, got "
                        f"shape {tuple(embedding.shape)}"
                    )
                if embedding.shape[-1] == text_size:
                    projected.append(embedding)
                elif embedding.shape[-1] == vision_size:
                    projected.append(
                        self.embed_vision(embedding.unsqueeze(0)).squeeze(0)
                    )
                else:
                    raise ValueError(
                        "Image embedding width must be either the pre-projector "
                        f"width {vision_size} or text width {text_size}, got "
                        f"{embedding.shape[-1]}"
                    )
            return projected
        return super()._process_image_input(image_input)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector=["embed_vision"],
            tower_model=[],
        )


@MULTIMODAL_REGISTRY.register_processor(
    CrossDeviceGemmaProcessor,
    info=Gemma4UnifiedProcessingInfo,
    dummy_inputs=Gemma4DummyInputsBuilder,
)
class CrossDeviceGemma4UnifiedForConditionalGeneration(
    Gemma4UnifiedForConditionalGeneration
):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image_embeds":
            modality = "image"
        return super().get_placeholder_str(modality, i)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        vision_config = config.vision_config
        config.vision_config = None
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            config.vision_config = vision_config
        if self.vision_embedder is not None or self.embed_vision is not None:
            raise RuntimeError(
                "Optimized server unexpectedly constructed vision modules"
            )
        logger.info("Cross-device Gemma initialized without vision modules")

    def _parse_and_validate_image_input(self, **kwargs: object):
        image_embeds = kwargs.pop("image_embeds", None)
        if image_embeds is not None:
            return Gemma4ImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )
        return super()._parse_and_validate_image_input(**kwargs)

    def _process_image_input(
        self,
        image_input: Gemma4ImageInputs | Gemma4ImageEmbeddingInputs,
    ) -> list[torch.Tensor]:
        if image_input["type"] == "image_embeds":
            return list(image_input["data"])
        return super()._process_image_input(image_input)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector=[],
            tower_model=[],
        )
