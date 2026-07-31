from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

import torch
from transformers import BatchFeature
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.gemma4_mm import (
    Gemma4DummyInputsBuilder,
    Gemma4ImageInputs,
    Gemma4MultiModalProcessor,
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
