from collections.abc import Mapping

import torch

def build_inputs_embeds(
    model,
    tensors: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    embedding_layer = model.model.language_model.get_input_embeddings()
    device = embedding_layer.weight.device

    input_ids = tensors["input_ids"].to(device)
    attention_mask = tensors["attention_mask"].to(device)
    token_types = tensors["mm_token_type_ids"].to(device)
    image_features = tensors["image_features"].to(device=device, dtype=embedding_layer.weight.dtype)

    image_mask = input_ids.eq(model.config.image_token_id)
    multimodal_ids = {
        model.config.image_token_id,
        model.config.video_token_id,
        model.config.audio_token_id,
    }
    multimodal_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in multimodal_ids:
        if token_id is not None:
            multimodal_mask |= input_ids.eq(token_id)

    safe_input_ids = input_ids.masked_fill(multimodal_mask, model.config.text_config.pad_token_id)
    inputs_embeds = embedding_layer(safe_input_ids)
    expanded_image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(expanded_image_mask, image_features)
    return input_ids, attention_mask, token_types, inputs_embeds
