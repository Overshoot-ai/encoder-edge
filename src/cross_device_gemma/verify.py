from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor, AutoTokenizer

from . import MODEL_ID
from .modeling import build_inputs_embeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare normal and split Gemma inference")
    parser.add_argument("--client-artifact", type=Path, required=True)
    parser.add_argument("--server-artifact", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", default="What is shown in this image?")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedVisionEmbedder

    config = AutoConfig.from_pretrained(args.client_artifact)
    processor = AutoProcessor.from_pretrained(args.client_artifact)
    tokenizer = AutoTokenizer.from_pretrained(args.server_artifact)

    embedder = Gemma4UnifiedVisionEmbedder(config.vision_config, config.text_config)
    embedder.load_state_dict(load_file(args.client_artifact / "vision.safetensors"))
    embedder.to(device="cuda", dtype=torch.bfloat16).eval()

    image = Image.open(args.image).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.question},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    )

    with torch.inference_mode():
        image_positions = inputs["image_position_ids"].to("cuda")
        padded_features = embedder(inputs["pixel_values"].to("cuda"), image_positions)
        image_features = padded_features[~image_positions.eq(-1).all(dim=-1)]

    tensors = {
        "image_features": image_features,
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "mm_token_type_ids": inputs["mm_token_type_ids"],
    }
    full_model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
    ).to("cuda").eval()
    server_model = AutoModelForMultimodalLM.from_pretrained(
        args.server_artifact,
        dtype=torch.bfloat16,
    ).to("cuda").eval()
    if server_model.model.embed_vision is not None:
        raise RuntimeError("Server artifact unexpectedly loaded an image embedder")

    normal_inputs = {
        "input_ids": inputs["input_ids"].to("cuda"),
        "pixel_values": inputs["pixel_values"].to("cuda"),
        "image_position_ids": inputs["image_position_ids"].to("cuda"),
        "attention_mask": inputs["attention_mask"].to("cuda"),
        "mm_token_type_ids": inputs["mm_token_type_ids"].to("cuda"),
    }
    input_ids, attention_mask, token_types, inputs_embeds = build_inputs_embeds(server_model, tensors)

    with torch.inference_mode():
        normal_logits = full_model(**normal_inputs, use_cache=False, logits_to_keep=1).logits.float()
        split_logits = server_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mm_token_type_ids=token_types,
            use_cache=False,
            logits_to_keep=1,
        ).logits.float()
        max_logit_difference = (normal_logits - split_logits).abs().max().item()

        normal_ids = full_model.generate(
            **normal_inputs,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
        )
        split_ids = server_model.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mm_token_type_ids=token_types,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
        )

    prompt_length = input_ids.shape[1]
    normal_generated = normal_ids[0, prompt_length:]
    split_generated = split_ids[0, prompt_length:]
    tokens_equal = torch.equal(normal_generated, split_generated)

    print(f"visual_tokens={image_features.shape[0]}")
    print(f"max_logit_difference={max_logit_difference:.8f}")
    print(f"generated_tokens_equal={tokens_equal}")
    print(f"normal_answer={tokenizer.decode(normal_generated, skip_special_tokens=True).strip()}")
    print(f"split_answer={tokenizer.decode(split_generated, skip_special_tokens=True).strip()}")
    if not tokens_equal:
        raise SystemExit("Split generation differs from normal generation")
