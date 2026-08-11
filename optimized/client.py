import argparse
import base64
import io
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoConfig, AutoImageProcessor
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4MultimodalEmbedder,
    Gemma4VisionModel,
)
from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedVisionEmbedder,
)


class StreamingImageClient:
    def __init__(self, artifact: Path, server: str, model: str):
        self.server = server.rstrip("/")
        self.model = model
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        dtype = torch.bfloat16 if self.device.type == "mps" else torch.float32
        config = AutoConfig.from_pretrained(artifact)
        self.image_processor = AutoImageProcessor.from_pretrained(artifact)
        vision_state = load_file(artifact / "vision.safetensors")
        self.is_gemma4 = config.model_type == "gemma4"
        if self.is_gemma4:
            self.vision_tower = Gemma4VisionModel(config.vision_config)
            self.vision_tower.load_state_dict(
                {
                    name.removeprefix("vision_tower."): tensor
                    for name, tensor in vision_state.items()
                    if name.startswith("vision_tower.")
                }
            )
            self.embedder = Gemma4MultimodalEmbedder(
                config.vision_config,
                config.text_config,
            )
            self.embedder.load_state_dict(
                {
                    name.removeprefix("embed_vision."): tensor
                    for name, tensor in vision_state.items()
                    if name.startswith("embed_vision.")
                }
            )
            self.vision_tower.to(device=self.device, dtype=dtype).eval()
            self.embedder.to(device=self.device, dtype=dtype).eval()
        else:
            self.embedder = Gemma4UnifiedVisionEmbedder(
                config.vision_config, config.text_config
            )
            self.embedder.load_state_dict(vision_state)
            self.embedder.to(device=self.device, dtype=dtype).eval()

    def encode_image(self, image: Image.Image) -> tuple[torch.Tensor, float, float]:
        started = time.perf_counter()
        inputs = self.image_processor(
            images=image.convert("RGB"),
            return_tensors="pt",
        )
        preprocess_ms = (time.perf_counter() - started) * 1000
        positions = inputs["image_position_ids"].to(self.device)
        if self.device.type == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            if self.is_gemma4:
                hidden_states = self.vision_tower(
                    pixel_values=inputs["pixel_values"].to(
                        device=self.device,
                        dtype=self.embedder.embedding_projection.weight.dtype,
                    ),
                    pixel_position_ids=positions,
                ).last_hidden_state
                features = self.embedder(inputs_embeds=hidden_states)
            else:
                features = self.embedder(
                    inputs["pixel_values"].to(self.device), positions
                )
        if self.device.type == "mps":
            torch.mps.synchronize()
        encode_ms = (time.perf_counter() - started) * 1000
        if not self.is_gemma4:
            features = features[~positions.eq(-1).all(dim=-1)]
        return features, preprocess_ms, encode_ms

    def stream(self, image: Image.Image, question: str):
        total_started = time.perf_counter()
        features, preprocess_ms, encode_ms = self.encode_image(image)

        started = time.perf_counter()
        buffer = io.BytesIO()
        torch.save(features.cpu().contiguous(), buffer)
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_embeds",
                                "image_embeds": base64.b64encode(
                                    buffer.getvalue()
                                ).decode(),
                            },
                            {"type": "text", "text": question},
                        ],
                    }
                ],
                "max_tokens": 128,
                "temperature": 0,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        ).encode()
        serialize_ms = (time.perf_counter() - started) * 1000
        request = Request(
            f"{self.server}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        remote_started = time.perf_counter()
        first_token_at = None
        usage = None
        with urlopen(request, timeout=300) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    break
                event = json.loads(data)
                usage = event.get("usage") or usage
                choices = event.get("choices", [])
                text = choices[0].get("delta", {}).get("content") if choices else None
                if text:
                    first_token_at = first_token_at or time.perf_counter()
                    yield {"type": "token", "text": text}

        finished = time.perf_counter()
        yield {
            "type": "done",
            "client_preprocess_ms": preprocess_ms,
            "client_encode_ms": encode_ms,
            "client_serialize_ms": serialize_ms,
            "request_bytes": len(payload),
            "visual_tokens": features.shape[0],
            "pipeline_ttft_ms": (first_token_at - total_started) * 1000,
            "remote_ttft_ms": (first_token_at - remote_started) * 1000,
            "pipeline_e2e_ms": (finished - total_started) * 1000,
            "remote_e2e_ms": (finished - remote_started) * 1000,
            "usage": usage,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", default="gemma-4-12b-optimized")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", required=True)
    args = parser.parse_args()
    client = StreamingImageClient(args.artifact, args.server, args.model)
    for event in client.stream(Image.open(args.image), args.question):
        print(event.get("text", ""), end="", flush=True)
        if event["type"] == "done":
            print("\n" + json.dumps(event, indent=2))


if __name__ == "__main__":
    main()
