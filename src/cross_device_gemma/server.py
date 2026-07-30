import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch
from safetensors import SafetensorError
from safetensors.torch import load as load_tensors
from transformers import AutoModelForMultimodalLM, AutoTokenizer

from .modeling import build_inputs_embeds

FIELDS = {"image_features", "input_ids", "attention_mask", "mm_token_type_ids"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the language side of split Gemma")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    model = AutoModelForMultimodalLM.from_pretrained(args.artifact, dtype=torch.bfloat16).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.artifact)
    if model.model.embed_vision is not None:
        raise RuntimeError("Server artifact contains an image embedder")

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status: int, body: dict) -> None:
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:
            if self.path != "/generate":
                self.respond(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 16 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                tensors = load_tensors(self.rfile.read(length))
                if set(tensors) != FIELDS:
                    raise ValueError("Unexpected tensors")
                ids, features = tensors["input_ids"], tensors["image_features"]
                if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] > 4096:
                    raise ValueError("Invalid prompt")
                if features.ndim != 2 or features.shape[1] != model.config.text_config.hidden_size:
                    raise ValueError("Invalid features")
                if features.shape[0] != ids.eq(model.config.image_token_id).sum().item():
                    raise ValueError("Feature count does not match prompt")
                if features.shape[0] > 1120 or not torch.isfinite(features).all().item():
                    raise ValueError("Invalid feature values")
                if tensors["attention_mask"].shape != ids.shape or tensors["mm_token_type_ids"].shape != ids.shape:
                    raise ValueError("Invalid metadata")

                input_ids, attention_mask, token_types, inputs_embeds = build_inputs_embeds(model, tensors)
                with torch.inference_mode():
                    output = model.generate(
                        input_ids=input_ids,
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        mm_token_type_ids=token_types,
                        do_sample=False,
                        max_new_tokens=128,
                    )
                answer = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
                self.respond(200, {"answer": answer})
            except (ValueError, RuntimeError, SafetensorError) as error:
                self.respond(400, {"error": str(error)})

    HTTPServer((args.host, args.port), Handler).serve_forever()
