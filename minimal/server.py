import argparse
import base64
import io
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer

from .modeling import build_inputs_embeds


class TimedStreamer(BaseStreamer):
    def __init__(self):
        self.prompt = True
        self.first_token_at = None
        self.generated_tokens = 0

    def put(self, value) -> None:
        if self.prompt:
            self.prompt = False
            return
        self.first_token_at = self.first_token_at or time.perf_counter()
        self.generated_tokens += value.numel()

    def end(self) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the language side of split Gemma"
    )
    parser.add_argument("--server-artifact", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    model = (
        AutoModelForMultimodalLM.from_pretrained(
            args.server_artifact, dtype=torch.bfloat16
        )
        .to("cuda")
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.server_artifact)
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
            if self.path != "/v1/chat/completions":
                self.respond(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 16 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                started = time.perf_counter()
                request = json.loads(self.rfile.read(length))
                receive_ms = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                messages, encoded = [], None
                for message in request["messages"]:
                    content = []
                    for item in message["content"]:
                        if item["type"] == "image_embeds":
                            if encoded is not None:
                                raise ValueError("Only one image is supported")
                            encoded = item["image_embeds"]
                            content.append({"type": "image"})
                        else:
                            content.append(item)
                    messages.append({"role": message["role"], "content": content})
                if encoded is None:
                    raise ValueError("Missing image_embeds")
                features = torch.load(
                    io.BytesIO(base64.b64decode(encoded, validate=True)),
                    map_location="cpu",
                    weights_only=True,
                )
                if (
                    features.ndim != 2
                    or features.shape[1] != model.config.text_config.hidden_size
                ):
                    raise ValueError("Invalid features")
                if (
                    features.shape[0] > 1120
                    or not torch.isfinite(features).all().item()
                ):
                    raise ValueError("Invalid feature values")

                ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_dict=True,
                    return_tensors="pt",
                )["input_ids"]
                marker = ids[0].eq(model.config.image_token_id).nonzero().flatten()
                if marker.numel() != 1:
                    raise ValueError("Expected one image marker")
                index = marker.item()
                image_ids = torch.tensor(
                    [model.config.boi_token_id]
                    + [model.config.image_token_id] * features.shape[0]
                    + [model.config.eoi_token_id]
                ).unsqueeze(0)
                ids = torch.cat((ids[:, :index], image_ids, ids[:, index + 1 :]), dim=1)
                token_types = torch.zeros_like(ids)
                token_types[:, index + 1 : index + 1 + features.shape[0]] = 1
                tensors = {
                    "image_features": features,
                    "input_ids": ids,
                    "attention_mask": torch.ones_like(ids),
                    "mm_token_type_ids": token_types,
                }

                input_ids, attention_mask, token_types, inputs_embeds = (
                    build_inputs_embeds(model, tensors)
                )
                prepare_ms = (time.perf_counter() - started) * 1000
                streamer = TimedStreamer()
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    output = model.generate(
                        input_ids=input_ids,
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        mm_token_type_ids=token_types,
                        do_sample=False,
                        max_new_tokens=max(
                            1, min(int(request.get("max_tokens", 128)), 1024)
                        ),
                        streamer=streamer,
                    )
                torch.cuda.synchronize()
                gpu_e2e_ms = (time.perf_counter() - started) * 1000
                gpu_ttft_ms = (streamer.first_token_at - started) * 1000
                decode_ms = gpu_e2e_ms - gpu_ttft_ms
                answer = tokenizer.decode(
                    output[0, input_ids.shape[1] :], skip_special_tokens=True
                ).strip()
                self.respond(
                    200,
                    {
                        "id": "chatcmpl-local",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": request.get("model", "gemma-4-12b"),
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": answer},
                                "finish_reason": "stop",
                            }
                        ],
                        "metrics": {
                            "visual_tokens": features.shape[0],
                            "server_receive_ms": receive_ms,
                            "server_prepare_ms": prepare_ms,
                            "gpu_ttft_ms": gpu_ttft_ms,
                            "gpu_e2e_ms": gpu_e2e_ms,
                            "generated_tokens": streamer.generated_tokens,
                            "decode_tokens_per_second": max(
                                0, streamer.generated_tokens - 1
                            )
                            / (decode_ms / 1000),
                        },
                    },
                )
            except (
                KeyError,
                TypeError,
                ValueError,
                RuntimeError,
                json.JSONDecodeError,
            ) as error:
                self.respond(
                    400,
                    {"error": {"message": str(error), "type": "invalid_request_error"}},
                )

    HTTPServer((args.host, args.port), Handler).serve_forever()
