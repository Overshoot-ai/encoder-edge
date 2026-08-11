import argparse
import concurrent.futures
import gc
import http.client
import json
import queue
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import mlx.core as mx
import numpy as np
from PIL import Image
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

from .protocol import CONTENT_TYPE, encode_raw_request
from .mlx_vision_optimizations import (
    fuse_gemma4_qkv_epilogue,
    fuse_gemma4_rope_layout,
    make_segmented_gemma4_encoder,
    optimize_gemma4_positions,
)


class MLXBinaryStreamingImageClient:
    def __init__(
        self,
        checkpoint: str,
        server: str,
        model: str,
        project_on_server: bool = False,
        max_connections: int = 8,
        use_qkv_epilogue: bool = True,
        compression: str | None = None,
    ):
        if max_connections < 1:
            raise ValueError("max_connections must be positive")
        parsed = urlsplit(server)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("Server must be an HTTP URL")
        self.path = parsed.path.rstrip("/") + "/v1/chat/completions"
        self.host = parsed.hostname
        self.port = parsed.port or 80
        self.model_name = model
        self.compression = compression
        self._encode_lock = threading.Lock()
        self._connection_pool = queue.LifoQueue()
        self._connection_lock = threading.Lock()
        self._connection_changed = threading.Condition(self._connection_lock)
        self._connection_count = 0
        self._max_connections = max_connections

        started = time.perf_counter()
        mx.set_wired_limit(2 * 1024**3)
        full_model, self.processor = load(checkpoint)
        mx.synchronize()
        self.config = full_model.config
        self.vision_tower = full_model.vision_tower
        self.embed_vision = None if project_on_server else full_model.embed_vision
        optimize_gemma4_positions(self.vision_tower)
        if use_qkv_epilogue:
            fuse_gemma4_qkv_epilogue(self.vision_tower)
        else:
            fuse_gemma4_rope_layout(self.vision_tower)
        self.encode_vision = make_segmented_gemma4_encoder(
            self.vision_tower,
            self.embed_vision,
            segment_size=3,
            evaluate_segments=True,
        )
        del full_model
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
        self.load_ms = (time.perf_counter() - started) * 1000
        self.active_memory_gb = mx.get_active_memory() / 1e9

    def encode_image(
        self,
        image: Image.Image,
        question: str,
        max_soft_tokens: int | None = None,
    ):
        with self._encode_lock:
            image_processor = self.processor.image_processor
            previous_soft_tokens = image_processor.max_soft_tokens
            try:
                if max_soft_tokens is not None:
                    image_processor.max_soft_tokens = max_soft_tokens
                prompt = apply_chat_template(
                    self.processor,
                    self.config,
                    question,
                    num_images=1,
                )
                started = time.perf_counter()
                inputs = prepare_inputs(
                    self.processor,
                    images=[image.convert("RGB")],
                    prompts=prompt,
                    add_special_tokens=False,
                )
                processed = time.perf_counter()
                features = self.encode_vision(inputs["pixel_values"])
                mx.eval(features)
                mx.synchronize()
                encoded = time.perf_counter()
            finally:
                image_processor.max_soft_tokens = previous_soft_tokens
        if features.ndim != 3 or features.shape[0] != 1:
            raise RuntimeError(f"Unexpected MLX vision output shape: {features.shape}")
        features = features[0]
        if features.dtype != mx.bfloat16:
            raise RuntimeError(f"Expected BF16 MLX vision output, got {features.dtype}")
        return (
            features,
            (processed - started) * 1000,
            (encoded - processed) * 1000,
        )

    def _acquire_connection(self) -> http.client.HTTPConnection:
        with self._connection_changed:
            while True:
                try:
                    return self._connection_pool.get_nowait()
                except queue.Empty:
                    if self._connection_count < self._max_connections:
                        self._connection_count += 1
                        return http.client.HTTPConnection(
                            self.host, self.port, timeout=300
                        )
                    self._connection_changed.wait()

    def _release_connection(
        self, connection: http.client.HTTPConnection, reusable: bool
    ) -> None:
        with self._connection_changed:
            if reusable:
                self._connection_pool.put(connection)
            else:
                connection.close()
                self._connection_count -= 1
            self._connection_changed.notify()

    def _prepare_request(
        self,
        image: Image.Image,
        question: str,
        max_tokens: int,
        max_soft_tokens: int | None,
    ) -> tuple[bytes, dict]:
        total_started = time.perf_counter()
        features, preprocess_ms, encode_ms = self.encode_image(
            image,
            question,
            max_soft_tokens=max_soft_tokens,
        )
        started = time.perf_counter()
        tensor_bytes = np.array(features.view(mx.uint16), copy=True).tobytes(order="C")
        payload = encode_raw_request(
            tensor_bytes,
            tuple(features.shape),
            question,
            self.model_name,
            max_tokens,
            self.compression,
        )
        return payload, {
            "total_started": total_started,
            "client_preprocess_ms": preprocess_ms,
            "client_encode_ms": encode_ms,
            "client_serialize_ms": (time.perf_counter() - started) * 1000,
            "tensor_bytes": len(tensor_bytes),
            "visual_tokens": features.shape[0],
            "hidden_size": features.shape[1],
        }

    def _stream_prepared(self, payload: bytes, metrics: dict):
        remote_started = time.perf_counter()
        connection = self._acquire_connection()
        reusable = False
        response = None
        try:
            connection.request(
                "POST",
                self.path,
                body=payload,
                headers={
                    "Content-Type": CONTENT_TYPE,
                    "Accept": "text/event-stream",
                },
            )
            response = connection.getresponse()
            if response.status != 200:
                error = response.read().decode(errors="replace")
                raise RuntimeError(f"Gateway returned HTTP {response.status}: {error}")
            gateway_ttft_header = response.getheader("X-Gateway-TTFT-Ms")
            gateway_prepare_header = response.getheader("X-Gateway-Prepare-Ms")
            vllm_ttft_header = response.getheader("X-vLLM-TTFT-Ms")
            gateway_ttft_ms = (
                float(gateway_ttft_header) if gateway_ttft_header is not None else None
            )
            gateway_prepare_ms = (
                float(gateway_prepare_header)
                if gateway_prepare_header is not None
                else None
            )
            vllm_ttft_ms = (
                float(vllm_ttft_header) if vllm_ttft_header is not None else None
            )

            first_token_at = None
            usage = None
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
            response.read()
            reusable = True
            finished = time.perf_counter()
            first_token_at = first_token_at or finished
            yield {
                "type": "done",
                **{key: value for key, value in metrics.items() if key != "total_started"},
                "request_bytes": len(payload),
                "pipeline_ttft_ms": (first_token_at - metrics["total_started"])
                * 1000,
                "remote_ttft_ms": (first_token_at - remote_started) * 1000,
                "gateway_ttft_ms": gateway_ttft_ms,
                "gateway_prepare_ms": gateway_prepare_ms,
                "vllm_ttft_ms": vllm_ttft_ms,
                "transport_ttft_ms": (
                    (first_token_at - remote_started) * 1000 - gateway_ttft_ms
                    if gateway_ttft_ms is not None
                    else None
                ),
                "pipeline_e2e_ms": (finished - metrics["total_started"]) * 1000,
                "remote_e2e_ms": (finished - remote_started) * 1000,
                "usage": usage,
            }
        finally:
            if response is not None and not reusable:
                response.close()
            self._release_connection(connection, reusable)

    def stream(
        self,
        image: Image.Image,
        question: str,
        max_tokens: int = 128,
        max_soft_tokens: int | None = None,
    ):
        payload, metrics = self._prepare_request(
            image, question, max_tokens, max_soft_tokens
        )
        yield from self._stream_prepared(payload, metrics)

    def complete_many(
        self,
        requests,
        max_tokens: int = 128,
        max_soft_tokens: int | None = None,
        max_in_flight: int = 4,
    ) -> list[list[dict]]:
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be positive")
        items = list(requests)
        results: list[list[dict] | None] = [None] * len(items)
        pending = {}
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_in_flight)
        try:
            for index, (image, question) in enumerate(items):
                while len(pending) >= max_in_flight:
                    done, _ = concurrent.futures.wait(
                        pending, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for future in done:
                        results[pending.pop(future)] = future.result()
                payload, metrics = self._prepare_request(
                    image, question, max_tokens, max_soft_tokens
                )
                future = executor.submit(
                    lambda body=payload, values=metrics: list(
                        self._stream_prepared(body, values)
                    )
                )
                pending[future] = index
            for future in concurrent.futures.as_completed(pending):
                results[pending[future]] = future.result()
        except BaseException:
            for future in pending:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        if any(result is None for result in results):
            raise RuntimeError("Pipeline completed without a result for every request")
        return [result for result in results if result is not None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="mlx-community/gemma-4-e4b-it-4bit",
    )
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", default="gemma-4-e4b-optimized")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--project-on-server",
        action="store_true",
        help="Send pooled 768D vision states and project them on the server",
    )
    parser.add_argument(
        "--strict-qkv",
        action="store_true",
        help="Use the slower bit-exact QKV path instead of the accuracy-qualified default",
    )
    parser.add_argument(
        "--zstd",
        action="store_true",
        help="Losslessly compress BF16 visual features for transport",
    )
    args = parser.parse_args()
    client = MLXBinaryStreamingImageClient(
        args.checkpoint,
        args.server,
        args.model,
        project_on_server=args.project_on_server,
        use_qkv_epilogue=not args.strict_qkv,
        compression="zstd" if args.zstd else None,
    )
    for event in client.stream(
        Image.open(args.image),
        args.question,
        args.max_tokens,
    ):
        print(event.get("text", ""), end="", flush=True)
        if event["type"] == "done":
            print("\n" + json.dumps(event, indent=2))


if __name__ == "__main__":
    main()
