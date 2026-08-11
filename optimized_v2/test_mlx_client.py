import queue
import threading
import time
from unittest import mock

import pytest

pytest.importorskip("mlx_vlm")

import mlx.core as mx
from PIL import Image

from .mlx_client import MLXBinaryStreamingImageClient


def test_complete_many_is_bounded_and_ordered():
    client = object.__new__(MLXBinaryStreamingImageClient)
    active = 0
    maximum_active = 0
    lock = threading.Lock()

    def prepare(image, question, max_tokens, max_soft_tokens):
        return question.encode(), {"total_started": time.perf_counter()}

    def stream(payload, metrics):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep((4 - int(payload)) * 0.01)
        with lock:
            active -= 1
        yield {"type": "done", "index": int(payload)}

    client._prepare_request = prepare
    client._stream_prepared = stream
    results = client.complete_many(
        [(None, str(index)) for index in range(4)], max_in_flight=2
    )
    assert maximum_active == 2
    assert [events[-1]["index"] for events in results] == [0, 1, 2, 3]


def test_complete_many_rejects_invalid_limit():
    client = object.__new__(MLXBinaryStreamingImageClient)
    with pytest.raises(ValueError, match="max_in_flight"):
        client.complete_many([], max_in_flight=0)


def test_failed_connection_releases_slot_for_waiter():
    client = object.__new__(MLXBinaryStreamingImageClient)
    client.host = "localhost"
    client.port = 8002
    client._connection_pool = queue.LifoQueue()
    client._connection_lock = threading.Lock()
    client._connection_changed = threading.Condition(client._connection_lock)
    client._connection_count = 0
    client._max_connections = 1
    first = client._acquire_connection()
    acquired = []

    waiter = threading.Thread(target=lambda: acquired.append(client._acquire_connection()))
    waiter.start()
    time.sleep(0.01)
    assert waiter.is_alive()
    client._release_connection(first, reusable=False)
    waiter.join(timeout=1)

    assert not waiter.is_alive()
    assert len(acquired) == 1
    client._release_connection(acquired[0], reusable=False)


def test_encode_image_restores_soft_token_budget():
    client = object.__new__(MLXBinaryStreamingImageClient)
    client._encode_lock = threading.Lock()
    client.processor = mock.Mock()
    client.processor.image_processor.max_soft_tokens = 280
    client.config = mock.Mock()
    client.encode_vision = lambda pixels: mx.zeros((1, 1, 768), dtype=mx.bfloat16)
    image = Image.new("RGB", (16, 16))

    with mock.patch(
        "optimized_v2.mlx_client.apply_chat_template", return_value="prompt"
    ), mock.patch(
        "optimized_v2.mlx_client.prepare_inputs",
        return_value={"pixel_values": mx.zeros((1, 3, 16, 16))},
    ):
        client.encode_image(image, "question", max_soft_tokens=140)

    assert client.processor.image_processor.max_soft_tokens == 280
