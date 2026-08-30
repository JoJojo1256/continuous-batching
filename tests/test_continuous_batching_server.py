from __future__ import annotations

import json

import httpx
import pytest

from bench.loadgen import output_lengths
from bench.compare import comparison_rows
from continuous_batching.config import InferenceConfig
from continuous_batching.engine import ModelEngine
from continuous_batching.server import create_app
from tests.continuous_batching_fakes import FakeTokenizer, IncrementModel


def make_app(*, queue_capacity: int = 4):
    config = InferenceConfig(
        model_name="fake",
        dtype="float32",
        max_batch_size=2,
        max_seq_len=32,
        default_max_new_tokens=3,
        device="cpu",
        mode="continuous",
        static_batch_size=2,
    )
    engine = ModelEngine(config, model=IncrementModel(), tokenizer=FakeTokenizer())
    return create_app(config, engine=engine, queue_capacity=queue_capacity)


@pytest.mark.asyncio
async def test_sync_streaming_and_metrics_endpoints() -> None:
    app = make_app()
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            sync = await client.post(
                "/generate_sync", json={"prompt": "a", "max_new_tokens": 2}
            )
            stream = await client.post(
                "/generate", json={"prompt": "b", "max_new_tokens": 2}
            )
            metrics = await client.get("/metrics")

    assert sync.status_code == 200
    assert sync.json()["output_token_count"] == 2
    assert stream.status_code == 200
    assert stream.text.count("data: ") == 2
    assert metrics.json()["request_count"] == 2


def test_seeded_workload_shapes_are_reproducible() -> None:
    first = output_lengths(100, "bimodal", seed=4, minimum=8, maximum=64)
    second = output_lengths(100, "bimodal", seed=4, minimum=8, maximum=64)

    assert first == second
    assert all(8 <= value <= 64 for value in first)
    assert any(value <= 22 for value in first)
    assert any(value >= 50 for value in first)


def test_comparison_rows_extract_mode_and_workload(tmp_path) -> None:
    path = tmp_path / "continuous.jsonl"
    records = [
        {
            "type": "loadgen_config",
            "concurrency": 4,
            "length_workload": "bimodal",
        },
        {
            "type": "server_metrics",
            "provenance": {"config": {"mode": "continuous"}},
        },
        {
            "type": "client_request",
            "request_id": "one",
            "start_time": 1.0,
            "first_token_time": 1.1,
            "finish_time": 1.5,
            "ttft_ms": 100.0,
            "tpot_ms": 40.0,
            "end_to_end_ms": 500.0,
            "output_token_count": 10,
            "error": None,
        },
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    rows = comparison_rows([path])

    assert rows[0]["mode"] == "continuous"
    assert rows[0]["concurrency"] == 4
    assert rows[0]["workload"] == "bimodal"
    assert rows[0]["throughput_tokens_per_second"] == 20.0
