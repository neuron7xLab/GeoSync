# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Fail-closed behavior tests for unsupported connector protocols."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from mycelium_fractal_net.connectors.config import (
    BackendConfig,
    IngestionConfig,
    RestSourceConfig,
)
from mycelium_fractal_net.connectors.kafka_source import (
    KafkaConnectorUnavailableError,
    KafkaIngestor,
)
from mycelium_fractal_net.connectors.runner import RemoteBackend
from mycelium_fractal_net.connectors.transform import MFNRequest


class _DummyResponse:
    def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError("bad status")

    def json(self) -> dict[str, object]:
        return self._payload


class _DummyClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def post(self, path: str, *, json: dict[str, object]) -> _DummyResponse:  # noqa: A002
        self.calls.append((path, json))
        return _DummyResponse(self.payload)


def test_kafka_ingestor_is_explicitly_unavailable() -> None:
    with pytest.raises(KafkaConnectorUnavailableError, match="Kafka ingestion is disabled"):
        KafkaIngestor(bootstrap_servers="localhost:9092", topic="ticks")


def test_backend_config_rejects_grpc_protocol() -> None:
    with pytest.raises(ValidationError, match="Input should be 'rest'"):
        BackendConfig(protocol="grpc")


def test_ingestion_config_rejects_kafka_source_type() -> None:
    with pytest.raises(ValidationError, match="Input should be 'rest' or 'file'"):
        IngestionConfig(
            source_type="kafka",
        )


def test_ingestion_config_rest_source_still_supported() -> None:
    cfg = IngestionConfig(
        source_type="rest",
        rest_source=RestSourceConfig(url="https://example.com/feed"),
    )
    assert cfg.get_source_config().url == "https://example.com/feed"


def test_remote_backend_rejects_grpc_protocol() -> None:
    with pytest.raises(ValueError, match="protocol 'grpc' is not supported"):
        RemoteBackend(endpoint="https://mfn.internal", protocol="grpc")


def test_remote_backend_rest_extract_features_calls_rest_endpoint() -> None:
    backend = RemoteBackend(endpoint="https://mfn.internal", protocol="rest")
    client = _DummyClient(payload={"features": [1.0, 2.0]})
    async def _fake_get_client() -> _DummyClient:
        return client

    backend._get_client = _fake_get_client  # type: ignore[method-assign]

    req = MFNRequest(
        request_type="feature",
        request_id="req-1",
        timestamp=datetime.now(timezone.utc),
        seeds=[1.0, 2.0],
        grid_size=8,
    )

    result = asyncio.run(backend.extract_features(req))

    assert result.success is True
    assert client.calls[0][0] == "/api/v1/features/extract"


def test_remote_backend_rest_run_simulation_calls_rest_endpoint() -> None:
    backend = RemoteBackend(endpoint="https://mfn.internal", protocol="rest")
    client = _DummyClient(payload={"status": "ok"})
    async def _fake_get_client() -> _DummyClient:
        return client

    backend._get_client = _fake_get_client  # type: ignore[method-assign]

    req = MFNRequest(
        request_type="simulation",
        request_id="req-2",
        timestamp=datetime.now(timezone.utc),
        seeds=[3.0],
        grid_size=16,
    )

    result = asyncio.run(backend.run_simulation(req))

    assert result.success is True
    assert client.calls[0][0] == "/api/v1/simulation/run"
