# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Calibration edge tests for execution risk core invariants."""

from __future__ import annotations

from pathlib import Path

import pytest

from execution.risk import IdempotentRetryExecutor, JsonRiskStateStore, RiskLimits


def test_risk_limits_normalise_safety_thresholds() -> None:
    limits = RiskLimits(
        max_orders_per_interval=-5,
        interval_seconds=-1.0,
        kill_switch_limit_multiplier=0.25,
        kill_switch_violation_threshold=0,
        kill_switch_rate_limit_threshold=0,
        max_relative_drawdown=25,
    )

    assert limits.max_orders_per_interval == 0
    assert limits.interval_seconds == pytest.approx(0.0)
    assert limits.kill_switch_limit_multiplier == pytest.approx(1.0)
    assert limits.kill_switch_violation_threshold == 1
    assert limits.kill_switch_rate_limit_threshold == 1
    assert limits.max_relative_drawdown == pytest.approx(0.25)

    with pytest.raises(ValueError, match="max_relative_drawdown must be positive"):
        RiskLimits(max_relative_drawdown=0)
    with pytest.raises(ValueError, match="max_relative_drawdown must be <= 100"):
        RiskLimits(max_relative_drawdown=101)


def test_json_risk_state_store_sanitises_persisted_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "risk_state.json"
    path.write_text(
        '{"positions":{" BTCUSDT ":"2.5","":"7","ETHUSDT":"bad"},'
        '"last_notional":{"BTCUSDT":"50000","ETHUSDT":null}}'
    )

    positions, notionals = JsonRiskStateStore(path).load() or ({}, {})

    assert positions == {"BTCUSDT": 2.5}
    assert notionals == {"BTCUSDT": 50_000.0}


def test_idempotent_retry_executor_caches_first_success() -> None:
    attempts: list[int] = []
    executor = IdempotentRetryExecutor(backoff=lambda attempt: 0.0)

    def flaky(attempt: int) -> str:
        attempts.append(attempt)
        if attempt == 1:
            raise ValueError("transient")
        return f"ok-{attempt}"

    result = executor.run("order-state", flaky, retries=3, retry_exceptions=(ValueError,))
    cached = executor.run("order-state", lambda attempt: "must-not-run")

    assert result == "ok-2"
    assert cached == "ok-2"
    assert attempts == [1, 2]
