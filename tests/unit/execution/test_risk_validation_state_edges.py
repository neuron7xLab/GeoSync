# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Validation-state edge tests for execution risk core."""

from __future__ import annotations

import pytest

from execution.risk import LimitViolation, OrderRateExceeded, RiskLimits, RiskManager


def test_position_rejection_does_not_consume_rate_slot_or_mutate_exposure() -> None:
    manager = RiskManager(
        RiskLimits(
            max_position=1.0,
            max_notional=1_000.0,
            max_orders_per_interval=1,
            interval_seconds=60.0,
            kill_switch_violation_threshold=99,
        ),
        time_source=lambda: 100.0,
    )
    manager.register_fill("btc_usdt", "buy", qty=0.5, price=100.0)

    with pytest.raises(LimitViolation, match="Position cap exceeded"):
        manager.validate_order("BTC/USDT", "buy", qty=1.0, price=100.0)

    assert manager.current_position("BTC/USDT") == pytest.approx(0.5)
    assert manager.current_notional("BTC/USDT") == pytest.approx(50.0)

    manager.validate_order("BTC/USDT", "sell", qty=0.1, price=100.0)
    with pytest.raises(OrderRateExceeded, match="Order throttle exceeded"):
        manager.validate_order("BTC/USDT", "sell", qty=0.1, price=100.0)


def test_successful_validation_does_not_commit_exposure_until_fill() -> None:
    manager = RiskManager(
        RiskLimits(
            max_position=10.0,
            max_notional=10_000.0,
            max_orders_per_interval=0,
        )
    )

    manager.validate_order("eth_usdt", "buy", qty=2.0, price=1_000.0)

    assert manager.current_position("ETH/USDT") == pytest.approx(0.0)
    assert manager.current_notional("ETH/USDT") == pytest.approx(0.0)
    assert manager.exposure_snapshot() == {}

    manager.register_fill("eth_usdt", "buy", qty=2.0, price=1_000.0)
    assert manager.exposure_snapshot() == {
        "ETH/USDT": {"position": 2.0, "notional": 2_000.0}
    }


def test_valid_order_resets_prior_limit_violation_streak() -> None:
    manager = RiskManager(
        RiskLimits(
            max_position=1.0,
            max_notional=1_000.0,
            max_orders_per_interval=0,
            kill_switch_violation_threshold=5,
        )
    )
    manager.register_fill("btc_usdt", "buy", qty=0.9, price=100.0)

    with pytest.raises(LimitViolation):
        manager.validate_order("btc_usdt", "buy", qty=0.2, price=100.0)
    assert manager._limit_violation_streak == 1

    manager.validate_order("btc_usdt", "sell", qty=0.1, price=100.0)

    assert manager._limit_violation_streak == 0
    assert manager.current_position("BTC/USDT") == pytest.approx(0.9)
