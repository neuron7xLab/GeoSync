# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Every OMS terminal path must discharge the risk reservation — no leak.

The reservation flag is only safe to enable if EVERY way an order can leave the
live set releases its hold. Prior tests covered fill / placement-reject / cancel.
These add the remaining audited paths: venue-reported cancel and reject via
sync_remote_state, and requeue. After each, open_reservation_count() must be zero
— a leak would falsely block future admissions.
"""

from __future__ import annotations

import importlib
from dataclasses import replace

import pytest

from domain import Order, OrderSide, OrderStatus, OrderType

# ``execution.*`` is behind the forbidden_import_patterns gate → importlib.
_oms = importlib.import_module("execution.oms")
_risk = importlib.import_module("execution.risk")
_connectors = importlib.import_module("execution.connectors")

OMSConfig = _oms.OMSConfig
OrderManagementSystem = _oms.OrderManagementSystem
RiskManager = _risk.RiskManager
RiskLimits = _risk.RiskLimits
ExecutionConnector = _connectors.ExecutionConnector


class _SeqConnector(ExecutionConnector):
    def __init__(self) -> None:
        super().__init__(sandbox=True)
        self._counter = 0

    def place_order(self, order: Order, *, idempotency_key: str | None = None) -> Order:
        submitted = replace(order)
        if not submitted.order_id:
            submitted.mark_submitted(f"brk-{self._counter:04d}")
        self._counter += 1
        self._orders[submitted.order_id] = submitted
        return submitted

    def cancel_order(self, order_id: str) -> bool:
        return True


def _oms_with(manager: "RiskManager", tmp_path) -> "OrderManagementSystem":
    config = OMSConfig(
        state_path=tmp_path / "oms-state.json",
        auto_persist=False,
        max_retries=1,
        reserve_risk_exposure=True,
    )
    return OrderManagementSystem(_SeqConnector(), manager, config, lifecycle=None)


def _order() -> Order:
    return Order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=2.0,
        price=100.0,
        order_type=OrderType.LIMIT,
    )


def _submit_and_place(oms: "OrderManagementSystem", manager: "RiskManager", cid: str) -> Order:
    oms.submit(_order(), correlation_id=cid)
    assert manager.open_reservation_count() == 1
    placed = oms.process_next()
    assert placed.order_id is not None
    return placed


def test_sync_remote_cancel_releases_reservation(tmp_path) -> None:
    manager = RiskManager(RiskLimits(max_position=5.0, max_notional=float("inf")))
    oms = _oms_with(manager, tmp_path)
    placed = _submit_and_place(oms, manager, "c1")

    oms.sync_remote_state(replace(placed, status=OrderStatus.CANCELLED))

    assert manager.pending_position("BTCUSDT") == 0.0
    assert manager.open_reservation_count() == 0


def test_sync_remote_reject_releases_reservation(tmp_path) -> None:
    manager = RiskManager(RiskLimits(max_position=5.0, max_notional=float("inf")))
    oms = _oms_with(manager, tmp_path)
    placed = _submit_and_place(oms, manager, "c1")

    oms.sync_remote_state(
        replace(placed, status=OrderStatus.REJECTED, rejection_reason="venue reject")
    )

    assert manager.pending_position("BTCUSDT") == 0.0
    assert manager.open_reservation_count() == 0


def test_requeue_releases_the_prior_reservation(tmp_path) -> None:
    manager = RiskManager(RiskLimits(max_position=5.0, max_notional=float("inf")))
    oms = _oms_with(manager, tmp_path)
    placed = _submit_and_place(oms, manager, "c1")

    oms.requeue_order(placed.order_id)

    # The discarded order instance leaves no hold behind.
    assert manager.open_reservation_count() == 0
    assert manager.pending_position("BTCUSDT") == 0.0


def test_full_lifecycle_leaves_no_open_reservations(tmp_path) -> None:
    # A mix of terminal outcomes on distinct symbols must all net to zero holds.
    manager = RiskManager(RiskLimits(max_position=50.0, max_notional=float("inf")))
    oms = _oms_with(manager, tmp_path)

    filled = _submit_and_place(oms, manager, "fill")
    oms.register_fill(filled.order_id, 2.0, 100.0)  # convert

    cancelled = _submit_and_place(oms, manager, "cancel")
    assert oms.cancel(cancelled.order_id) is True  # release

    synced = _submit_and_place(oms, manager, "sync")
    oms.sync_remote_state(replace(synced, status=OrderStatus.CANCELLED))  # release

    assert manager.open_reservation_count() == 0
