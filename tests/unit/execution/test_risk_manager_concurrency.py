# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Regression: RiskManager exposure accounting must be thread-safe.

The live runner (:mod:`execution.live_loop`) drives one :class:`RiskManager`
from three concurrent watchdog threads — order-submission (``validate_order``),
fill-poller (``register_fill``) and heartbeat (``exposure_snapshot`` reads).
Every exposure mutator is a read-modify-write on ``_positions`` /
``_last_notional``; without serialisation two fills that read the same prior
position both write it back, so one fill is silently lost.

The default CPython GIL switches on a ~5 ms timer, not per bytecode, so a naive
high-contention loop almost never interleaves inside the tiny read→write window
and would pass even with the bug present. To make the defect observable
deterministically, these tests install an instrumented ``_positions`` map that
rendezvouses two threads *after* each has read the old value and *before* either
writes — forcing exactly the interleave the lock must prevent.

Falsification: delete the ``@_synchronized`` guard on ``register_fill`` and
``test_register_fill_lost_update_is_prevented`` fails (final position 1.0, not
2.0). With the guard, the second thread blocks on ``_state_lock`` at method
entry, never reaches the rendezvous, the barrier times out, and the two fills
serialise to the correct total.
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Iterator, MutableMapping

import pytest

# ``execution.*`` is behind the forbidden_import_patterns gate → importlib.
_risk = importlib.import_module("execution.risk")
RiskLimits = _risk.RiskLimits
RiskManager = _risk.RiskManager

_SYMBOL = "BTC-USD"
_CANON = "BTC/USD"  # normalize_symbol("BTC-USD") canonical form
_PRICE = 10.0
_RENDEZVOUS_TIMEOUT = 1.0


class _RendezvousPositions(MutableMapping[str, float]):
    """A ``_positions`` map that pauses between the read and write of a fill.

    On the first ``get`` for the tracked symbol from each of two threads, the
    caller blocks on a shared barrier so both threads observe the *same* prior
    value before either writes. Under the lock only one thread can be inside
    ``register_fill`` at a time, so the second never arrives, the barrier times
    out (``BrokenBarrierError``) and the reads serialise correctly.
    """

    def __init__(self, symbol: str) -> None:
        self._data: dict[str, float] = {}
        self._symbol = symbol
        self._barrier = threading.Barrier(2, timeout=_RENDEZVOUS_TIMEOUT)
        self._armed = True

    def get(self, key: str, default: float | None = None) -> float:  # type: ignore[override]
        value = self._data.get(key, default if default is not None else 0.0)
        if self._armed and key == self._symbol:
            try:
                self._barrier.wait()
            except threading.BrokenBarrierError:
                # Serialised path (guard held): the peer never reached the read.
                self._armed = False
        return value

    def __getitem__(self, key: str) -> float:
        return self._data[key]

    def __setitem__(self, key: str, value: float) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def _unbounded_manager() -> RiskManager:
    # Uncapped so no LimitViolation fires; we probe state integrity, not caps.
    # No risk_state_store -> _persist_risk_state is a no-op.
    return RiskManager(RiskLimits(max_notional=float("inf"), max_position=float("inf")))


def _join_all(threads: list[threading.Thread]) -> None:
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
    assert all(not t.is_alive() for t in threads), "worker thread deadlocked"


def test_register_fill_lost_update_is_prevented() -> None:
    """Two concurrent fills that read the same prior position must both land."""

    manager = _unbounded_manager()
    manager._positions = _RendezvousPositions(_CANON)  # type: ignore[assignment]

    errors: list[BaseException] = []
    elock = threading.Lock()

    def _fill() -> None:
        try:
            manager.register_fill(_SYMBOL, "buy", 1.0, _PRICE)
        except BaseException as exc:  # noqa: BLE001 - surface thread failures
            with elock:
                errors.append(exc)

    _join_all([threading.Thread(target=_fill) for _ in range(2)])

    assert not errors, f"register_fill raised: {errors[:3]}"
    # Without the lock both fills read 0.0 and write 1.0 -> final 1.0 (one lost).
    assert manager.current_position(_SYMBOL) == 2.0
    assert manager.current_notional(_SYMBOL) == pytest.approx(2.0 * _PRICE)


def test_validate_and_register_do_not_deadlock_or_corrupt() -> None:
    """A validating thread and a filling thread interleave without corruption.

    Exercises the reentrant lock across the two hot-path entry points that the
    order-submission and fill-poller watchdog threads hit concurrently, asserting
    the guard neither deadlocks nor drops a fill under a plain interleave.
    """

    manager = RiskManager(
        RiskLimits(
            max_notional=float("inf"),
            max_position=float("inf"),
            max_orders_per_interval=0,  # disable throttle so validate always admits
        )
    )

    errors: list[BaseException] = []
    elock = threading.Lock()

    def _validate() -> None:
        try:
            for _ in range(500):
                manager.validate_order(_SYMBOL, "buy", 1.0, _PRICE)
        except BaseException as exc:  # noqa: BLE001
            with elock:
                errors.append(exc)

    def _fill() -> None:
        try:
            for _ in range(500):
                manager.register_fill(_SYMBOL, "buy", 1.0, _PRICE)
        except BaseException as exc:  # noqa: BLE001
            with elock:
                errors.append(exc)

    _join_all([threading.Thread(target=_validate), threading.Thread(target=_fill)])

    assert not errors, f"concurrent validate/register raised: {errors[:3]}"
    # validate_order never mutates position; every one of the 500 fills must land.
    assert manager.current_position(_SYMBOL) == 500.0


def test_exposure_snapshot_matches_position_after_concurrent_fills() -> None:
    """After concurrent fills the snapshot's notional matches its position."""

    manager = _unbounded_manager()

    def _fill() -> None:
        for _ in range(400):
            manager.register_fill(_SYMBOL, "buy", 1.0, _PRICE)

    _join_all([threading.Thread(target=_fill) for _ in range(4)])

    snap = manager.exposure_snapshot()[_CANON]
    assert snap["position"] == pytest.approx(1600.0)
    assert snap["notional"] == pytest.approx(abs(snap["position"]) * _PRICE)
