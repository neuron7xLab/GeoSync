# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Ollivier-κ runtime policy: typed status, perf budget, cache, degraded mode.

Covers the hardening of ``_ollivier_worst_edge_kappa`` (the heaviest term in the
coherence-bridge hot path):
  * typed :class:`OllivierStatus` outcomes (no silent NaN swallow),
  * deterministic max-edges perf budget,
  * content+policy-keyed cache,
  * degraded behaviour: ABSTAIN (field omitted) by default, NO_GO (NaN) only
    under ``COHERENCE_REQUIRE_OLLIVIER=1`` — never a fabricated in-band κ,
  * a p95 latency benchmark against the default budget.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator

import numpy as np
import pytest

from coherence_bridge import geosync_adapter as ga
from coherence_bridge.geosync_adapter import (
    OllivierStatus,
    _ollivier_worst_edge_kappa,
    ollivier_required,
)


@pytest.fixture(autouse=True)
def _clean_env_and_cache() -> Iterator[None]:
    saved = {
        k: os.environ.get(k)
        for k in (
            "COHERENCE_OLLIVIER_MAX_EDGES",
            "COHERENCE_OLLIVIER_MAX_MS",
            "COHERENCE_REQUIRE_OLLIVIER",
        )
    }
    for k in saved:
        os.environ.pop(k, None)
    ga._OLLIVIER_CACHE.clear()
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        ga._OLLIVIER_CACHE.clear()


def _prices(seed: int = 7, n: int = 200) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 1.10 * np.exp(np.cumsum(rng.standard_normal(n) * 1e-3))


def test_status_ok_on_normal_prices() -> None:
    kappa, status = _ollivier_worst_edge_kappa(_prices())
    assert status is OllivierStatus.OK
    assert np.isfinite(kappa)
    assert kappa <= 1.0 + 1e-9  # INV-RC1 upper bound


def test_perf_budget_exceeded_is_typed_nan() -> None:
    os.environ["COHERENCE_OLLIVIER_MAX_EDGES"] = "1"
    kappa, status = _ollivier_worst_edge_kappa(_prices())
    assert status is OllivierStatus.PERF_BUDGET_EXCEEDED
    assert not np.isfinite(kappa)  # NaN — never a fabricated value


def test_cache_returns_identical_result() -> None:
    p = _prices()
    first = _ollivier_worst_edge_kappa(p)
    assert ga._OLLIVIER_CACHE  # populated
    second = _ollivier_worst_edge_kappa(p)
    assert first == second


def test_cache_honours_budget_change_not_stale() -> None:
    p = _prices()
    _, ok = _ollivier_worst_edge_kappa(p)
    assert ok is OllivierStatus.OK
    os.environ["COHERENCE_OLLIVIER_MAX_EDGES"] = "1"  # tighten budget
    _, degraded = _ollivier_worst_edge_kappa(p)
    assert degraded is OllivierStatus.PERF_BUDGET_EXCEEDED  # not the stale OK


def test_required_flag_parsing() -> None:
    assert ollivier_required() is False
    os.environ["COHERENCE_REQUIRE_OLLIVIER"] = "1"
    assert ollivier_required() is True
    os.environ["COHERENCE_REQUIRE_OLLIVIER"] = "true"
    assert ollivier_required() is True
    os.environ["COHERENCE_REQUIRE_OLLIVIER"] = "0"
    assert ollivier_required() is False


def test_benchmark_p95_within_default_budget() -> None:
    """p95 single-graph compute latency stays within the default 750 ms budget.

    Synthetic 200-bar series (the integration-test shape). Cache is cleared each
    iteration so we time the real compute, not a cache hit.
    """
    samples_ms: list[float] = []
    for seed in range(12):
        ga._OLLIVIER_CACHE.clear()
        p = _prices(seed=seed)
        t0 = time.perf_counter()
        _, status = _ollivier_worst_edge_kappa(p)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
        assert status is OllivierStatus.OK
    samples_ms.sort()
    p95 = samples_ms[int(0.95 * (len(samples_ms) - 1))]
    assert p95 <= ga._OLLIVIER_MAX_MS_DEFAULT, f"p95={p95:.1f}ms over budget"


# --- adapter-level degraded behaviour (ABSTAIN vs NO_GO) ----------------------


def _adapter_signal_under_env() -> dict[str, object]:
    import pandas as pd

    from coherence_bridge.geosync_adapter import GeoSyncAdapter

    rng = np.random.default_rng(7)
    n = 200
    close = 1.10 * np.exp(np.cumsum(rng.standard_normal(n) * 1e-3))
    df = pd.DataFrame(
        {
            "open": np.concatenate([[close[0]], close[:-1]]),
            "high": close * 1.0005,
            "low": close * 0.9995,
            "close": close,
            "volume": rng.integers(100, 1000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="min"),
    )
    adapter = GeoSyncAdapter()
    inst = adapter.instruments[0]
    adapter.update_market_data(inst, df)
    sig = adapter.get_signal(inst)
    assert sig is not None
    return sig


@pytest.mark.integration
def test_adapter_degraded_non_required_abstains() -> None:
    """Degraded κ + not-required ⇒ ollivier_kappa OMITTED (ABSTAIN), status set."""
    os.environ["COHERENCE_OLLIVIER_MAX_EDGES"] = "1"  # force degraded
    sig = _adapter_signal_under_env()
    assert sig["ollivier_status"] == OllivierStatus.PERF_BUDGET_EXCEEDED.value
    assert "ollivier_kappa" not in sig  # ABSTAIN, not a fabricated value


@pytest.mark.integration
def test_adapter_degraded_required_refuses() -> None:
    """Degraded κ + required ⇒ ollivier_kappa = NaN (downstream contract NO_GO)."""
    os.environ["COHERENCE_OLLIVIER_MAX_EDGES"] = "1"
    os.environ["COHERENCE_REQUIRE_OLLIVIER"] = "1"
    sig = _adapter_signal_under_env()
    assert sig["ollivier_status"] == OllivierStatus.PERF_BUDGET_EXCEEDED.value
    assert "ollivier_kappa" in sig
    kappa = sig["ollivier_kappa"]
    assert isinstance(kappa, float)
    assert not np.isfinite(kappa)
