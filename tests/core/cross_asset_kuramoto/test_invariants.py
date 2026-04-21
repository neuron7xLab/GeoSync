"""T4: INV-CAK1..CAK8 — each invariant has a dedicated test; failing-closed
on violations is verified (INV-CAK6)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from core.cross_asset_kuramoto import (
    build_panel,
    compute_log_returns,
    extract_phase,
    kuramoto_order,
)
from core.cross_asset_kuramoto.invariants import (
    CAKInvariantError,
    assert_cak1_parameter_freeze,
    assert_cak2_universe_freeze,
    assert_cak3_deterministic,
    assert_cak5_cost_required,
    assert_cak7_scale_invariance,
    assert_cak8_turnover_bounded,
    load_parameter_lock,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_PATH = REPO_ROOT / "results" / "cross_asset_kuramoto" / "PARAMETER_LOCK.json"
SPIKE_DATA = Path.home() / "spikes" / "cross_asset_sync_regime" / "data"


@pytest.fixture(scope="module")
def params():
    return load_parameter_lock(LOCK_PATH)


# -------- INV-CAK1 Parameter freeze -------- #


def test_cak1_pass_on_unmodified_params(params) -> None:
    assert_cak1_parameter_freeze(params, LOCK_PATH)


def test_cak1_fail_closed_on_param_divergence(params) -> None:
    bad = replace(params, cost_bps=params.cost_bps + 1.0)
    with pytest.raises(CAKInvariantError, match="INV-CAK1"):
        assert_cak1_parameter_freeze(bad, LOCK_PATH)


# -------- INV-CAK2 Universe freeze -------- #


def test_cak2_pass_on_unmodified_universe(params) -> None:
    assert_cak2_universe_freeze(params, LOCK_PATH)


def test_cak2_fail_closed_on_universe_drift(params) -> None:
    bad = replace(params, strategy_assets=params.strategy_assets + ("SOL",))
    with pytest.raises(CAKInvariantError, match="INV-CAK2"):
        assert_cak2_universe_freeze(bad, LOCK_PATH)


# -------- INV-CAK3 Determinism -------- #


def test_cak3_on_identity() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = a.copy()
    assert_cak3_deterministic(a, b)


def test_cak3_fail_closed_on_difference() -> None:
    with pytest.raises(CAKInvariantError, match="INV-CAK3"):
        assert_cak3_deterministic(np.array([1.0]), np.array([1.1]))


# -------- INV-CAK4 No future leak — covered in test_no_future_leak.py -------- #


# -------- INV-CAK5 Cost model -------- #


def test_cak5_requires_positive_cost_when_emitting_performance() -> None:
    assert_cak5_cost_required(cost_bps=10.0, emit_performance=True)  # OK
    # Emitting performance with zero cost must fail-closed
    with pytest.raises(CAKInvariantError, match="INV-CAK5"):
        assert_cak5_cost_required(cost_bps=0.0, emit_performance=True)


def test_cak5_allows_zero_cost_for_diagnostic_only() -> None:
    # diagnostic-only mode (emit_performance=False) tolerates cost=0
    assert_cak5_cost_required(cost_bps=0.0, emit_performance=False)


# -------- INV-CAK6 Fail-closed via ValueError subclass -------- #


def test_cak6_error_is_valueerror_subclass() -> None:
    assert issubclass(CAKInvariantError, ValueError)


# -------- INV-CAK7 Scale invariance -------- #


def test_cak7_price_scale_does_not_change_r(params) -> None:
    if not SPIKE_DATA.is_dir():
        pytest.skip("spike data not present")
    panel = build_panel(params.regime_assets, SPIKE_DATA, params.ffill_limit_bdays)
    scaled = panel * 137.5  # arbitrary positive constant
    r_base = kuramoto_order(
        extract_phase(compute_log_returns(panel), params.detrend_window_bdays).dropna(),
        params.r_window_bdays,
    )
    r_scaled = kuramoto_order(
        extract_phase(compute_log_returns(scaled), params.detrend_window_bdays).dropna(),
        params.r_window_bdays,
    )
    assert_cak7_scale_invariance(r_base, r_scaled, rtol=1e-10)


# -------- INV-CAK8 Turnover bounded -------- #


def test_cak8_empty_turnover_is_ok() -> None:
    assert_cak8_turnover_bounded(np.array([]))


def test_cak8_normal_turnover_is_ok() -> None:
    assert_cak8_turnover_bounded(np.array([0.1, 0.3, 1.5, 1.9]))


def test_cak8_fail_closed_on_excess() -> None:
    with pytest.raises(CAKInvariantError, match="INV-CAK8"):
        assert_cak8_turnover_bounded(np.array([0.5, 2.1]))


def test_cak8_fail_closed_on_nonfinite() -> None:
    with pytest.raises(CAKInvariantError, match="INV-CAK8"):
        assert_cak8_turnover_bounded(np.array([0.5, np.inf]))
