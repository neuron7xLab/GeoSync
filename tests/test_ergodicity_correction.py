# mypy: ignore-errors
"""Tests for Peters-Kelly ergodicity correction (SDE formulation)."""

from __future__ import annotations

import numpy as np

from geosync.neuroeconomics.ergodicity_correction import ErgodicityCorrection


def test_sde_correction_equals_neg_sigma_sq_half() -> None:
    """Δg = -σ²/2 analytically verified."""
    ec = ErgodicityCorrection()
    rng = np.random.default_rng(42)
    for _ in range(20):
        returns = rng.standard_normal(500) * 0.02
        state = ec.update(returns)
        expected = -(state.volatility**2) / 2.0
        assert abs(state.sde_drift_correction - expected) < 1e-6


def test_time_average_drift_formula() -> None:
    """time_average = ensemble - σ²/2."""
    ec = ErgodicityCorrection()
    returns = np.full(500, 0.01) + np.random.default_rng(42).standard_normal(500) * 0.001
    state = ec.update(returns)
    expected_time = state.ensemble_drift + state.sde_drift_correction
    assert abs(state.time_average_drift - expected_time) < 1e-6


def test_high_volatility_non_ergodic() -> None:
    """High σ, low μ → is_ergodic=False."""
    ec = ErgodicityCorrection()
    returns = np.random.default_rng(42).standard_normal(500) * 0.5
    state = ec.update(returns)
    assert not state.is_ergodic


def test_near_zero_vol_ergodic() -> None:
    """σ ≈ 0 → is_ergodic=True."""
    ec = ErgodicityCorrection()
    returns = np.full(500, 0.01) + np.random.default_rng(42).standard_normal(500) * 1e-5
    state = ec.update(returns)
    assert state.is_ergodic


def test_kelly_bounded() -> None:
    """Kelly ∈ [0, 1] always."""
    ec = ErgodicityCorrection()
    rng = np.random.default_rng(42)
    for _ in range(50):
        state = ec.update(rng.standard_normal(200) * 0.01 + 0.001)
        assert 0.0 <= state.kelly_corrected <= 1.0


def test_pragmatic_bounded() -> None:
    """pragmatic_corrected ∈ [0, 1] always."""
    ec = ErgodicityCorrection()
    rng = np.random.default_rng(42)
    for _ in range(50):
        state = ec.update(rng.standard_normal(200))
        assert 0.0 <= state.pragmatic_corrected <= 1.0


def test_correct_pragmatic_applies_discount() -> None:
    """correct_pragmatic reduces value by pragmatic_corrected."""
    ec = ErgodicityCorrection()
    state = ec.update(np.random.default_rng(42).standard_normal(300))
    corrected = ec.correct_pragmatic(1.0, state)
    assert corrected == state.pragmatic_corrected
    assert 0.0 <= corrected <= 1.0


def test_short_series_safe() -> None:
    """< 10 returns → safe defaults."""
    ec = ErgodicityCorrection()
    state = ec.update(np.array([0.01, -0.01, 0.02]))
    assert not state.is_ergodic
    assert state.nei == 999.0


def test_gap_always_non_negative() -> None:
    """Ergodicity gap = σ²/2 ≥ 0."""
    ec = ErgodicityCorrection()
    rng = np.random.default_rng(42)
    for _ in range(50):
        state = ec.update(rng.standard_normal(200))
        # gap = -sde_drift_correction
        assert state.sde_drift_correction <= 0.0 + 1e-10


def test_frozen_dataclass() -> None:
    """ErgodicityResult is frozen."""
    ec = ErgodicityCorrection()
    state = ec.update(np.random.default_rng(42).standard_normal(300))
    with pytest.raises(AttributeError):
        state.nei = 0.0  # type: ignore[misc]


import pytest  # noqa: E402
