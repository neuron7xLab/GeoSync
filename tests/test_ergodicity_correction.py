# mypy: ignore-errors
"""Tests for Peters-Kelly ergodicity correction."""

from __future__ import annotations

import numpy as np

from geosync.neuroeconomics.ergodicity_correction import ErgodicityCorrection


def test_pure_noise_severe_or_abort():
    """μ ≈ 0 → no edge → SEVERE or ABORT."""
    ec = ErgodicityCorrection()
    rng = np.random.default_rng(42)
    state = ec.update(rng.standard_normal(500))
    assert state.regime in ("SEVERE", "ABORT")
    assert state.pragmatic_discount <= 0.5


def test_strong_trend_ergodic():
    """Strong drift, low vol → ERGODIC."""
    ec = ErgodicityCorrection()
    returns = np.full(500, 0.01) + np.random.default_rng(42).standard_normal(500) * 0.001
    state = ec.update(returns)
    assert state.regime == "ERGODIC"
    assert state.nei < 0.5


def test_gap_always_non_negative():
    """σ²/2 ≥ 0 for ANY input."""
    ec = ErgodicityCorrection()
    rng = np.random.default_rng(42)
    for _ in range(50):
        state = ec.update(rng.standard_normal(200))
        assert state.ergodicity_gap >= 0.0


def test_kelly_bounded():
    """Kelly ∈ [0, 1] always."""
    ec = ErgodicityCorrection()
    rng = np.random.default_rng(42)
    for _ in range(50):
        state = ec.update(rng.standard_normal(200) * 0.01 + 0.001)
        assert 0.0 <= state.kelly_corrected <= 1.0


def test_discount_bounded():
    """Pragmatic discount ∈ [0, 1] always."""
    ec = ErgodicityCorrection()
    rng = np.random.default_rng(42)
    for _ in range(50):
        state = ec.update(rng.standard_normal(200))
        assert 0.0 <= state.pragmatic_discount <= 1.0


def test_correct_pragmatic():
    """correct_pragmatic reduces value by discount."""
    ec = ErgodicityCorrection()
    state = ec.update(np.random.default_rng(42).standard_normal(300))
    corrected = ec.correct_pragmatic(1.0, state)
    assert corrected <= 1.0
    assert corrected >= 0.0
    assert corrected == state.pragmatic_discount


def test_short_series_abort():
    """< 10 returns → ABORT."""
    ec = ErgodicityCorrection()
    state = ec.update(np.array([0.01, -0.01, 0.02]))
    assert state.regime == "ABORT"
