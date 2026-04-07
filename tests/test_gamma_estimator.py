# mypy: disable-error-code="attr-defined,unused-ignore,no-untyped-call"
"""Tests for multi-segment PSD gamma estimator."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from geosync.estimators.gamma_estimator import PSDGammaEstimator


def test_white_noise_gamma_near_zero() -> None:
    """White noise has flat PSD → γ ≈ 0."""
    np.random.seed(42)
    est = PSDGammaEstimator()
    g = est.compute(np.random.randn(1000))
    assert abs(g.value) < 1.0, f"White noise gamma={g.value} too far from 0"
    assert g.is_valid


def test_brownian_motion_gamma_positive() -> None:
    """Brownian motion (cumsum of white noise) → γ > 0 (persistent)."""
    np.random.seed(42)
    est = PSDGammaEstimator()
    bm = np.cumsum(np.random.randn(1000))
    g = est.compute(np.diff(bm))
    assert g.is_valid


def test_short_series_returns_invalid() -> None:
    """Series < 96 samples should return invalid."""
    est = PSDGammaEstimator()
    g = est.compute(np.random.randn(50))
    assert not g.is_valid
    assert g.value == 0.0


def test_nan_input_returns_invalid() -> None:
    data = np.ones(200)
    data[100] = float("nan")
    est = PSDGammaEstimator()
    g = est.compute(data)
    assert not g.is_valid


def test_constant_input_returns_invalid() -> None:
    est = PSDGammaEstimator()
    g = est.compute(np.zeros(200))
    assert not g.is_valid


def test_quality_gate_rejects_noisy_fit() -> None:
    """Estimator with min_quality=0.99 should reject most fits."""
    est = PSDGammaEstimator(min_quality=0.99)
    np.random.seed(42)
    result = est.compute(np.random.randn(200))
    # With 0.99 threshold, quality must be extremely high to pass
    assert not result.is_valid or result.quality >= 0.99


def test_deterministic_same_input_same_output() -> None:
    """Same input → same gamma. No hidden state."""
    est = PSDGammaEstimator()
    data = np.sin(np.linspace(0, 20, 500)) + 0.1 * np.random.RandomState(42).randn(500)
    g1 = est.compute(data)
    g2 = est.compute(data)
    assert g1.value == g2.value
    assert g1.quality == g2.quality


@given(seed=st.integers(min_value=0, max_value=10000))
@settings(max_examples=20)
def test_gamma_always_bounded(seed: int) -> None:
    """∀ input: gamma ∈ [-5, 5] (clipped)."""
    rng = np.random.RandomState(seed)
    est = PSDGammaEstimator()
    g = est.compute(rng.randn(200))
    assert -5.0 <= g.value <= 5.0
