# mypy: ignore-errors
"""Tests for DFA gamma estimator — non-stationary robust."""

from __future__ import annotations

import numpy as np
import pytest

from geosync.estimators.dfa_gamma_estimator import DFAGammaEstimator


@pytest.fixture
def est():
    return DFAGammaEstimator(bootstrap_n=50)


def test_white_noise_H_near_half(est):
    rng = np.random.default_rng(42)
    g = est.compute(rng.standard_normal(2048))
    assert g.is_valid
    assert 0.3 <= g.hurst <= 0.7, f"White noise H={g.hurst}"
    assert abs(g.hurst - (g.value - 1) / 2) < 1e-4


def test_persistent_H_above_half(est):
    rng = np.random.default_rng(42)
    # Cumsum of white noise = random walk = H≈1.0 for path
    g = est.compute(np.cumsum(rng.standard_normal(2048)))
    assert g.is_valid
    assert g.hurst > 0.7, f"Persistent H={g.hurst}"
    assert g.value > 2.0


def test_anti_persistent(est):
    rng = np.random.default_rng(42)
    g = est.compute(np.diff(rng.standard_normal(2049)))
    # Anti-persistent: H < 0.5
    if g.is_valid:
        assert g.hurst < 0.5


def test_short_series_invalid(est):
    rng = np.random.default_rng(42)
    for n in [10, 30, 64, 127]:
        g = est.compute(rng.standard_normal(n))
        assert not g.is_valid


def test_ci_ordered(est):
    rng = np.random.default_rng(42)
    g = est.compute(np.cumsum(rng.standard_normal(1024)))
    if g.is_valid:
        assert g.ci_low <= g.ci_high


def test_hurst_consistency(est):
    rng = np.random.default_rng(42)
    for _ in range(10):
        g = est.compute(np.cumsum(rng.standard_normal(512)))
        if g.is_valid:
            assert abs(g.hurst - (g.value - 1) / 2) < 1e-4


def test_regime_shift_detection(est):
    rng = np.random.default_rng(42)
    # First half: persistent. Second half: noise.
    series = np.concatenate(
        [
            np.cumsum(rng.standard_normal(512)),
            rng.standard_normal(512),
        ]
    )
    g = est.compute(series)
    assert isinstance(g.regime_shift, bool)


def test_gamma_never_assigned_one(est):
    for seed in range(20):
        g = est.compute(np.random.default_rng(seed).standard_normal(512))
        if g.is_valid:
            assert g.value != 1.0


def test_quality_bounded(est):
    rng = np.random.default_rng(42)
    g = est.compute(rng.standard_normal(1024))
    assert 0.0 <= g.quality <= 1.0
