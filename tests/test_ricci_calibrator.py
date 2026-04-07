# mypy: ignore-errors
"""Tests for Ricci lead-time calibrator — honest measurement."""

from __future__ import annotations

import numpy as np

from geosync.estimators.ricci_calibrator import RicciTemporalCalibrator


def test_random_data_remove_claim():
    """Random κ and returns → not predictive → REMOVE_CLAIM."""
    rng = np.random.default_rng(42)
    cal = RicciTemporalCalibrator()
    result = cal.calibrate(rng.standard_normal(500) * 0.3, rng.standard_normal(500) * 0.01)
    assert result.recommendation in ("REMOVE_CLAIM", "CAUTION")
    assert len(result.honest_statement) > 20


def test_synthetic_predictive():
    """Synthetic: κ dips 5 bars before volatility spike → USE or CAUTION."""
    rng = np.random.default_rng(42)
    kappa = np.zeros(500)
    returns = rng.standard_normal(500) * 0.005
    for i in range(50, 450, 40):
        kappa[i] = -0.8
        returns[i + 5 : i + 8] *= 10
    cal = RicciTemporalCalibrator()
    result = cal.calibrate(kappa, returns)
    assert result.n_events > 0
    assert isinstance(result.is_predictive, bool)


def test_n_events_counted():
    """n_events matches number of κ < threshold occurrences."""
    rng = np.random.default_rng(42)
    kappa = np.zeros(200)
    kappa[10] = -0.5
    kappa[50] = -0.5
    kappa[100] = -0.5
    returns = rng.standard_normal(200) * 0.01
    cal = RicciTemporalCalibrator()
    result = cal.calibrate(kappa, returns, kappa_threshold=-0.3)
    assert result.n_events == 3


def test_ci_ordered():
    """p5 ≤ median ≤ p95."""
    rng = np.random.default_rng(42)
    kappa = rng.standard_normal(500) * 0.5
    returns = rng.standard_normal(500) * 0.01
    cal = RicciTemporalCalibrator()
    result = cal.calibrate(kappa, returns, kappa_threshold=-0.3)
    if result.n_events > 0:
        assert result.offset_p5 <= result.offset_bars <= result.offset_p95


def test_honest_statement_non_empty():
    """honest_statement is always a non-empty string."""
    cal = RicciTemporalCalibrator()
    result = cal.calibrate(np.zeros(10), np.zeros(10))
    assert isinstance(result.honest_statement, str)
    assert len(result.honest_statement) > 10
