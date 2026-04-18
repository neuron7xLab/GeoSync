# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Mathematical invariants of the DRO-ARA v7 engine.

These tests pin the derivation contract ``gamma = 2·H + 1`` and validate the
fail-closed posture against degenerate inputs. They are pure unit tests: no
stochastic generators, no external data, no non-determinism.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.dro_ara import (
    H_CRITICAL,
    H_DRIFT,
    R2_MIN,
    RS_LONG_THRESH,
    Regime,
    classify,
    derive_gamma,
    geosync_observe,
    risk_scalar,
)


def test_gamma_is_derived_from_H() -> None:
    rng = np.random.default_rng(42)
    price = 100.0 + np.cumsum(rng.normal(0, 1, size=2048))
    gamma, H, r2 = derive_gamma(price)
    assert abs(gamma - (2 * H + 1)) < 1e-5
    assert 0.01 <= H <= 0.99
    assert 0.0 <= r2 <= 1.0


def test_risk_scalar_bounds() -> None:
    assert risk_scalar(1.0) == 1.0
    assert risk_scalar(0.0) == 0.0
    assert risk_scalar(2.0) == 0.0
    assert 0.0 <= risk_scalar(1.4) <= 1.0
    assert risk_scalar(-5.0) == 0.0


def test_classify_invalid_when_not_stationary() -> None:
    assert classify(gamma=1.8, r2=0.95, stationary=False) is Regime.INVALID


def test_classify_invalid_when_r2_below_gate() -> None:
    assert classify(gamma=1.8, r2=R2_MIN - 0.01, stationary=True) is Regime.INVALID


def test_classify_critical_boundary() -> None:
    gamma_critical = 2 * (H_CRITICAL - 0.01) + 1
    assert classify(gamma=gamma_critical, r2=0.95, stationary=True) is Regime.CRITICAL


def test_classify_drift_boundary() -> None:
    gamma_drift = 2 * (H_DRIFT + 0.01) + 1
    assert classify(gamma=gamma_drift, r2=0.95, stationary=True) is Regime.DRIFT


def test_classify_transition() -> None:
    gamma_mid = 2 * 0.50 + 1
    assert classify(gamma=gamma_mid, r2=0.95, stationary=True) is Regime.TRANSITION


def test_rs_long_threshold_constant() -> None:
    assert RS_LONG_THRESH == 0.33


def test_observe_rejects_nan() -> None:
    price = np.full(1024, 100.0)
    price[500] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        geosync_observe(price)


def test_observe_rejects_inf() -> None:
    price = np.full(1024, 100.0)
    price[500] = np.inf
    with pytest.raises(ValueError, match="NaN/Inf"):
        geosync_observe(price)


def test_observe_rejects_constant() -> None:
    with pytest.raises(ValueError, match="constant"):
        geosync_observe(np.full(1024, 100.0))


def test_observe_rejects_too_short() -> None:
    with pytest.raises(ValueError, match="need"):
        geosync_observe(np.arange(10, dtype=np.float64))


def test_observe_rejects_2d() -> None:
    with pytest.raises(ValueError, match="1-D required"):
        geosync_observe(np.zeros((100, 100)))


def test_observe_deterministic() -> None:
    rng = np.random.default_rng(17)
    price = 100.0 + np.cumsum(rng.normal(0, 1, size=1024))
    a = geosync_observe(price)
    b = geosync_observe(price)
    assert a == b
