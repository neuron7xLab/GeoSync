# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Unit tests for the DRO-ARA regime filter.

Tests use real :func:`core.dro_ara.geosync_observe` on seeded synthetic series —
no mocks, no fakes.  Fail-closed multiplier semantics on INVALID/DRIFT are
pinned exactly (==, not isclose).
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from core.strategies.dro_ara_filter import (
    MULTIPLIER_CRITICAL,
    MULTIPLIER_DRIFT,
    MULTIPLIER_INVALID,
    MULTIPLIER_TRANSITION,
    apply_regime_filter,
    regime_multiplier,
)


def _ou(seed: int, n: int = 2048) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    x = np.empty(n, dtype=np.float64)
    x[0] = 100.0
    for t in range(1, n):
        x[t] = x[t - 1] + 0.08 * (100.0 - x[t - 1]) + 0.6 * rng.normal()
    return x


def _gbm(seed: int, n: int = 2048) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    r = 0.002 + 0.01 * rng.normal(size=n)
    return 100.0 * np.exp(np.cumsum(r))


def test_invalid_regime_is_exactly_zero() -> None:
    assert regime_multiplier("INVALID", "STABLE") == 0.0
    assert regime_multiplier("INVALID", None) == 0.0


def test_drift_regime_is_exactly_zero() -> None:
    assert regime_multiplier("DRIFT", "CONVERGING") == 0.0
    assert regime_multiplier("DRIFT", None) == 0.0


def test_transition_regime_halves_signal() -> None:
    assert regime_multiplier("TRANSITION", "STABLE") == MULTIPLIER_TRANSITION
    assert regime_multiplier("TRANSITION", "CONVERGING") == MULTIPLIER_TRANSITION
    assert regime_multiplier("TRANSITION", None) == MULTIPLIER_TRANSITION


def test_critical_converging_passes_through() -> None:
    assert regime_multiplier("CRITICAL", "CONVERGING") == MULTIPLIER_CRITICAL
    assert regime_multiplier("CRITICAL", "STABLE") == MULTIPLIER_CRITICAL


def test_critical_diverging_reduced_to_half() -> None:
    assert regime_multiplier("CRITICAL", "DIVERGING") == MULTIPLIER_TRANSITION
    assert regime_multiplier("CRITICAL", None) == MULTIPLIER_TRANSITION


def test_unknown_regime_is_fail_closed() -> None:
    assert regime_multiplier("BOGUS", "STABLE") == 0.0


def test_apply_on_ou_yields_nonzero_multiplier() -> None:
    price = _ou(seed=1)
    filtered, obs = apply_regime_filter(raw_signal=1.0, price_window=price)
    mult = float(obs["regime_multiplier"])  # type: ignore[arg-type]
    assert obs["regime"] in {"CRITICAL", "TRANSITION"}
    assert mult >= MULTIPLIER_TRANSITION
    assert filtered == pytest.approx(mult)  # raw == 1.0


def test_apply_on_gbm_drifts_to_zero() -> None:
    price = _gbm(seed=2)
    filtered, obs = apply_regime_filter(raw_signal=1.0, price_window=price)
    mult = float(obs["regime_multiplier"])  # type: ignore[arg-type]
    assert obs["regime"] in {"INVALID", "DRIFT"}
    assert mult == 0.0
    assert filtered == 0.0


def test_apply_preserves_raw_sign_on_critical() -> None:
    price = _ou(seed=3)
    pos, _ = apply_regime_filter(raw_signal=+2.0, price_window=price)
    neg, _ = apply_regime_filter(raw_signal=-2.0, price_window=price)
    assert pos >= 0.0 and neg <= 0.0
    assert pos == pytest.approx(-neg)


def test_apply_raises_on_nan_input() -> None:
    price = _ou(seed=4).copy()
    price[500] = np.nan
    with pytest.raises(ValueError):
        apply_regime_filter(raw_signal=1.0, price_window=price)


def test_multiplier_constants_are_bounded() -> None:
    for mult in (
        MULTIPLIER_CRITICAL,
        MULTIPLIER_TRANSITION,
        MULTIPLIER_DRIFT,
        MULTIPLIER_INVALID,
    ):
        assert 0.0 <= mult <= 1.0


def test_invalid_multiplier_is_exactly_zero_constant() -> None:
    assert MULTIPLIER_INVALID == 0.0
    assert MULTIPLIER_DRIFT == 0.0
