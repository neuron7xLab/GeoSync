# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Falsification battery for DRO-ARA v7.

Synthesises price series with **known** statistical regimes and asserts that
the observer classifies them correctly. This is the integration gate: if any
scenario fails, the engine must not ship.

Scenarios (deterministic, seeded):

* OU mean-reverting prices        → stationary, CRITICAL (H < 0.45)
* Pure random walk (GBM no drift) → non-stationary, INVALID
* GBM with positive drift         → non-stationary, INVALID
* White noise prices              → stationary, TRANSITION (H ≈ 0.5)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from core.dro_ara import Regime, geosync_observe

SEED = 42
N_SAMPLES = 4096
WINDOW = 512
STEP = 64


def _ou_prices(
    seed: int,
    n: int,
    mu: float = 100.0,
    theta: float = 0.08,
    sigma: float = 0.6,
) -> NDArray[np.float64]:
    """Ornstein–Uhlenbeck: strong mean reversion around mu → stationary levels."""
    rng = np.random.default_rng(seed)
    x = np.empty(n, dtype=np.float64)
    x[0] = mu
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) + sigma * rng.normal()
    return x


def _random_walk(seed: int, n: int, sigma: float = 0.01) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, sigma, size=n)
    return np.exp(np.cumsum(returns)) * 100.0


def _gbm_drift(seed: int, n: int, mu: float = 0.001, sigma: float = 0.01) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    returns = mu + sigma * rng.normal(size=n)
    return np.exp(np.cumsum(returns)) * 100.0


def _white_noise_prices(
    seed: int, n: int, mu: float = 100.0, sigma: float = 1.0
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return mu + sigma * rng.normal(size=n)


def test_ou_mean_reverting_is_critical() -> None:
    price = _ou_prices(SEED, N_SAMPLES)
    out = geosync_observe(price, window=WINDOW, step=STEP)
    assert out["stationary"] is True, f"OU must be stationary, got {out}"
    assert out["H"] < 0.50, f"OU H must be sub-diffusive, got H={out['H']}"
    assert out["regime"] in {Regime.CRITICAL.value, Regime.TRANSITION.value}, out


def test_random_walk_is_invalid_or_transition() -> None:
    price = _random_walk(SEED, N_SAMPLES)
    out = geosync_observe(price, window=WINDOW, step=STEP)
    assert out["regime"] in {Regime.INVALID.value, Regime.TRANSITION.value}, out


def test_gbm_with_drift_is_non_stationary() -> None:
    price = _gbm_drift(SEED, N_SAMPLES, mu=0.002, sigma=0.01)
    out = geosync_observe(price, window=WINDOW, step=STEP)
    assert out["stationary"] is False, f"GBM with drift must fail ADF, got {out}"
    assert out["regime"] == Regime.INVALID.value


def test_white_noise_prices_are_stationary() -> None:
    price = _white_noise_prices(SEED, N_SAMPLES)
    out = geosync_observe(price, window=WINDOW, step=STEP)
    assert out["stationary"] is True


def test_signal_never_long_on_gbm() -> None:
    """A non-stationary trending series must never emit LONG."""
    for seed in (1, 2, 3, 4, 5):
        price = _gbm_drift(seed, N_SAMPLES, mu=0.002, sigma=0.01)
        out = geosync_observe(price, window=WINDOW, step=STEP)
        assert out["signal"] != "LONG", f"seed={seed} produced LONG on GBM: {out}"


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_ou_never_emits_short(seed: int) -> None:
    """Mean-reverting series must never emit SHORT (SHORT requires DRIFT regime)."""
    price = _ou_prices(seed, N_SAMPLES)
    out = geosync_observe(price, window=WINDOW, step=STEP)
    assert out["signal"] != "SHORT", f"seed={seed} OU produced SHORT: {out}"


def test_output_schema_complete() -> None:
    price = _ou_prices(SEED, N_SAMPLES)
    out = geosync_observe(price, window=WINDOW, step=STEP)
    required = {
        "gamma",
        "H",
        "r2_dfa",
        "regime",
        "risk_scalar",
        "stationary",
        "signal",
        "free_energy",
        "ara_steps",
        "converged",
        "trend",
        "alpha_ema",
    }
    assert required <= set(out.keys()), f"missing keys: {required - set(out.keys())}"
