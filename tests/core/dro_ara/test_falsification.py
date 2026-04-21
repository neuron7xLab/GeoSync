# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Falsification battery for DRO-ARA v7.

Synthesises price series with **known** statistical regimes and asserts that
the observer classifies them correctly. This is the integration gate: if any
scenario fails, the engine must not ship.

Post-PR #345 convention: ADF runs on log-returns (not raw prices), matching
the DFA transform. INVALID therefore encodes a *true* unit root in returns,
not in levels — a non-trivial condition.

Scenarios (deterministic, seeded):

* OU mean-reverting prices        → stationary returns, CRITICAL/TRANSITION
* Pure random walk (GBM no drift) → stationary returns (i.i.d.), never LONG
* GBM with positive drift         → stationary returns (μ+σZ), never LONG
* White noise prices              → stationary returns, TRANSITION (H ≈ 0.5)
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


def test_random_walk_returns_are_stationary_no_long() -> None:
    """Random walk: prices I(1), log-returns i.i.d. → ADF stationary post-RFC.

    Hurst estimate for a pure RW is ≈ 0.5 ± finite-sample noise; regime
    therefore lands in {CRITICAL, TRANSITION, DRIFT}. The invariant that
    *must* hold (INV-DRO4): RW has no true mean-reversion edge, so signal
    must never be LONG — regardless of which specific non-INVALID regime
    the finite sample produces.
    """
    price = _random_walk(SEED, N_SAMPLES)
    out = geosync_observe(price, window=WINDOW, step=STEP)
    assert out["stationary"] is True, f"RW returns must be stationary post-RFC: {out}"
    assert out["regime"] != Regime.INVALID.value, f"RW should not be INVALID: {out}"
    assert out["signal"] != "LONG", f"INV-DRO4: RW must never emit LONG: {out}"


def test_gbm_with_drift_returns_are_stationary_no_long() -> None:
    """GBM with drift: prices I(1), log-returns ~ N(μ, σ²) → ADF stationary.

    Post-RFC (PR #345) the stationarity test targets returns. GBM returns
    have no unit root — they are i.i.d. Gaussian — so ``stationary=True``.
    Trend at the price level surfaces in the ARA trend path as
    DRIFT/DIVERGING, which blocks LONG via INV-DRO4. The true falsification
    invariant is the signal gate, not the stationarity classification.
    """
    price = _gbm_drift(SEED, N_SAMPLES, mu=0.002, sigma=0.01)
    out = geosync_observe(price, window=WINDOW, step=STEP)
    assert out["stationary"] is True, f"GBM returns must be stationary post-RFC: {out}"
    assert out["regime"] != Regime.INVALID.value, f"GBM post-RFC must not be INVALID: {out}"
    assert out["signal"] != "LONG", f"INV-DRO4: GBM drift must never emit LONG: {out}"


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
