# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Tests for T2b — Stuart-Landau ES Proximity (Lee et al. PNAS 2025).

Invariants tested
-----------------
INV-SL1   amplitude ≥ 0  (universal)
INV-SL2   es_proximity ∈ [0, 1]  (universal)
INV-T2b   rolling ES peak precedes R(t) peak by τ ≥ 1 bar  (qualitative)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert

from core.physics.stuart_landau_es import (
    StuartLandauResult,
    crisis_signal_sl,
    fit_stuart_landau,
    rolling_es_proximity,
)


def _synthetic_prices(
    T: int = 64,
    N: int = 5,
    seed: int = 0,
    regime: str = "quiet",
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    if regime == "quiet":
        rets = rng.standard_normal((T, N)) * 0.005
    elif regime == "crisis":
        common = np.cumsum(rng.standard_normal(T) * 0.01)
        common = common * np.linspace(0.5, 2.0, T)
        rets = (common[:, None] + 0.1 * rng.standard_normal((T, N))) * 0.01
    else:
        raise ValueError(f"unknown regime {regime!r}")
    prices: NDArray[np.float64] = (100.0 * np.exp(np.cumsum(rets, axis=0))).astype(np.float64)
    return prices


def test_amplitude_positive() -> None:
    """INV-SL1: amplitude ≥ 0 across many seeds (universal property)."""
    violations: list[tuple[int, float]] = []
    for seed in range(8):
        prices = _synthetic_prices(T=32, N=5, seed=seed)
        res = fit_stuart_landau(prices, K_steps=8, int_steps=80, seed=seed)
        amin = float(res.amplitude.min())
        if amin < 0.0:
            violations.append((seed, amin))
    assert isinstance(res, StuartLandauResult)
    assert res.amplitude.shape == (5,)
    assert not violations, (
        f"INV-SL1 VIOLATED: amplitude<0 in {len(violations)} of 8 seeds. "
        f"First: {violations[0] if violations else None}. "
        f"Stuart-Landau A=|z| must be non-negative by construction. "
        f"Tested over T=32, N=5, K_steps=8, int_steps=80."
    )


def test_es_proximity_bounded() -> None:
    """INV-SL2: es_proximity ∈ [0, 1] universally."""
    out_of_bounds: list[tuple[int, str, float]] = []
    for seed in range(8):
        for regime in ("quiet", "crisis"):
            prices = _synthetic_prices(T=32, N=5, seed=seed, regime=regime)
            res = fit_stuart_landau(prices, K_steps=8, int_steps=80, seed=seed)
            if not (0.0 <= res.es_proximity <= 1.0):
                out_of_bounds.append((seed, regime, res.es_proximity))
    assert not out_of_bounds, (
        f"INV-SL2 VIOLATED: es_proximity outside [0,1] in "
        f"{len(out_of_bounds)} cases of 16. "
        f"First violation: {out_of_bounds[0] if out_of_bounds else None}. "
        f"Tested over 8 seeds × 2 regimes (quiet, crisis). "
        f"K_steps=8, int_steps=80."
    )


def test_rolling_window_shape() -> None:
    """rolling_es_proximity output shape and NaN-prefix contract."""
    T, N, W = 48, 5, 16
    prices = _synthetic_prices(T=T, N=N, seed=0)
    out = rolling_es_proximity(prices, window=W, K_steps=6, int_steps=60)
    assert out.shape == (
        T,
    ), f"rolling shape mismatch: expected ({T},), got {out.shape}. window={W}."
    assert np.all(np.isnan(out[: W - 1])), (
        f"Pre-window indices [0,{W - 1}) must be NaN; "
        f"got finite at {np.where(np.isfinite(out[: W - 1]))[0]}."
    )
    valid = out[W - 1 :]
    assert np.any(
        np.isfinite(valid)
    ), "At least one rolling ES proximity value must be finite in [W-1, T)."


def test_crisis_signal_api() -> None:
    """API contract on crisis_signal_sl dict shape and types."""
    prices = _synthetic_prices(T=32, N=5, seed=0)
    sig = crisis_signal_sl(prices, K_steps=6, int_steps=60)
    required = {
        "es_proximity",
        "hysteresis_area",
        "order_parameter",
        "amplitude_max",
        "amplitude_min",
        "is_explosive",
        "leads_r_peak",
    }
    missing = required - sig.keys()
    assert not missing, f"crisis_signal_sl missing keys: {missing}"
    assert isinstance(sig["es_proximity"], float)
    assert isinstance(sig["is_explosive"], bool)
    assert isinstance(sig["leads_r_peak"], bool)
    es_val = sig["es_proximity"]
    assert isinstance(es_val, float) and 0.0 <= es_val <= 1.0
    op_val = sig["order_parameter"]
    assert isinstance(op_val, float) and 0.0 <= op_val <= 1.0
    amax = sig["amplitude_max"]
    amin = sig["amplitude_min"]
    assert isinstance(amax, float) and isinstance(amin, float)
    assert amax >= amin >= 0.0


def _rolling_R(prices: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Rolling Kuramoto order parameter via Hilbert phases."""
    T = prices.shape[0]
    out = np.full(T, np.nan, dtype=np.float64)
    log_prices = np.log(prices)
    for t in range(window, T):
        slab = np.diff(log_prices[t - window : t + 1], axis=0)
        centred = slab - slab.mean(axis=0)
        z = hilbert(centred, axis=0)
        phase_last = np.angle(z[-1, :])
        out[t] = float(np.abs(np.mean(np.exp(1j * phase_last))))
    return out


def _smooth_box(x: NDArray[np.float64], width: int = 5) -> NDArray[np.float64]:
    out = np.full_like(x, np.nan)
    half = width // 2
    for i in range(len(x)):
        lo = max(0, i - half)
        hi = min(len(x), i + half + 1)
        seg = x[lo:hi]
        if np.any(np.isfinite(seg)):
            out[i] = float(np.nanmean(seg))
    return out


def test_leads_r_peak_on_synthetic_crisis() -> None:
    """INV-T2b: median ES peak precedes R peak across crisis seeds (smoothed)."""
    taus: list[int] = []
    T, N, W = 120, 6, 24
    for seed in range(8):
        rng = np.random.default_rng(seed)
        # construction: noise → ramp → peak → decay common factor
        factor = np.zeros(T, dtype=np.float64)
        factor[30:75] = np.linspace(0.0, 0.05, 45)
        factor[75:95] = 0.05
        factor[95:] = 0.05 * np.exp(-(np.arange(T - 95)) * 0.15)
        rets = factor[:, None] + rng.standard_normal((T, N)) * 0.003
        prices = 100.0 * np.exp(np.cumsum(rets, axis=0))

        es_raw = rolling_es_proximity(prices, window=W, K_steps=12, int_steps=120, seed=seed)
        r_raw = _rolling_R(prices, window=W)
        if np.all(np.isnan(es_raw[W:])) or np.all(np.isnan(r_raw[W:])):
            continue
        es_smooth = _smooth_box(es_raw, width=5)
        r_smooth = _smooth_box(r_raw, width=5)
        es_peak = int(np.nanargmax(es_smooth))
        r_peak = int(np.nanargmax(r_smooth))
        taus.append(r_peak - es_peak)

    assert len(taus) >= 6, (
        f"INV-T2b test sample too small: {len(taus)} valid seeds of 8. "
        f"Synthetic crisis must yield finite rolling series; "
        f"check window/K_steps."
    )
    median_tau = float(np.median(taus))
    leads_count = sum(1 for t in taus if t >= 1)
    assert median_tau >= 1.0, (
        f"INV-T2b VIOLATED on synthetic crisis: median(τ)={median_tau} < 1. "
        f"Per-seed taus: {taus}. leads_count={leads_count}/{len(taus)}. "
        f"Construction: factor ramp t∈[30,75), peak [75,95), decay 95+. "
        f"T={T}, N={N}, W={W}, smoothing=box-5."
    )
    assert leads_count >= max(4, len(taus) // 2), (
        f"INV-T2b weak: only {leads_count}/{len(taus)} seeds had ES leading R. "
        f"Per-seed taus: {taus}."
    )
