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
import pytest
from numpy.typing import NDArray

from core.physics.stuart_landau_es import (
    StuartLandauResult,
    _analytic_signal,
    _estimate_growth_rate,
    _extract_amplitude_phase_omega,
    _hysteresis_sweep,
    _validate_z0,
    crisis_signal_sl,
    fit_stuart_landau,
    rolling_es_proximity,
)

pytestmark = pytest.mark.heavy_math


def _envelope_oscillator_prices(
    growth_rate: float,
    T: int = 96,
    N: int = 4,
    freq: float = 0.8,
    amp0: float = 0.01,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Synthetic price panel whose log-return envelope grows/decays at ``growth_rate``.

    Each asset's log-returns are an exponential-envelope sinusoid
    r_j(t) = amp0 · e^{growth_rate · t} · sin(freq·t + φ_j) + tiny noise, so the
    Hilbert amplitude envelope A_j(t) ∝ e^{growth_rate · t} and the Stuart-Landau
    growth rate μ_j must recover ``growth_rate`` in sign:
        growth_rate < 0 → decaying (stable focus, μ < 0)
        growth_rate > 0 → growing (limit cycle, μ > 0).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T - 1, dtype=np.float64)
    env = np.exp(growth_rate * t)
    log_rets = np.zeros((T - 1, N), dtype=np.float64)
    for j in range(N):
        log_rets[:, j] = (
            amp0 * env * np.sin(freq * t + j) + rng.standard_normal(T - 1) * 1e-5
        )
    prices: NDArray[np.float64] = np.empty((T, N), dtype=np.float64)
    prices[0, :] = 100.0
    prices[1:, :] = 100.0 * np.exp(np.cumsum(log_rets, axis=0))
    return prices


def test_mu_negative_for_stable_decaying_oscillator() -> None:
    """INV-SL1 substrate: decaying-envelope (stable focus) ⇒ growth rate μ < 0.

    Stuart-Landau μ is the linear growth rate (d ln|z|/dt). A stable focus has a
    decaying amplitude envelope, so the derived μ must be strictly negative —
    the genuine Hopf threshold is μ = 0, not the prior fabricated
    tanh(variance−0.5) map.
    """
    mus: list[float] = []
    for seed in range(6):
        prices = _envelope_oscillator_prices(growth_rate=-0.05, seed=seed)
        _, _, _, mu = _extract_amplitude_phase_omega(prices)
        mus.append(float(mu.max()))  # worst (least-negative) asset
    worst = max(mus)
    assert worst < 0.0, (
        f"INV-SL μ-growth-rate VIOLATED: stable decaying oscillator gave "
        f"max μ={worst:.6e} ≥ 0, expected μ<0 (decaying envelope = stable focus). "
        f"μ = OLS slope of ln amplitude envelope; Hopf threshold at μ=0. "
        f"Tested 6 seeds, growth_rate=-0.05, per-seed max μ={mus}."
    )


def test_mu_positive_for_growing_limit_cycle() -> None:
    """INV-SL1 substrate: growing-envelope (limit cycle) ⇒ growth rate μ > 0.

    A supercritical Hopf / limit cycle has a growing amplitude envelope, so the
    derived μ must be strictly positive, mirror-imaging the stable case about
    the μ=0 Hopf threshold.
    """
    mus: list[float] = []
    for seed in range(6):
        prices = _envelope_oscillator_prices(growth_rate=+0.05, seed=seed)
        _, _, _, mu = _extract_amplitude_phase_omega(prices)
        mus.append(float(mu.min()))  # worst (least-positive) asset
    worst = min(mus)
    assert worst > 0.0, (
        f"INV-SL μ-growth-rate VIOLATED: growing limit-cycle signal gave "
        f"min μ={worst:.6e} ≤ 0, expected μ>0 (growing envelope = limit cycle). "
        f"μ = OLS slope of ln amplitude envelope; Hopf threshold at μ=0. "
        f"Tested 6 seeds, growth_rate=+0.05, per-seed min μ={mus}."
    )


def test_mu_clamped_finite_and_bounded() -> None:
    """μ stays finite and within the ±1 Euler-stability clamp on all regimes.

    Guards that the growth-rate estimator never emits NaN/Inf and respects the
    numerical-stability bound feeding the Stuart-Landau RHS (|z|*=√μ ≤ 1), so
    INV-SL1/SL2 downstream cannot be corrupted by an unbounded μ.
    """
    bad: list[tuple[float, float]] = []
    n_t, n_assets = 8, 3
    t = np.arange(n_t, dtype=np.float64)
    # Directly build finite log-envelopes A=e^{rate·t}: short window keeps even
    # steep rates finite (exp(2·7)≈1.2e6), so the raw OLS slope == rate exactly.
    # This isolates the estimator+clamp from price-construction overflow.
    for raw_rate in (-2.0, -0.05, 0.0, 0.05, 2.0):
        envelope = np.tile(np.exp(raw_rate * t)[:, None], (1, n_assets))
        mu = _estimate_growth_rate(envelope)
        if not np.all(np.isfinite(mu)) or np.any(np.abs(mu) > 1.0 + 1e-12):
            bad.append((raw_rate, float(np.abs(mu).max())))
        # |rate|>1 must be clamped exactly to the ±1 band, sign preserved
        if abs(raw_rate) > 1.0 and not np.allclose(mu, np.sign(raw_rate) * 1.0):
            bad.append((raw_rate, float(np.abs(mu).max())))
    assert not bad, (
        f"INV-SL μ bound VIOLATED: μ non-finite, |μ|>1, or steep rate unclamped "
        f"in {len(bad)} cases. First: {bad[0] if bad else None}. "
        f"μ clamp = ±1 (Euler-stable, |z|*=√μ≤1). "
        f"Tested raw rates (-2,-0.05,0,0.05,2) × {n_assets} assets, n_t={n_t}."
    )


def test_z0_validation_rejects_shape_and_nonfinite_components() -> None:
    """Fitted-state validation must reject wrong shape and either non-finite axis."""
    good = np.array([1.0 + 0.0j, 0.5 + 0.25j], dtype=np.complex128)
    copied = _validate_z0(good, 2)
    assert copied.shape == (2,)
    assert copied.dtype == np.complex128
    assert copied is not good

    with pytest.raises(ValueError, match="z0 shape"):
        _validate_z0(good, 3)

    bad_real = np.array([np.nan + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
    with pytest.raises(ValueError, match="z0 must be finite"):
        _validate_z0(bad_real, 2)

    bad_imag = np.array([1.0 + np.inf * 1j, 1.0 + 0.0j], dtype=np.complex128)
    with pytest.raises(ValueError, match="z0 must be finite"):
        _validate_z0(bad_imag, 2)


def test_hysteresis_sweep_uses_validated_external_z0() -> None:
    """External fitted state must enter the sweep through the same z0 guard."""
    mu = np.array([0.1, 0.2], dtype=np.float64)
    omega = np.array([0.3, 0.4], dtype=np.float64)
    bad_z0 = np.array([1.0 + 0.0j, np.nan + 0.0j], dtype=np.complex128)
    with pytest.raises(ValueError, match="z0 must be finite"):
        _hysteresis_sweep(
            mu,
            omega,
            K_low=0.0,
            K_high=1.0,
            K_steps=3,
            int_steps=2,
            dt=0.05,
            seed=0,
            z0=bad_z0,
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
    prices: NDArray[np.float64] = (100.0 * np.exp(np.cumsum(rets, axis=0))).astype(
        np.float64
    )
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
    """Rolling Kuramoto order parameter via analytic phases."""
    T = prices.shape[0]
    out = np.full(T, np.nan, dtype=np.float64)
    log_prices = np.log(prices)
    for t in range(window, T):
        slab = np.diff(log_prices[t - window : t + 1], axis=0)
        centred = slab - slab.mean(axis=0)
        z = _analytic_signal(centred, axis=0)
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

        es_raw = rolling_es_proximity(
            prices,
            window=W,
            K_steps=12,
            int_steps=120,
            seed=seed,
        )
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
