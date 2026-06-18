# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""T2b — Stuart-Landau ES Proximity as Crisis Early-Warning Lead Indicator.

Stuart-Landau ensemble extension of explosive synchronization detection
(parallel to T2 in ``core.physics.explosive_sync``). Where T2 uses pure
phase Kuramoto, T2b retains amplitude dynamics:

    dz_i/dt = (μ_i + i·ω_i) z_i − |z_i|² z_i + (K/N) Σ_j (z_j − z_i)

with z_i = A_i · exp(i·θ_i). Amplitude dynamics matter near the
bifurcation — they are the substrate the Lee et al. PNAS 2025 framework
uses to characterise proximity to first-order (explosive) transition.

Detection protocol mirrors T2: forward + backward K-sweep on the fitted
ensemble, hysteresis area on R(K) integrated over K-range. The leading
claim (T2b) is that on cross-asset price data, rolling ES proximity
peaks PRECEDE R(t) peaks by τ ≥ 1 bar.

Invariants
----------
INV-SL1   amplitude ≥ 0 (universal)
INV-SL2   es_proximity ∈ [0, 1] (universal)
INV-T2b   ES_peak(t) precedes R_peak(t+τ) with τ ≥ 1 (qualitative;
          falsifiable hypothesis tested in benchmarks/rolling_es_proximity_oos.py)

References
----------
Lee, U. et al. (2025). "Proximity to explosive synchronization
    determines network collapse and recovery trajectories in neural
    and economic crises." PNAS 122(44). DOI: 10.1073/pnas.2505434122
Stuart, J. T. (1960); Landau, L. D. (1944).
Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence.
Gómez-Gardeñes et al. PRL 106, 128701 (2011).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert

_DEFAULT_K_LOW: Final[float] = 0.0
_DEFAULT_K_HIGH: Final[float] = 4.0
_DEFAULT_K_STEPS: Final[int] = 16
_DEFAULT_INT_STEPS: Final[int] = 200
_DEFAULT_DT: Final[float] = 0.05
_AMPLITUDE_FLOOR: Final[float] = 0.0  # INV-SL1 hard bound
# Numerical-stability clamp on the estimated linear growth rate μ. This is a
# bounds: (not physics) clamp — μ=0 stays the genuine Hopf threshold inside the
# band; the band only caps the explicit-Euler Stuart-Landau fixed point at
# |z|*=√μ ≤ 1 (disk scale of the seeded ensemble) to forbid blow-up on
# pathological inputs. Realistic per-bar log-amplitude rates are O(1e-2..1e-1).
_MU_GROWTH_CLAMP: Final[float] = 1.0  # bounds: Euler-stable, |z|*≤1

__all__ = [
    "StuartLandauResult",
    "fit_stuart_landau",
    "rolling_es_proximity",
    "crisis_signal_sl",
]


@dataclass(frozen=True, slots=True)
class StuartLandauResult:
    """Snapshot result of a Stuart-Landau fit + ES proximity sweep.

    Attributes
    ----------
    amplitude
        Per-asset complex amplitude |z_i| at the last bar. Shape (N,).
        Bound: A_i ≥ 0 (INV-SL1).
    phase
        Per-asset phase arg(z_i) at the last bar in radians. Shape (N,).
    order_parameter
        R = |⟨e^{iθ}⟩| over assets at the last bar. Bound: R ∈ [0, 1].
    hysteresis_area
        ∫|R_forward(K) − R_backward(K)| dK / K_range. Non-negative.
        Larger area ⇒ system has wider memory of prior coupling state ⇒
        first-order (explosive) transition more likely.
    es_proximity
        clip(hysteresis_area, 0, 1). Reported as the canonical [0,1]
        proximity scalar (INV-SL2).
    leads_r_peak
        Default ``False`` at the single-snapshot fit. Set to ``True``
        only by higher-level rolling/OOS analysers when an ES peak has
        been observed to precede the most recent R peak by τ ≥ 1 bar.
    """

    amplitude: NDArray[np.float64]
    phase: NDArray[np.float64]
    order_parameter: float
    hysteresis_area: float
    es_proximity: float
    leads_r_peak: bool

    def __post_init__(self) -> None:
        if np.any(self.amplitude < _AMPLITUDE_FLOOR):
            raise ValueError(
                f"INV-SL1 VIOLATED: amplitude<{_AMPLITUDE_FLOOR} in fitted "
                f"Stuart-Landau ensemble. min={float(self.amplitude.min()):.6e}. "
                f"Source: stuart_landau_es.fit_stuart_landau. "
                f"Reference: Lee et al. PNAS 2025 (DOI 10.1073/pnas.2505434122)."
            )
        if not (0.0 <= self.es_proximity <= 1.0):
            raise ValueError(
                f"INV-SL2 VIOLATED: es_proximity={self.es_proximity:.6e} "
                f"outside [0, 1]. hysteresis_area={self.hysteresis_area:.6e}. "
                f"Source: stuart_landau_es.fit_stuart_landau. "
                f"Reference: Lee et al. PNAS 2025."
            )


def _validate_prices(prices: NDArray[np.float64], min_T: int = 8) -> None:
    """Common contract checks shared by fit_stuart_landau and rolling."""
    if prices.ndim != 2:
        raise ValueError(
            f"INV-SL contract VIOLATED: prices shape (T,N) required, "
            f"got ndim={prices.ndim}, shape={prices.shape}. "
            f"Source: stuart_landau_es._validate_prices."
        )
    if prices.shape[0] < min_T:
        raise ValueError(
            f"INV-SL contract VIOLATED: T≥{min_T} required for "
            f"Hilbert+sweep, got T={prices.shape[0]}, N={prices.shape[1]}. "
            f"Source: stuart_landau_es._validate_prices."
        )
    if prices.shape[1] < 2:
        raise ValueError(
            f"INV-SL contract VIOLATED: N≥2 oscillators required, "
            f"got N={prices.shape[1]}, T={prices.shape[0]}. "
            f"Source: stuart_landau_es._validate_prices."
        )
    if not np.all(np.isfinite(prices)):
        raise ValueError(
            "INV-SL contract VIOLATED: prices must be finite (no NaN/Inf). "
            "Source: stuart_landau_es._validate_prices. "
            "Use ffill or drop incomplete bars upstream."
        )
    if np.any(prices <= 0.0):
        raise ValueError(
            "INV-SL contract VIOLATED: prices must be strictly positive "
            "(log-returns require it). "
            "Source: stuart_landau_es._validate_prices."
        )


def _estimate_growth_rate(
    amplitude_envelope: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Per-asset Stuart-Landau linear growth rate μ from a Hilbert amplitude envelope.

    Derivation
    ----------
    The Stuart-Landau amplitude obeys the radial ODE

        d|z|/dt = μ·|z| − |z|³.

    Away from saturation (|z|² ≪ |μ|) the cubic term is negligible and the
    envelope grows/decays exponentially: |z|(t) ≈ |z|₀·e^{μ·t}, i.e.

        d(ln|z|)/dt = μ.

    So μ is exactly the linear growth/decay rate of the LOG amplitude envelope.
    We estimate it as the deterministic ordinary-least-squares slope of
    ln A(t) regressed on bar index t (units: rate per bar). This makes μ a
    genuine rate with μ < 0 ⟺ decaying (stable focus), μ > 0 ⟺ growing
    (limit cycle), and μ = 0 the true Hopf threshold — replacing the prior
    fabricated ``tanh(variance/var_max − 0.5)`` aesthetic map, which had no
    derivation linking return variance to a growth rate and placed the
    bifurcation arbitrarily at "half the max variance".

    The OLS slope is closed-form (no iterative fit), hence fully
    deterministic: μ_i = Σ_t (t−t̄)(lnA_i,t − lnĀ_i) / Σ_t (t−t̄)².

    Reference: Stuart (1960); Landau (1944); Kuramoto (1984) §A.
    """
    n_t = amplitude_envelope.shape[0]
    t = np.arange(n_t, dtype=np.float64)
    t_centred = t - t.mean()
    t_var = float((t_centred * t_centred).sum())
    if t_var <= 0.0:
        raise ValueError(
            "INV-SL contract VIOLATED: growth-rate regression needs T≥3 "
            f"envelope points with nonzero time variance, got n_t={n_t}. "
            "Source: stuart_landau_es._estimate_growth_rate."
        )
    # ln of a strictly positive floored envelope; floor keeps the regression
    # finite if the analytic amplitude touches zero on a bar.
    # bounds: 1e-12 floor only guards log(0); does not bias the slope.
    log_amp = np.log(np.maximum(amplitude_envelope, 1e-12))
    log_amp_centred = log_amp - log_amp.mean(axis=0)
    mu = (t_centred[:, None] * log_amp_centred).sum(axis=0) / t_var
    mu_clamped = np.clip(  # bounds: Euler-stable ±1 (see _MU_GROWTH_CLAMP); μ=0 stays Hopf
        mu, -_MU_GROWTH_CLAMP, _MU_GROWTH_CLAMP
    )
    return mu_clamped.astype(np.float64)


def _extract_amplitude_phase_omega(
    prices: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Hilbert analytic signal of demeaned log-returns → amplitude, phase, ω, μ.

    Returns last-bar amplitude / phase, per-asset mean instantaneous frequency,
    and the per-asset linear growth rate μ estimated as the OLS slope of the
    log analytic-amplitude envelope (see ``_estimate_growth_rate``).
    """
    log_returns = np.diff(np.log(prices), axis=0)
    centred = log_returns - log_returns.mean(axis=0)
    z = hilbert(centred, axis=0)
    amplitude_envelope = np.abs(z).astype(np.float64)
    amplitude_last = amplitude_envelope[-1, :]
    phase_last = np.angle(z[-1, :]).astype(np.float64)
    phase_unwrapped = np.unwrap(np.angle(z), axis=0)
    omega = np.diff(phase_unwrapped, axis=0).mean(axis=0).astype(np.float64)
    mu_signed = _estimate_growth_rate(amplitude_envelope)
    return amplitude_last, phase_last, omega, mu_signed


def _stuart_landau_step(
    z: NDArray[np.complex128],
    mu: NDArray[np.float64],
    omega: NDArray[np.float64],
    K: float,
    dt: float,
) -> NDArray[np.complex128]:
    """Single explicit Euler step of the Stuart-Landau ensemble RHS."""
    n = z.shape[0]
    coupling = (K / n) * (z.sum() - n * z)
    self_term = (mu + 1j * omega) * z - (np.abs(z) ** 2) * z
    out: NDArray[np.complex128] = (z + dt * (self_term + coupling)).astype(
        np.complex128, copy=False
    )
    return out


def _r_at_K(
    mu: NDArray[np.float64],
    omega: NDArray[np.float64],
    K: float,
    z0: NDArray[np.complex128],
    steps: int,
    dt: float,
) -> tuple[float, NDArray[np.complex128]]:
    """Integrate ensemble at fixed K, return (R_final, z_final)."""
    z = z0.copy()
    for _ in range(steps):
        z = _stuart_landau_step(z, mu, omega, K, dt)
    R = float(np.abs(np.mean(np.exp(1j * np.angle(z)))))
    return R, z


def _hysteresis_sweep(
    mu: NDArray[np.float64],
    omega: NDArray[np.float64],
    *,
    K_low: float,
    K_high: float,
    K_steps: int,
    int_steps: int,
    dt: float,
    seed: int,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Forward + backward K-sweep; return (area_normalized, R_fwd, R_bwd)."""
    rng = np.random.default_rng(seed)
    n = mu.shape[0]
    K_grid = np.linspace(K_low, K_high, K_steps)
    z: NDArray[np.complex128] = np.empty(n, dtype=np.complex128)
    z.real = rng.standard_normal(n) * 0.1
    z.imag = rng.standard_normal(n) * 0.1

    R_fwd = np.zeros(K_steps, dtype=np.float64)
    for i, K in enumerate(K_grid):
        R_fwd[i], z = _r_at_K(mu, omega, float(K), z, int_steps, dt)

    R_bwd = np.zeros(K_steps, dtype=np.float64)
    for i, K in enumerate(K_grid[::-1]):
        R_bwd[K_steps - 1 - i], z = _r_at_K(mu, omega, float(K), z, int_steps, dt)

    K_range = max(K_high - K_low, 1e-12)
    area = float(np.trapezoid(np.abs(R_fwd - R_bwd), K_grid)) / K_range
    return area, R_fwd, R_bwd


def fit_stuart_landau(
    prices: NDArray[np.float64],
    *,
    K_low: float = _DEFAULT_K_LOW,
    K_high: float = _DEFAULT_K_HIGH,
    K_steps: int = _DEFAULT_K_STEPS,
    int_steps: int = _DEFAULT_INT_STEPS,
    dt: float = _DEFAULT_DT,
    seed: int = 42,
) -> StuartLandauResult:
    """Fit Stuart-Landau ensemble to a price-panel snapshot, sweep ES proximity.

    Steps:
        1. log-returns → Hilbert analytic signal → (A, θ, ω) per asset
        2. μ_i = OLS slope of ln(amplitude envelope) vs bar index = the
           Stuart-Landau linear growth rate (μ<0 stable focus, μ>0 limit
           cycle, μ=0 Hopf); clamped to ±1 for Euler stability
        3. K-sweep hysteresis on the fitted ensemble (forward + backward)
        4. ES proximity = clip(area / K_range, 0, 1)

    Parameters
    ----------
    prices
        Shape ``(T, N)``, strictly positive, finite. T ≥ 8, N ≥ 2.
    K_low, K_high, K_steps, int_steps, dt, seed
        K-sweep configuration.

    Raises
    ------
    ValueError
        On any contract violation (shape, finiteness, positivity).
    """
    _validate_prices(prices, min_T=8)
    if K_steps < 2:
        raise ValueError(
            f"INV-SL contract VIOLATED: K_steps≥2 required, got {K_steps}. "
            f"Source: stuart_landau_es.fit_stuart_landau."
        )
    if int_steps < 1:
        raise ValueError(
            f"INV-SL contract VIOLATED: int_steps≥1 required, got {int_steps}. "
            f"Source: stuart_landau_es.fit_stuart_landau."
        )
    if K_low >= K_high:
        raise ValueError(
            f"INV-SL contract VIOLATED: K_low<K_high required, "
            f"got K_low={K_low}, K_high={K_high}. "
            f"Source: stuart_landau_es.fit_stuart_landau."
        )

    amplitude, phase, omega, mu = _extract_amplitude_phase_omega(prices)
    R_now = float(np.abs(np.mean(np.exp(1j * phase))))
    area, _, _ = _hysteresis_sweep(
        mu,
        omega,
        K_low=K_low,
        K_high=K_high,
        K_steps=K_steps,
        int_steps=int_steps,
        dt=dt,
        seed=seed,
    )
    es_proximity = float(min(max(area, 0.0), 1.0))

    return StuartLandauResult(
        amplitude=np.abs(amplitude),
        phase=phase,
        order_parameter=R_now,
        hysteresis_area=area,
        es_proximity=es_proximity,
        leads_r_peak=False,
    )


def rolling_es_proximity(
    prices: NDArray[np.float64],
    window: int,
    *,
    step: int = 1,
    K_low: float = _DEFAULT_K_LOW,
    K_high: float = _DEFAULT_K_HIGH,
    K_steps: int = _DEFAULT_K_STEPS,
    int_steps: int = _DEFAULT_INT_STEPS,
    dt: float = _DEFAULT_DT,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Causal rolling ES proximity series, one fit per window endpoint.

    Output is shape ``(T,)``; indices ``[0, window-1)`` are NaN
    (insufficient lookback). Failed windows (rare contract violations
    on edge segments) are also NaN. Walks left→right with stride ``step``.
    """
    _validate_prices(prices, min_T=window)
    if window < 8:
        raise ValueError(
            f"INV-SL contract VIOLATED: window≥8 required, got {window}. "
            f"Source: stuart_landau_es.rolling_es_proximity."
        )
    if step < 1:
        raise ValueError(
            f"INV-SL contract VIOLATED: step≥1 required, got {step}. "
            f"Source: stuart_landau_es.rolling_es_proximity."
        )
    T = prices.shape[0]
    out = np.full(T, np.nan, dtype=np.float64)
    for t in range(window - 1, T, step):
        slab = prices[t - window + 1 : t + 1]
        try:
            res = fit_stuart_landau(
                slab,
                K_low=K_low,
                K_high=K_high,
                K_steps=K_steps,
                int_steps=int_steps,
                dt=dt,
                seed=seed,
            )
            out[t] = res.es_proximity
        except (ValueError, FloatingPointError):
            out[t] = np.nan
    return out


def crisis_signal_sl(
    prices: NDArray[np.float64],
    *,
    es_threshold: float = 0.3,
    K_low: float = _DEFAULT_K_LOW,
    K_high: float = _DEFAULT_K_HIGH,
    K_steps: int = _DEFAULT_K_STEPS,
    int_steps: int = _DEFAULT_INT_STEPS,
    dt: float = _DEFAULT_DT,
    seed: int = 42,
) -> dict[str, float | bool]:
    """Single-shot Stuart-Landau crisis signal, parallel API to T2 crisis_signal.

    Keys:
        es_proximity, hysteresis_area, order_parameter,
        amplitude_max, amplitude_min, is_explosive, leads_r_peak.
    """
    res = fit_stuart_landau(
        prices,
        K_low=K_low,
        K_high=K_high,
        K_steps=K_steps,
        int_steps=int_steps,
        dt=dt,
        seed=seed,
    )
    return {
        "es_proximity": res.es_proximity,
        "hysteresis_area": res.hysteresis_area,
        "order_parameter": res.order_parameter,
        "amplitude_max": float(res.amplitude.max()),
        "amplitude_min": float(res.amplitude.min()),
        "is_explosive": bool(res.es_proximity > es_threshold),
        "leads_r_peak": res.leads_r_peak,
    }
