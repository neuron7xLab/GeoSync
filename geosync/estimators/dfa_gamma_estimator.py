# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Detrended Fluctuation Analysis γ estimator — robust on non-stationary data.

Algorithm (Peng et al. 1994):
  1. Profile: y(k) = Σ(x_i - mean)
  2. Divide into windows of size s
  3. Detrend each window (linear fit)
  4. F(s) = sqrt(mean residual variance)
  5. H = slope of log F(s) vs log s
  6. γ = 2H + 1 (DERIVED, never assigned)

Cross-validation: Daubechies db4 wavelet detail coefficients
yield independent H estimate via variance scaling.

Advantage over Welch PSD: no stationarity assumption.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DFAEstimate:
    """DFA gamma estimate with wavelet cross-validation.

    INV: abs(gamma - (2*hurst_exponent + 1)) < 1e-10
    """

    hurst_exponent: float
    gamma: float  # = 2*hurst_exponent + 1, NEVER assigned directly
    dfa_fluctuations: tuple[float, ...]
    scale_range: tuple[int, int]
    r_squared: float
    wavelet_confirmed: bool
    n_samples: int
    computation_time_ms: float

    def __post_init__(self) -> None:
        expected = 2.0 * self.hurst_exponent + 1.0
        if abs(self.gamma - expected) > 1e-10:
            raise ValueError(
                f"γ={self.gamma} ≠ 2H+1={expected}. gamma must be DERIVED from hurst_exponent."
            )


def _invalid_estimate(n: int, elapsed_ms: float) -> DFAEstimate:
    return DFAEstimate(
        hurst_exponent=0.0,
        gamma=1.0,  # 2*0.0+1
        dfa_fluctuations=(),
        scale_range=(0, 0),
        r_squared=0.0,
        wavelet_confirmed=False,
        n_samples=n,
        computation_time_ms=elapsed_ms,
    )


class DFAGammaEstimator:
    """Detrended Fluctuation Analysis for non-stationary series.

    Parameters
    ----------
    n_scales
        Number of log-spaced window sizes.
    min_quality
        Minimum R² for valid estimate (raises ValueError if below).
    """

    def __init__(
        self,
        n_scales: int = 20,
        min_quality: float = 0.95,
    ) -> None:
        self.n_scales = n_scales
        self.min_quality = min_quality

    def compute(self, data: np.ndarray) -> DFAEstimate:
        """Compute γ via DFA. DERIVED from H, never assigned."""
        t0 = time.perf_counter()
        x = np.asarray(data, dtype=np.float64).ravel()
        n = x.size

        if n < 128 or not np.all(np.isfinite(x)):
            elapsed = (time.perf_counter() - t0) * 1000.0
            return _invalid_estimate(n, elapsed)

        s_min = max(4, n // 50)
        s_max = n // 4
        if s_max <= s_min:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return _invalid_estimate(n, elapsed)

        scales = np.unique(np.logspace(np.log10(s_min), np.log10(s_max), self.n_scales).astype(int))
        scales = scales[scales >= 4]

        H, r_sq, fluct, valid_scales = _dfa_fit(x, scales)
        elapsed = (time.perf_counter() - t0) * 1000.0

        if not math.isfinite(H) or len(fluct) < 4:
            return _invalid_estimate(n, elapsed)

        # Wavelet cross-validation (db4)
        wavelet_confirmed = _wavelet_check(x, H)

        if r_sq < self.min_quality:
            raise ValueError(
                f"DFA r_squared={r_sq:.4f} < {self.min_quality}. "
                "Unreliable scaling fit — increase data length or check stationarity."
            )

        return DFAEstimate(
            hurst_exponent=round(H, 6),
            gamma=round(2.0 * round(H, 6) + 1.0, 6),
            dfa_fluctuations=tuple(float(f) for f in fluct),
            scale_range=(int(valid_scales[0]), int(valid_scales[-1])),
            r_squared=round(r_sq, 6),
            wavelet_confirmed=wavelet_confirmed,
            n_samples=n,
            computation_time_ms=round(elapsed, 3),
        )


def _dfa_fit(x: np.ndarray, scales: np.ndarray) -> tuple[float, float, list[float], list[int]]:
    """Core DFA: returns (H, R², fluctuations, valid_scales)."""
    y = np.cumsum(x - np.mean(x))
    n = len(y)
    fluct: list[float] = []
    valid: list[int] = []

    for s in scales:
        s_int = int(s)
        n_seg = n // s_int
        if n_seg < 1:
            continue
        var_sum = 0.0
        for v in range(n_seg):
            seg = y[v * s_int : (v + 1) * s_int]
            t = np.arange(s_int, dtype=np.float64)
            coeffs = np.polyfit(t, seg, 1)
            trend = coeffs[0] * t + coeffs[1]
            var_sum += float(np.mean((seg - trend) ** 2))
        f_s = np.sqrt(var_sum / n_seg)
        if f_s > 0 and math.isfinite(f_s):
            fluct.append(float(f_s))
            valid.append(s_int)

    if len(fluct) < 4:
        return 0.0, 0.0, [], []

    log_s = np.log(np.array(valid, dtype=np.float64))
    log_f = np.log(np.array(fluct, dtype=np.float64))

    coeffs = np.polyfit(log_s, log_f, 1)
    H = float(coeffs[0])

    pred = coeffs[0] * log_s + coeffs[1]
    ss_res = float(np.sum((log_f - pred) ** 2))
    ss_tot = float(np.sum((log_f - np.mean(log_f)) ** 2))
    r_sq = 1.0 - ss_res / (ss_tot + 1e-12)

    return H, max(0.0, r_sq), fluct, valid


def _wavelet_check(x: np.ndarray, dfa_H: float, tol: float = 0.15) -> bool:
    """Wavelet-based H estimate (db4) for cross-validation.

    H_wavelet derived from variance scaling of detail coefficients:
      Var(d_j) ∝ 2^{j(2H+1)} → H = (slope - 1) / 2
    """
    try:
        import pywt  # noqa: PLC0415
    except ImportError:
        return False

    coeffs = pywt.wavedec(x, "db4", level=min(8, int(np.log2(len(x))) - 2))
    if len(coeffs) < 4:
        return False

    detail_vars: list[float] = []
    levels: list[float] = []
    for j, d in enumerate(coeffs[1:], start=1):
        v = float(np.var(d))
        if v > 0:
            detail_vars.append(np.log2(v))
            levels.append(float(j))

    if len(detail_vars) < 3:
        return False

    lv = np.array(levels)
    ld = np.array(detail_vars)
    slope = float(np.polyfit(lv, ld, 1)[0])
    H_wavelet = (slope - 1.0) / 2.0

    return abs(H_wavelet - dfa_H) < tol
