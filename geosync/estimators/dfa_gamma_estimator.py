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

Advantage over Welch PSD: no stationarity assumption.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DFAEstimate:
    """DFA gamma estimate with regime shift detection."""

    value: float  # γ = 2H+1, never 1.0
    hurst: float  # H = (γ-1)/2
    quality: float  # R² of log-log fit [0,1]
    ci_low: float  # 95% bootstrap lower
    ci_high: float  # 95% bootstrap upper
    is_valid: bool  # quality >= 0.85 AND finite
    is_stationary: bool  # ADF p < 0.05
    local_H: float  # H on last N//4 window
    regime_shift: bool  # |H_global - H_local| > 0.2


class DFAGammaEstimator:
    """Detrended Fluctuation Analysis for non-stationary series.

    Parameters
    ----------
    n_scales
        Number of log-spaced window sizes.
    min_quality
        Minimum R² for valid estimate.
    bootstrap_n
        Bootstrap resamples for CI.
    """

    def __init__(
        self,
        n_scales: int = 20,
        min_quality: float = 0.85,
        bootstrap_n: int = 200,
    ) -> None:
        self.n_scales = n_scales
        self.min_quality = min_quality
        self.bootstrap_n = bootstrap_n

    def compute(self, data: np.ndarray) -> DFAEstimate:
        """Compute γ via DFA. DERIVED, never assigned."""
        x = np.asarray(data, dtype=np.float64)
        if x.ndim != 1 or x.size < 128 or not np.all(np.isfinite(x)):
            return DFAEstimate(0.0, 0.0, 0.0, 0.0, 0.0, False, False, 0.0, False)

        n = x.size
        s_min = max(4, n // 50)
        s_max = n // 4
        if s_max <= s_min:
            return DFAEstimate(0.0, 0.0, 0.0, 0.0, 0.0, False, False, 0.0, False)

        scales = np.unique(np.logspace(np.log10(s_min), np.log10(s_max), self.n_scales).astype(int))
        scales = scales[scales >= 4]

        # Global H
        H_global, r_sq, fluct, valid_scales = self._dfa_fit(x, scales)
        if not math.isfinite(H_global):
            return DFAEstimate(0.0, 0.0, 0.0, 0.0, 0.0, False, False, 0.0, False)

        # Local H (last N//4)
        local_window = x[-(n // 4) :]
        if len(local_window) >= 64:
            s_min_l = max(4, len(local_window) // 20)
            s_max_l = len(local_window) // 4
            if s_max_l > s_min_l:
                scales_l = np.unique(
                    np.logspace(np.log10(s_min_l), np.log10(s_max_l), 10).astype(int)
                )
                H_local, _, _, _ = self._dfa_fit(local_window, scales_l)
            else:
                H_local = H_global
        else:
            H_local = H_global

        regime_shift = abs(H_global - H_local) > 0.2

        # Stationarity
        is_stationary = _adf_stationary(x)

        # Bootstrap CI
        ci_low, ci_high = self._bootstrap_ci(x, scales)

        gamma = float(np.clip(2 * H_global + 1, -5.0, 5.0))
        hurst = float(np.clip((gamma - 1.0) / 2.0, 0.0, 2.0))
        is_valid = r_sq >= self.min_quality and math.isfinite(gamma)

        return DFAEstimate(
            value=gamma,
            hurst=round(hurst, 4),
            quality=round(r_sq, 4),
            ci_low=round(ci_low, 4),
            ci_high=round(ci_high, 4),
            is_valid=is_valid,
            is_stationary=is_stationary,
            local_H=round(H_local, 4),
            regime_shift=regime_shift,
        )

    def _dfa_fit(
        self, x: np.ndarray, scales: np.ndarray
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        """Core DFA: returns (H, R², fluctuations, valid_scales)."""
        y = np.cumsum(x - np.mean(x))
        n = len(y)
        fluct = []
        valid = []

        for s in scales:
            s = int(s)
            n_seg = n // s
            if n_seg < 1:
                continue
            var_sum = 0.0
            for v in range(n_seg):
                seg = y[v * s : (v + 1) * s]
                t = np.arange(s, dtype=np.float64)
                coeffs = np.polyfit(t, seg, 1)
                trend = coeffs[0] * t + coeffs[1]
                var_sum += np.mean((seg - trend) ** 2)
            f_s = np.sqrt(var_sum / n_seg)
            if f_s > 0 and math.isfinite(f_s):
                fluct.append(f_s)
                valid.append(s)

        if len(fluct) < 4:
            return 0.0, 0.0, np.array([]), np.array([])

        log_s = np.log(np.array(valid, dtype=np.float64))
        log_f = np.log(np.array(fluct, dtype=np.float64))

        coeffs = np.polyfit(log_s, log_f, 1)
        H = float(coeffs[0])

        # R²
        pred = coeffs[0] * log_s + coeffs[1]
        ss_res = np.sum((log_f - pred) ** 2)
        ss_tot = np.sum((log_f - np.mean(log_f)) ** 2)
        r_sq = float(1.0 - ss_res / (ss_tot + 1e-12))

        return H, max(0.0, r_sq), np.array(fluct), np.array(valid)

    def _bootstrap_ci(self, x: np.ndarray, scales: np.ndarray) -> tuple[float, float]:
        """Bootstrap 95% CI on H via block resampling."""
        rng = np.random.Generator(np.random.PCG64(42))
        n = len(x)
        block_size = max(n // 10, 16)
        Hs: list[float] = []

        for _ in range(self.bootstrap_n):
            # Block bootstrap (preserves autocorrelation)
            n_blocks = n // block_size + 1
            starts = rng.integers(0, max(1, n - block_size), size=n_blocks)
            sample = np.concatenate([x[s : s + block_size] for s in starts])[:n]
            H_b, r_sq_b, _, _ = self._dfa_fit(sample, scales)
            if r_sq_b >= 0.5 and math.isfinite(H_b):
                Hs.append(2 * H_b + 1)  # store gamma, not H

        if len(Hs) < 20:
            return -5.0, 5.0
        return float(np.percentile(Hs, 2.5)), float(np.percentile(Hs, 97.5))


def _adf_stationary(x: np.ndarray) -> bool:
    try:
        from statsmodels.tsa.stattools import adfuller  # noqa: PLC0415

        result = adfuller(x, maxlag=int(np.sqrt(len(x))), autolag=None)
        return float(result[1]) < 0.05
    except (ImportError, ValueError, np.linalg.LinAlgError):
        return True
