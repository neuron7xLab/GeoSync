# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Peters-Kelly ergodicity correction for non-ergodic markets.

Theory (Peters 2019):
  Ensemble growth: g_ens = μ
  Time growth:     g_time = μ - σ²/2
  Ergodicity gap:  Δg = σ²/2  (always ≥ 0)

  NEI = σ² / (2|μ| + ε)  — Non-Ergodicity Index
    < 0.5  → ERGODIC (proceed)
    0.5–1  → MILD (reduce 50%)
    1–2    → SIGNIFICANT (observe)
    > 2    → SEVERE (abort)
    μ ≈ 0  → ABORT (no edge)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ErgodicityState:
    """Ergodicity assessment for current returns window."""

    mu: float  # drift
    sigma: float  # volatility
    ergodicity_gap: float  # σ²/2, always ≥ 0
    nei: float  # non-ergodicity index
    kelly_corrected: float  # f* ∈ [0, 1]
    pragmatic_discount: float  # exp(-Δg × horizon) ∈ [0, 1]
    regime: str  # ERGODIC / MILD / SIGNIFICANT / SEVERE / ABORT


class ErgodicityCorrection:
    """Peters-Kelly correction for decision pipeline.

    Parameters
    ----------
    horizon_bars
        Expected trade duration for discount computation.
    mu_epsilon
        Minimum |μ| to avoid division by zero.
    """

    def __init__(
        self,
        horizon_bars: float = 10.0,
        mu_epsilon: float = 1e-8,
    ) -> None:
        self.horizon = horizon_bars
        self.mu_eps = mu_epsilon

    def update(self, returns: np.ndarray) -> ErgodicityState:
        """Compute ergodicity state from returns window."""
        r = np.asarray(returns, dtype=np.float64)
        r = r[np.isfinite(r)]

        if len(r) < 10:
            return ErgodicityState(
                mu=0.0,
                sigma=0.0,
                ergodicity_gap=0.0,
                nei=999.0,
                kelly_corrected=0.0,
                pragmatic_discount=0.0,
                regime="ABORT",
            )

        mu = float(np.mean(r))
        sigma = float(np.std(r, ddof=1))

        # Ergodicity gap: always ≥ 0
        gap = sigma**2 / 2.0

        # Non-Ergodicity Index
        nei = sigma**2 / (2.0 * abs(mu) + self.mu_eps)

        # Kelly with ergodicity correction
        if sigma > 1e-12 and abs(mu) > self.mu_eps:
            f_kelly = mu / (sigma**2)
            correction = max(0.0, 1.0 - min(1.0, nei))
            kelly = float(np.clip(f_kelly * correction, 0.0, 1.0))
        else:
            kelly = 0.0

        # Pragmatic discount: exp(-Δg × horizon)
        discount = float(np.clip(math.exp(-gap * self.horizon), 0.0, 1.0))

        # Regime classification
        if abs(mu) < self.mu_eps:
            regime = "ABORT"
        elif nei < 0.5:
            regime = "ERGODIC"
        elif nei < 1.0:
            regime = "MILD"
        elif nei < 2.0:
            regime = "SIGNIFICANT"
        else:
            regime = "SEVERE"

        return ErgodicityState(
            mu=round(mu, 8),
            sigma=round(sigma, 8),
            ergodicity_gap=round(gap, 8),
            nei=round(nei, 4),
            kelly_corrected=round(kelly, 6),
            pragmatic_discount=round(discount, 6),
            regime=regime,
        )

    def correct_pragmatic(
        self,
        pragmatic: float,
        state: ErgodicityState,
    ) -> float:
        """Apply ergodicity discount to pragmatic value."""
        return pragmatic * state.pragmatic_discount
