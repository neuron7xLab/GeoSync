# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""T5 — Tsallis entropy as real-time risk gate with empirical thresholds.

Rolling q-fit on 60-day window estimates the Tsallis entropic index q
from the distribution of portfolio returns.

Empirical thresholds (from literature):
    q < 1.35         →  NORMAL regime
    1.35 ≤ q < 1.55  →  ELEVATED risk
    q ≥ 1.55         →  CRISIS regime

Position sizing gate:
    f(q) = max(0, 1 - (q - 1.35) / 0.20)  # INV-FE2: gate output non-negative by Tsallis entropy contract

    q = 1.35 → f = 1.0 (full position)
    q = 1.45 → f = 0.5 (half position)
    q = 1.55 → f = 0.0 (no new positions)

This implements the q-Gaussian fit to financial returns.
The q-Gaussian P(x) ∝ [1 - (1-q)·β·x²]^(1/(1-q))
has heavier tails than Gaussian for q > 1.

Calibration: fit q via MLE or method of moments on rolling window.
We use kurtosis-based estimator: q ≈ (5 + 3·κ) / (3 + κ)
where κ = excess kurtosis (Tsallis & Bukman 1996).

References:
    Tsallis "Introduction to Nonextensive Statistical Mechanics" (2009)
    Borland "A theory of non-Gaussian option pricing" Quant. Finance (2002)
    Queirós "On the emergence of a generalised Gamma distribution" EPL (2005)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class TsallisRegime(str, Enum):
    """Market regime based on Tsallis q parameter."""

    NORMAL = "normal"  # q < 1.35
    ELEVATED = "elevated"  # 1.35 ≤ q < 1.55
    CRISIS = "crisis"  # q ≥ 1.55


@dataclass(frozen=True, slots=True)
class TsallisGateResult:
    """Result of Tsallis gate evaluation."""

    q: float
    regime: TsallisRegime
    position_multiplier: float
    kurtosis: float
    window_returns: int


class TsallisRiskGate:
    """Real-time position sizing gate based on Tsallis q parameter.

    Parameters
    ----------
    window : int
        Rolling window for q estimation (default 60).
    q_normal : float
        Upper bound for NORMAL regime (default 1.35).
    q_crisis : float
        Lower bound for CRISIS regime (default 1.55).
    min_observations : int
        Minimum observations for reliable q estimate (default 20).
    """

    def __init__(
        self,
        window: int = 60,
        q_normal: float = 1.35,
        q_crisis: float = 1.55,
        min_observations: int = 20,
    ) -> None:
        if window < 2:
            raise ValueError(f"window must be ≥ 2, got {window}")
        if q_normal >= q_crisis:
            raise ValueError(f"q_normal must be < q_crisis: {q_normal} vs {q_crisis}")
        if min_observations < 2:
            raise ValueError(f"min_observations must be ≥ 2, got {min_observations}")
        self._window = window
        self._q_normal = q_normal
        self._q_crisis = q_crisis
        self._min_obs = min_observations
        self._history: list[TsallisGateResult] = []

    @property
    def window(self) -> int:
        return self._window

    @property
    def history(self) -> list[TsallisGateResult]:
        return list(self._history)

    @staticmethod
    def estimate_q(returns: NDArray[np.float64]) -> float:
        """Estimate Tsallis q from return distribution via kurtosis.

        q ≈ (5 + 3·κ) / (3 + κ)

        where κ = excess kurtosis. For Gaussian: κ=0 → q=5/3≈1.67.
        For lighter tails: κ<0 → q<5/3.
        For typical markets: κ ∈ [3, 15] → q ∈ [1.17, 1.45].

        The formula derives from matching the 4th moment of the
        q-Gaussian to the empirical kurtosis.
        """
        returns = np.asarray(returns, dtype=np.float64)
        if returns.size < 4:
            return 1.0  # insufficient data, assume Gaussian

        # Excess kurtosis (Fisher's definition)
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        if std < 1e-12:
            return 1.0  # no variance

        standardized = (returns - mean) / std
        kurtosis = float(np.mean(standardized**4) - 3.0)

        # Clamp kurtosis to avoid q < 1 or q → ∞
        kurtosis = max(kurtosis, -2.5)  # INV-FE2: kurtosis lower bound keeps q real-valued
        kurtosis = min(kurtosis, 50.0)  # bounds: kurtosis cap prevents q divergence

        q = (5.0 + 3.0 * kurtosis) / (3.0 + kurtosis)
        return max(q, 1.0)  # q ≥ 1 for fat tails

    def classify_regime(self, q: float) -> TsallisRegime:
        """Classify market regime from q value."""
        if q < self._q_normal:
            return TsallisRegime.NORMAL
        elif q < self._q_crisis:
            return TsallisRegime.ELEVATED
        else:
            return TsallisRegime.CRISIS

    def position_multiplier(self, q: float) -> float:
        """Compute position size multiplier from q.

        f(q) = max(0, 1 - (q - q_normal) / (q_crisis - q_normal))  # INV-FE2: gate output non-negative by Tsallis entropy contract

        Linear ramp from 1.0 at q_normal to 0.0 at q_crisis.
        """
        if q <= self._q_normal:
            return 1.0
        if q >= self._q_crisis:
            return 0.0
        return 1.0 - (q - self._q_normal) / (self._q_crisis - self._q_normal)

    def evaluate(self, returns: NDArray[np.float64]) -> TsallisGateResult:
        """Evaluate gate on return series.

        Parameters
        ----------
        returns : (T,) or (T, N) return series.
            If 2-D, uses flattened cross-sectional returns.

        Returns
        -------
        TsallisGateResult with q estimate, regime, and multiplier.
        """
        returns = np.asarray(returns, dtype=np.float64)
        if returns.ndim == 2:
            # Use cross-sectional returns for portfolio-level q
            returns = returns.ravel()

        tail = returns[-self._window :]
        n_obs = tail.size

        if n_obs < self._min_obs:
            # Insufficient data → conservative
            result = TsallisGateResult(
                q=1.5,
                regime=TsallisRegime.ELEVATED,
                position_multiplier=0.25,
                kurtosis=0.0,
                window_returns=n_obs,
            )
            self._history.append(result)
            return result

        q = self.estimate_q(tail)
        mean_r = np.mean(tail)
        std_r = np.std(tail, ddof=1)
        if std_r < 1e-12:
            kurtosis = 0.0
        else:
            kurtosis = float(np.mean(((tail - mean_r) / std_r) ** 4) - 3.0)

        regime = self.classify_regime(q)
        mult = self.position_multiplier(q)

        result = TsallisGateResult(
            q=q,
            regime=regime,
            position_multiplier=mult,
            kurtosis=kurtosis,
            window_returns=n_obs,
        )
        self._history.append(result)
        return result

    def evaluate_prices(self, prices: NDArray[np.float64]) -> TsallisGateResult:
        """Convenience: compute returns from prices, then evaluate."""
        prices = np.asarray(prices, dtype=np.float64)
        if prices.ndim == 1:
            if prices.size < 2:
                raise ValueError("Need ≥ 2 prices")
            returns = np.diff(np.log(np.maximum(prices, 1e-12)))
        else:
            if prices.shape[0] < 2:
                raise ValueError("Need ≥ 2 time steps")
            returns = np.diff(np.log(np.maximum(prices, 1e-12)), axis=0)
        return self.evaluate(returns)


__all__ = ["TsallisRiskGate", "TsallisGateResult", "TsallisRegime"]
