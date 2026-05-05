# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""T6 — dF/dt ≤ 0 trading gate with order-book temperature calibration.

Free energy formulation:
    F = U - T_LOB(t) · S_q(t)

where:
    U        = portfolio risk exposure (e.g. VaR or weighted drawdown)
    T_LOB(t) = kinetic temperature from order book (Li et al. Entropy 2024)
    S_q(t)   = Tsallis entropy of position weights

Trade admitted only if:
    ΔF = F_after - F_before ≤ 0

T_LOB calibration (when order book data is available):
    T_LOB = (2/N) · Σ ½·m_i·v_i²
    where m_i = lot size, v_i = price velocity at level i

When order book not available, falls back to:
    T_LOB = σ² / σ²_ref · T_base

where σ = realised volatility, T_base = 0.60 (TACL calibration).

Backtest validation target: gate trigger rate 5–20%.
    If 0%  → gate trivially satisfied → useless
    If 95% → gate too tight → system can't trade

This is the first backtest of thermodynamic trading constraint
in the literature if published.

References:
    Li et al. "Kinetic temperature of order book" Entropy (2024)
    Tsallis "Nonextensive statistics" J. Stat. Phys. (1988)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class FreeEnergyTradeDecision:
    """Decision from free energy trading gate."""

    allowed: bool
    F_before: float
    F_after: float
    delta_F: float
    U_before: float
    U_after: float
    T_LOB: float
    S_q_before: float
    S_q_after: float


@dataclass(frozen=True, slots=True)
class GateStatistics:
    """Gate trigger statistics for validation."""

    total_checks: int
    total_rejected: int
    trigger_rate: float  # rejected / total
    mean_delta_F: float
    is_calibrated: bool  # True if 5% ≤ trigger_rate ≤ 20%


class FreeEnergyTradingGate:
    """Thermodynamic trading gate: ΔF ≤ 0 required for trade admission.

    Parameters
    ----------
    T_base : float
        Base temperature when LOB data unavailable (default 0.60).
    q : float
        Tsallis entropic index (default 1.5).
    vol_reference : float
        Reference volatility for temperature scaling (default 0.01).
    """

    def __init__(
        self,
        T_base: float = 0.60,
        q: float = 1.5,
        vol_reference: float = 0.01,
    ) -> None:
        if T_base <= 0:
            raise ValueError(f"T_base must be > 0, got {T_base}")
        if q <= 1.0:
            raise ValueError(f"q must be > 1.0, got {q}")
        if vol_reference <= 0:
            raise ValueError(f"vol_reference must be > 0, got {vol_reference}")
        self._T_base = T_base
        self._q = q
        self._vol_ref = vol_reference
        self._total_checks = 0
        self._total_rejected = 0
        self._delta_F_history: list[float] = []

    @property
    def T_base(self) -> float:
        return self._T_base

    def compute_T_LOB(
        self,
        order_book_velocities: NDArray[np.float64] | None = None,
        order_book_sizes: NDArray[np.float64] | None = None,
        realized_volatility: float | None = None,
    ) -> float:
        """Compute LOB kinetic temperature.

        If order book data available:
            T_LOB = (2/N) · Σ ½·m_i·v_i²

        Otherwise fallback:
            T_LOB = (σ/σ_ref)² · T_base
        """
        if order_book_velocities is not None and order_book_sizes is not None:
            v = np.asarray(order_book_velocities, dtype=np.float64)
            m = np.asarray(order_book_sizes, dtype=np.float64)
            if v.size == 0:
                return self._T_base
            kinetic = 0.5 * np.sum(m * v**2)
            T = 2.0 * kinetic / max(v.size, 1)
            return max(float(T), 1e-6)

        if realized_volatility is not None:
            ratio = realized_volatility / self._vol_ref
            return self._T_base * ratio**2

        return self._T_base

    def tsallis_entropy(self, weights: NDArray[np.float64]) -> float:
        """S_q = (1 - Σ w_i^q) / (q-1) on normalised position weights."""
        w = np.abs(np.asarray(weights, dtype=np.float64))
        total = w.sum()
        if total < 1e-12:
            return 0.0
        w = w / total
        return (1.0 - float(np.sum(w**self._q))) / (self._q - 1.0)

    def compute_risk_exposure(
        self,
        positions: NDArray[np.float64],
        returns: NDArray[np.float64],
    ) -> float:
        """Portfolio risk exposure U = Σ|pos_i| · |ret_i| (simple VaR proxy)."""
        pos = np.abs(np.asarray(positions, dtype=np.float64))
        ret = np.abs(np.asarray(returns, dtype=np.float64))
        if pos.shape != ret.shape:
            raise ValueError(f"Shape mismatch: {pos.shape} vs {ret.shape}")
        return float(np.sum(pos * ret))

    def check(
        self,
        positions_before: NDArray[np.float64],
        positions_after: NDArray[np.float64],
        recent_returns: NDArray[np.float64],
        T_LOB: float | None = None,
        realized_volatility: float | None = None,
    ) -> FreeEnergyTradeDecision:
        """Check if trade (position change) satisfies ΔF ≤ 0.

        Parameters
        ----------
        positions_before : (N,) current positions.
        positions_after : (N,) proposed positions.
        recent_returns : (N,) recent asset returns.
        T_LOB : float, order-book temperature (if pre-computed).
        realized_volatility : float, for temperature fallback.

        Returns
        -------
        FreeEnergyTradeDecision.
        """
        pos_before = np.asarray(positions_before, dtype=np.float64)
        pos_after = np.asarray(positions_after, dtype=np.float64)
        returns = np.asarray(recent_returns, dtype=np.float64)

        # Temperature
        if T_LOB is None:
            T_LOB = self.compute_T_LOB(realized_volatility=realized_volatility)

        # Risk exposure
        U_before = self.compute_risk_exposure(pos_before, returns)
        U_after = self.compute_risk_exposure(pos_after, returns)

        # Entropy
        S_before = self.tsallis_entropy(pos_before)
        S_after = self.tsallis_entropy(pos_after)

        # Free energy
        F_before = U_before - T_LOB * S_before
        F_after = U_after - T_LOB * S_after
        delta_F = F_after - F_before

        allowed = delta_F <= 0.0
        self._total_checks += 1
        if not allowed:
            self._total_rejected += 1
        self._delta_F_history.append(delta_F)

        return FreeEnergyTradeDecision(
            allowed=bool(allowed),
            F_before=F_before,
            F_after=F_after,
            delta_F=delta_F,
            U_before=U_before,
            U_after=U_after,
            T_LOB=T_LOB,
            S_q_before=S_before,
            S_q_after=S_after,
        )

    def statistics(self) -> GateStatistics:
        """Get gate trigger statistics for calibration validation."""
        total = self._total_checks
        rejected = self._total_rejected
        rate = rejected / max(total, 1)
        mean_dF = float(np.mean(self._delta_F_history)) if self._delta_F_history else 0.0
        return GateStatistics(
            total_checks=total,
            total_rejected=rejected,
            trigger_rate=rate,
            mean_delta_F=mean_dF,
            is_calibrated=0.05 <= rate <= 0.20 if total >= 20 else False,
        )

    def reset_statistics(self) -> None:
        """Reset gate counters."""
        self._total_checks = 0
        self._total_rejected = 0
        self._delta_F_history.clear()


__all__ = [
    "FreeEnergyTradingGate",
    "FreeEnergyTradeDecision",
    "GateStatistics",
]
