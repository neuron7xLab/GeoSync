# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""T3 — Conservation Laws for Portfolio Energy.

Defines portfolio energy:
    E_kinetic   = ½ · Σ(position_i · velocity_i²)
        where velocity_i = 5-period price return
    E_potential = -Σ(position_i · expected_return_i)
        where expected_return from Kuramoto coherence signal
    E_total     = E_kinetic + E_potential

Conservation constraint:
    ΔE_total per rebalance ≤ ε (transaction cost threshold)

This is NOT claiming physical energy conservation.
It IS claiming: portfolio rebalancing is energy-conservative unless
an explicit regime signal overrides it. This is testable. This is honest.

Violation = energy injection from external force (regime change signal).
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

_logger = logging.getLogger(__name__)


class PortfolioEnergyConservation:
    """Track and enforce portfolio energy conservation.

    Parameters
    ----------
    epsilon : float
        Maximum allowed |ΔE| per rebalance (default 0.05).
        Free parameter — requires calibration via transaction cost analysis.
    return_window : int
        Window for computing price returns as velocity proxy (default 5).
    """

    def __init__(self, epsilon: float = 0.05, return_window: int = 5) -> None:
        if epsilon < 0:
            raise ValueError(f"epsilon must be ≥ 0, got {epsilon}")
        if return_window < 1:
            raise ValueError(f"return_window must be ≥ 1, got {return_window}")
        self._epsilon = epsilon
        self._return_window = return_window
        self._violation_count = 0

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def violation_count(self) -> int:
        return self._violation_count

    @staticmethod
    def compute_kinetic(
        positions: NDArray[np.float64],
        returns: NDArray[np.float64],
    ) -> float:
        """E_kinetic = ½ · Σ(|position_i| · return_i²).

        Uses |position| to ensure kinetic energy is non-negative.
        """
        positions = np.asarray(positions, dtype=np.float64)
        returns = np.asarray(returns, dtype=np.float64)
        if positions.shape != returns.shape:
            raise ValueError(
                f"positions and returns must match: {positions.shape} vs {returns.shape}"
            )
        return 0.5 * float(np.sum(np.abs(positions) * returns ** 2))

    @staticmethod
    def compute_potential(
        positions: NDArray[np.float64],
        expected_returns: NDArray[np.float64],
    ) -> float:
        """E_potential = -Σ(position_i · expected_return_i).

        Negative sign: aligned positions with expected returns → lower potential
        → more stable configuration. This mirrors gravitational potential.
        """
        positions = np.asarray(positions, dtype=np.float64)
        expected_returns = np.asarray(expected_returns, dtype=np.float64)
        if positions.shape != expected_returns.shape:
            raise ValueError(
                f"positions and expected_returns must match: "
                f"{positions.shape} vs {expected_returns.shape}"
            )
        return -float(np.sum(positions * expected_returns))

    def compute_total(
        self,
        positions: NDArray[np.float64],
        returns: NDArray[np.float64],
        expected_returns: NDArray[np.float64],
    ) -> float:
        """E_total = E_kinetic + E_potential."""
        ek = self.compute_kinetic(positions, returns)
        ep = self.compute_potential(positions, expected_returns)
        return ek + ep

    def check_conservation(
        self,
        E_before: float,
        E_after: float,
    ) -> bool:
        """Check |ΔE| ≤ ε.

        If violated, increments internal counter and logs warning.
        Returns True if conservation holds.
        """
        delta = abs(E_after - E_before)
        conserved = delta <= self._epsilon
        if not conserved:
            self._violation_count += 1
            _logger.warning(
                "Energy conservation violated: ΔE=%.6f > ε=%.6f (violation #%d)",
                delta,
                self._epsilon,
                self._violation_count,
            )
        return conserved

    def reset_violations(self) -> None:
        """Reset violation counter."""
        self._violation_count = 0


__all__ = ["PortfolioEnergyConservation"]
