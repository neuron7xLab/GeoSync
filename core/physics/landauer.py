# SPDX-License-Identifier: MIT
"""T7 — Landauer Bound for Inference Efficiency.

Landauer's principle (PHYSICAL, not metaphor):
    E_erase ≥ k_B · T · ln(2) per bit erased
    At T=300K: E_erase ≥ 2.87 × 10⁻²¹ J per bit

Application to GeoSync inference:
    Each numerical update involves implicit information erasure.
    Entropy generation per step: ΔS = n_params · k_B · ln(2)

Practical constraint:
    Minimise n_active_params per inference cycle.
    Prefer sparse models over dense for same predictive power.

HONEST DOCUMENTATION:
    kT·ln(2) ≈ 3×10⁻²¹ J vs actual GPU: ~10⁻¹² J per op.
    This is 9 orders of magnitude above Landauer limit.
    The Landauer bound is a THEORETICAL FLOOR, not a practical target.
    We use it as: minimise unnecessary computation, track as
    efficiency proxy. The metric is "how far from the physical limit",
    not "we achieve the physical limit".

E=mc² is NOT used. Landauer IS physical.
"""

from __future__ import annotations

import math

import numpy as np

# Physical constants (SI units)
K_BOLTZMANN: float = 1.380649e-23  # J/K (exact, 2019 SI redefinition)
ROOM_TEMPERATURE: float = 300.0    # K
LANDAUER_ENERGY: float = K_BOLTZMANN * ROOM_TEMPERATURE * math.log(2)
# ≈ 2.87 × 10⁻²¹ J per bit erased


class LandauerInferenceProfiler:
    """Track inference efficiency relative to Landauer bound.

    Parameters
    ----------
    T : float
        Operating temperature in Kelvin (default 300K).
    gpu_energy_per_op : float
        Estimated energy per GPU operation in Joules (default 1e-12).
        Typical for modern GPUs (NVIDIA A100: ~0.3 pJ/FLOP).
    """

    def __init__(
        self,
        T: float = ROOM_TEMPERATURE,
        gpu_energy_per_op: float = 1e-12,
    ) -> None:
        if T <= 0:
            raise ValueError(f"Temperature must be > 0, got {T}")
        if gpu_energy_per_op <= 0:
            raise ValueError(f"gpu_energy_per_op must be > 0, got {gpu_energy_per_op}")
        self._T = T
        self._gpu_energy_per_op = gpu_energy_per_op
        self._landauer_per_bit = K_BOLTZMANN * T * math.log(2)

    @property
    def landauer_per_bit(self) -> float:
        """Minimum energy per bit erasure at operating temperature (J)."""
        return self._landauer_per_bit

    def entropy_per_step(self, n_active_params: int) -> float:
        """Entropy generated per inference step (in bits).

        Each parameter update erases ≥ 1 bit of prior information.
        For float32: up to 32 bits per parameter, but effective
        information content is typically much less due to redundancy.
        We use 1 bit per parameter as conservative lower bound.

        Parameters
        ----------
        n_active_params : int
            Number of actively updated parameters.

        Returns
        -------
        Entropy in bits.
        """
        if n_active_params < 0:
            raise ValueError(f"n_active_params must be ≥ 0, got {n_active_params}")
        return float(n_active_params)

    def minimum_energy(self, n_active_params: int) -> float:
        """Landauer minimum energy for given parameter count (Joules).

        E_min = n_params · k_B · T · ln(2)
        """
        return n_active_params * self._landauer_per_bit

    def actual_energy(self, n_active_params: int) -> float:
        """Estimated actual GPU energy for given parameter count (Joules)."""
        return n_active_params * self._gpu_energy_per_op

    def efficiency(self, accuracy: float, n_active_params: int) -> float:
        """Inference efficiency = accuracy / entropy_generated.

        Higher is better: more predictive power per bit of computation.

        Parameters
        ----------
        accuracy : float
            Model predictive accuracy (0-1 scale).
        n_active_params : int
            Number of active parameters.

        Returns
        -------
        Efficiency ratio. Dimensionless.
        """
        entropy = self.entropy_per_step(n_active_params)
        if entropy < 1e-12:
            return float("inf") if accuracy > 0 else 0.0
        return accuracy / entropy

    def landauer_ratio(self, n_active_params: int) -> float:
        """Ratio of actual GPU energy to Landauer minimum.

        Measures how far the hardware operates above the physical limit.
        Typical: ~10⁹ (9 orders of magnitude above Landauer).
        """
        e_min = self.minimum_energy(n_active_params)
        e_actual = self.actual_energy(n_active_params)
        if e_min < 1e-30:
            return float("inf")
        return e_actual / e_min

    def recommend_pruning(
        self,
        param_importances: np.ndarray,
        target_efficiency: float,
        current_accuracy: float,
    ) -> dict[str, float | int]:
        """Recommend parameter pruning for target efficiency.

        Parameters
        ----------
        param_importances : (N,) importance scores (higher = more important).
        target_efficiency : desired efficiency metric.
        current_accuracy : current model accuracy.

        Returns
        -------
        Dict with pruning recommendation.
        """
        importances = np.asarray(param_importances, dtype=np.float64)
        n_total = importances.size

        if n_total == 0:
            return {
                "n_total": 0,
                "n_keep": 0,
                "n_prune": 0,
                "prune_ratio": 0.0,
                "estimated_efficiency": 0.0,
            }

        # Sort by importance (ascending — least important first)
        sorted_idx = np.argsort(importances)

        # Binary search for minimum params that achieve target efficiency
        best_n = n_total
        for n_keep in range(1, n_total + 1):
            # Estimate accuracy loss proportional to pruned importance
            kept = sorted_idx[-n_keep:]
            kept_importance = importances[kept].sum()
            total_importance = importances.sum()
            if total_importance > 0:
                est_accuracy = current_accuracy * (kept_importance / total_importance)
            else:
                est_accuracy = current_accuracy

            eff = self.efficiency(est_accuracy, n_keep)
            if eff >= target_efficiency:
                best_n = n_keep
                break

        return {
            "n_total": n_total,
            "n_keep": best_n,
            "n_prune": n_total - best_n,
            "prune_ratio": (n_total - best_n) / n_total,
            "estimated_efficiency": self.efficiency(current_accuracy, best_n),
        }


__all__ = ["LandauerInferenceProfiler", "K_BOLTZMANN", "ROOM_TEMPERATURE", "LANDAUER_ENERGY"]
