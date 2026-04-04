# SPDX-License-Identifier: MIT
"""T4 — Thermodynamic Entropy Control for Risk Manager.

Shannon entropy:
    S = -Σ w_i · log(w_i)   where w_i = |pos_i| / Σ|pos_j|

Tsallis entropy (q-generalisation for fat tails):
    S_q = (1 - Σ w_i^q) / (q-1),  q=1.5

    q=1.5 justification: empirically fitted to Pareto-distributed
    financial returns (Borland 2002, Tsallis et al. 2003). The
    q-Gaussian with q≈1.5 reproduces heavy tails observed in
    intraday returns. This is a calibrated parameter, not intuition.

Ricci curvature coupling:
    T_effective = T_base · exp(-κ_min)
    Negative curvature → higher T → more entropy allowed → looser gate.
    Positive curvature → lower T → tighter gate → more conservative.

Free energy gate:
    F = U - T_eff · S_q
    dF/dt ≤ 0 required for position update.

Lyapunov stability proof sketch:
    dF/dt = dU/dt - T·dS/dt
    dU/dt bounded by Kelly-optimal sizing
    dS/dt ≥ 0 by second-law analog (diversification increases entropy)
    → dF/dt ≤ dU/dt ≤ 0 when Kelly is respected

T_base=0.60: derived from TACL calibration in tacl/energy_model.py,
not from intuition. The value matches the production temperature used
in the EnergyModel class.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class ThermodynamicRiskGate:
    """Thermodynamic risk gate using Tsallis entropy and Ricci curvature.

    Parameters
    ----------
    q : float
        Tsallis entropic index (default 1.5).
    T_base : float
        Base temperature from TACL calibration (default 0.60).
    """

    def __init__(self, q: float = 1.5, T_base: float = 0.60) -> None:
        if q <= 0:
            raise ValueError(f"q must be > 0, got {q}")
        if abs(q - 1.0) < 1e-12:
            raise ValueError("q=1.0 degenerates; use shannon_entropy instead")
        if T_base <= 0:
            raise ValueError(f"T_base must be > 0, got {T_base}")
        self._q = q
        self._T_base = T_base

    @property
    def q(self) -> float:
        return self._q

    @property
    def T_base(self) -> float:
        return self._T_base

    @staticmethod
    def shannon_entropy(weights: NDArray[np.float64]) -> float:
        """S = -Σ w_i · ln(w_i) for position weight distribution.

        Input weights are normalised internally: w_i = |pos_i| / Σ|pos_j|.
        """
        w = np.asarray(weights, dtype=np.float64)
        w = np.abs(w)
        total = w.sum()
        if total < 1e-12:
            return 0.0
        w = w / total
        nonzero = w[w > 0]
        return -float(np.sum(nonzero * np.log(nonzero)))

    def tsallis_entropy(self, weights: NDArray[np.float64]) -> float:
        """S_q = (1 - Σ w_i^q) / (q-1).

        q=1.5 captures fat-tailed portfolio weight distributions.
        As q→1, reduces to Shannon entropy (L'Hôpital).
        """
        w = np.asarray(weights, dtype=np.float64)
        w = np.abs(w)
        total = w.sum()
        if total < 1e-12:
            return 0.0
        w = w / total
        return (1.0 - float(np.sum(w ** self._q))) / (self._q - 1.0)

    def ricci_temperature(self, kappa_min: float) -> float:
        """T_eff = T_base · exp(-κ_min).

        κ_min < 0 (negative curvature) → T_eff > T_base → looser risk.
        κ_min > 0 (positive curvature) → T_eff < T_base → tighter risk.
        """
        return self._T_base * np.exp(-kappa_min)

    @staticmethod
    def free_energy(U: float, T: float, S: float) -> float:
        """F = U - T·S (Helmholtz free energy)."""
        return U - T * S

    def gate(self, dU: float, dS: float, kappa_min: float = 0.0) -> bool:
        """Free energy gate with Ricci-coupled temperature.

        Parameters
        ----------
        dU : float
            Change in internal energy (P&L delta).
        dS : float
            Change in Tsallis entropy.
        kappa_min : float
            Minimum Ollivier-Ricci curvature from network.

        Returns
        -------
        True if dF/dt ≤ 0 (position update allowed).
        """
        T_eff = self.ricci_temperature(kappa_min)
        dF = dU - T_eff * dS
        return dF <= 0.0

    def gate_with_details(
        self, dU: float, dS: float, kappa_min: float = 0.0
    ) -> dict[str, float | bool]:
        """Gate with full diagnostic output."""
        T_eff = self.ricci_temperature(kappa_min)
        dF = dU - T_eff * dS
        return {
            "allowed": dF <= 0.0,
            "dF": dF,
            "dU": dU,
            "dS": dS,
            "T_eff": T_eff,
            "kappa_min": kappa_min,
        }


__all__ = ["ThermodynamicRiskGate"]
