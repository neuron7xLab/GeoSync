# SPDX-License-Identifier: LicenseRef-TradePulse-Proprietary
"""Configuration and parameter validation for the Kuramoto simulation engine.

All public parameters are validated via Pydantic v2 models so that downstream
code never receives out-of-range or type-incorrect inputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class KuramotoConfig(BaseModel):
    """Validated configuration container for a Kuramoto simulation run.

    Parameters
    ----------
    N:
        Number of coupled oscillators.  Must be ≥ 2.
    K:
        Global coupling strength (dimensionless).  Positive values lead to
        synchronisation; negative values may lead to anti-phase clustering.
        Ignored when an explicit ``adjacency`` matrix is provided.
    omega:
        Natural frequencies of each oscillator in radians per unit time.
        Must be a 1-D array of length ``N``.  If ``None`` the frequencies are
        drawn from N(0, 1) using ``seed``.
    dt:
        Integration step in the same time units as ``omega``.  Must be > 0.
    steps:
        Total number of integration steps to run.  Must be ≥ 1.
    adjacency:
        Optional N×N coupling matrix **A** (weights; need not be binary).
        When provided, the coupling term becomes ``K · Σⱼ Aᵢⱼ sin(θⱼ − θᵢ)``
        and ``K`` acts as a global scale factor applied to every edge.
        When ``None`` a fully-connected (all-to-all) topology is used.
    theta0:
        Initial phases in radians for each oscillator.  Must be a 1-D array
        of length ``N``.  If ``None`` the phases are drawn from Uniform(0, 2π)
        using ``seed``.
    seed:
        Integer random seed for reproducible draws of ``omega`` and/or
        ``theta0``.  Has no effect when both arrays are supplied explicitly.

    Examples
    --------
    >>> cfg = KuramotoConfig(N=5, K=2.0, dt=0.01, steps=1000, seed=42)
    >>> cfg.N
    5
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    N: int = Field(default=10, ge=2, description="Number of oscillators (≥ 2)")
    K: float = Field(default=1.0, description="Global coupling strength")
    omega: np.ndarray | None = Field(
        default=None,
        description="Natural frequencies [rad/time]; length N. Drawn from N(0,1) if None.",
    )
    dt: float = Field(
        default=0.01,
        gt=0.0,
        description="Integration time-step (must be > 0)",
    )
    steps: int = Field(
        default=1000,
        ge=1,
        description="Number of integration steps (≥ 1)",
    )
    adjacency: np.ndarray | None = Field(
        default=None,
        description="N×N adjacency/coupling matrix. None → fully-connected.",
    )
    theta0: np.ndarray | None = Field(
        default=None,
        description="Initial phases [rad]; length N. Drawn from U(0, 2π) if None.",
    )
    seed: int | None = Field(
        default=None,
        description="Integer RNG seed for reproducibility.",
    )

    @model_validator(mode="after")
    def _validate_arrays(self) -> KuramotoConfig:
        """Cross-field validation for array dimensions and content."""
        N = self.N

        # ── omega ───────────────────────────────────────────────────────────
        if self.omega is not None:
            omega = np.asarray(self.omega, dtype=np.float64)
            if omega.ndim != 1 or omega.shape[0] != N:
                raise ValueError(
                    f"'omega' must be a 1-D array of length N={N}; "
                    f"got shape {omega.shape}."
                )
            if not np.isfinite(omega).all():
                raise ValueError("'omega' contains non-finite values (NaN or Inf).")
            object.__setattr__(self, "omega", omega)

        # ── theta0 ──────────────────────────────────────────────────────────
        if self.theta0 is not None:
            theta0 = np.asarray(self.theta0, dtype=np.float64)
            if theta0.ndim != 1 or theta0.shape[0] != N:
                raise ValueError(
                    f"'theta0' must be a 1-D array of length N={N}; "
                    f"got shape {theta0.shape}."
                )
            if not np.isfinite(theta0).all():
                raise ValueError("'theta0' contains non-finite values (NaN or Inf).")
            object.__setattr__(self, "theta0", theta0)

        # ── adjacency ───────────────────────────────────────────────────────
        if self.adjacency is not None:
            adj = np.asarray(self.adjacency, dtype=np.float64)
            if adj.ndim != 2 or adj.shape != (N, N):
                raise ValueError(
                    f"'adjacency' must be an (N×N)=({N}×{N}) matrix; "
                    f"got shape {adj.shape}."
                )
            if not np.isfinite(adj).all():
                raise ValueError(
                    "'adjacency' contains non-finite values (NaN or Inf)."
                )
            object.__setattr__(self, "adjacency", adj)

        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary (NumPy arrays converted to lists)."""
        return {
            "N": self.N,
            "K": self.K,
            "omega": self.omega.tolist() if self.omega is not None else None,
            "dt": self.dt,
            "steps": self.steps,
            "adjacency": (
                self.adjacency.tolist() if self.adjacency is not None else None
            ),
            "theta0": self.theta0.tolist() if self.theta0 is not None else None,
            "seed": self.seed,
        }
