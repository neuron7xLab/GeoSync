# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Configuration and parameter validation for the Kuramoto simulation engine."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class KuramotoConfig(BaseModel):
    """Validated configuration for deterministic Kuramoto simulations.

    Coupling semantics:
    - ``adjacency is None``: global all-to-all coupling with per-edge weight ``K/N``.
    - ``adjacency is not None``: explicit weighted topology with effective weight ``K*A_ij``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    N: int = Field(default=10, ge=2, description="Number of oscillators (≥ 2)")
    K: float = Field(default=1.0, description="Global coupling strength scale")
    omega: np.ndarray | None = Field(
        default=None,
        description="Natural frequencies [rad/time]; length N. Drawn from N(0,1) if None.",
    )
    dt: float = Field(default=0.01, gt=0.0, description="Integration time-step (must be > 0)")
    steps: int = Field(default=1000, ge=1, description="Number of integration steps (≥ 1)")
    adjacency: np.ndarray | None = Field(
        default=None,
        description=(
            "Optional N×N weighted coupling matrix. Diagonal values are accepted but ignored "
            "during integration (no-self-coupling)."
        ),
    )
    theta0: np.ndarray | None = Field(
        default=None,
        description="Initial phases [rad]; length N. Drawn from U(0, 2π) if None.",
    )
    seed: int | None = Field(default=None, description="Integer RNG seed for reproducibility.")

    @model_validator(mode="after")
    def _validate_arrays(self) -> KuramotoConfig:
        """Validate scalar constraints and array shapes/finiteness."""
        N = self.N

        if not np.isfinite(self.K):
            raise ValueError("'K' must be finite.")
        if self.seed is not None and self.seed < 0:
            raise ValueError("'seed' must be >= 0 when provided.")

        if self.omega is not None:
            omega = np.asarray(self.omega, dtype=np.float64)
            if omega.ndim != 1 or omega.shape[0] != N:
                raise ValueError(f"'omega' must be a 1-D array of length N={N}; got shape {omega.shape}.")
            if not np.isfinite(omega).all():
                raise ValueError("'omega' contains non-finite values (NaN or Inf).")
            object.__setattr__(self, "omega", omega)

        if self.theta0 is not None:
            theta0 = np.asarray(self.theta0, dtype=np.float64)
            if theta0.ndim != 1 or theta0.shape[0] != N:
                raise ValueError(f"'theta0' must be a 1-D array of length N={N}; got shape {theta0.shape}.")
            if not np.isfinite(theta0).all():
                raise ValueError("'theta0' contains non-finite values (NaN or Inf).")
            object.__setattr__(self, "theta0", theta0)

        if self.adjacency is not None:
            adj = np.asarray(self.adjacency, dtype=np.float64)
            if adj.ndim != 2 or adj.shape != (N, N):
                raise ValueError(f"'adjacency' must be an (N×N)=({N}×{N}) matrix; got shape {adj.shape}.")
            if not np.isfinite(adj).all():
                raise ValueError("'adjacency' contains non-finite values (NaN or Inf).")
            object.__setattr__(self, "adjacency", adj)

        return self

    @property
    def coupling_mode(self) -> str:
        """Return active coupling semantics label for summaries/serialization."""
        return "adjacency" if self.adjacency is not None else "global"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a deterministic, JSON-compatible dictionary."""
        return {
            "N": self.N,
            "K": self.K,
            "omega": self.omega.tolist() if self.omega is not None else None,
            "dt": self.dt,
            "steps": self.steps,
            "adjacency": self.adjacency.tolist() if self.adjacency is not None else None,
            "theta0": self.theta0.tolist() if self.theta0 is not None else None,
            "seed": self.seed,
            "coupling_mode": self.coupling_mode,
        }
