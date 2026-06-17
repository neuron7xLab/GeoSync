# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Kuramoto ODE simulation engine.

Coupling semantics implemented by this module:

- Global all-to-all mode (no explicit adjacency):
  ``dθ_i/dt = ω_i + (K/N) * Σ_{j != i} sin(θ_j - θ_i)``
- Explicit adjacency mode:
  ``dθ_i/dt = ω_i + K * Σ_j A_ij sin(θ_j - θ_i)``

FP-1 (single owner of scale, Issue #1096): ``K`` is the *only* scalar coupling
gain and it is applied exactly once, inside the RHS (:func:`_dtheta_dt`). The
resolved adjacency ``A_norm`` carries topology/affinity only — it never has a
hidden global ``K`` folded into it. Global mode stores ``1/N`` off-diagonal
weights; explicit mode stores the caller's topology verbatim. In both modes,
no-self-coupling is enforced by zeroing diagonal weights, so the boundary
spectral scale ``K · λ_max(A_norm)`` and the integrated RHS agree on the same
``K`` and the same ``A_norm``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import KuramotoConfig

__all__ = ["KuramotoEngine", "KuramotoResult", "run_simulation"]

_logger = logging.getLogger(__name__)


@dataclass(slots=True)
class KuramotoResult:
    """Simulation output container with shape and finiteness safeguards."""

    phases: NDArray[np.float64]
    order_parameter: NDArray[np.float64]
    time: NDArray[np.float64]
    config: KuramotoConfig
    summary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate_shapes()
        self.summary = self._compute_summary()

    def _validate_shapes(self) -> None:
        expected_steps = self.config.steps + 1
        expected_n = self.config.N
        if self.phases.shape != (expected_steps, expected_n):
            raise ValueError(
                "Result phases shape mismatch: "
                f"expected {(expected_steps, expected_n)}, got {self.phases.shape}."
            )
        if self.order_parameter.shape != (expected_steps,):
            raise ValueError(
                "Result order_parameter shape mismatch: "
                f"expected {(expected_steps,)}, got {self.order_parameter.shape}."
            )
        if self.time.shape != (expected_steps,):
            raise ValueError(
                f"Result time shape mismatch: expected {(expected_steps,)}, got {self.time.shape}."
            )

        if not np.isfinite(self.phases).all():
            raise ValueError("Result contains non-finite phase values.")
        if not np.isfinite(self.order_parameter).all():
            raise ValueError("Result contains non-finite order_parameter values.")
        if not np.isfinite(self.time).all():
            raise ValueError("Result contains non-finite time values.")
        tol = 1e-12
        if np.any(self.order_parameter < -tol) or np.any(self.order_parameter > 1.0 + tol):
            raise ValueError(
                "Result order_parameter values must stay within [0, 1] (±1e-12 tolerance)."
            )

    def _compute_summary(self) -> dict[str, Any]:
        R = self.order_parameter
        cfg = self.config
        return {
            "final_R": float(R[-1]),
            "mean_R": float(R.mean()),
            "max_R": float(R.max()),
            "min_R": float(R.min()),
            "std_R": float(R.std()),
            "N": cfg.N,
            "K": cfg.K,
            "dt": cfg.dt,
            "steps": cfg.steps,
            "total_time": float(self.time[-1]),
            "coupling_mode": cfg.coupling_mode,
            "seed": cfg.seed,
        }


class KuramotoEngine:
    """Deterministic Kuramoto RK4 integrator for validated configurations."""

    def __init__(self, config: KuramotoConfig) -> None:
        self._cfg = config
        self._K = float(config.K)
        self._omega, self._theta0 = self._resolve_initial_conditions(config)
        self._adj = self._resolve_adjacency(config)
        self._validate_runtime_inputs(self._omega, self._theta0, self._adj)

    def run(self) -> KuramotoResult:
        """Integrate ODE trajectories and return validated :class:`KuramotoResult`."""
        cfg = self._cfg
        N = cfg.N
        steps = cfg.steps
        dt = cfg.dt
        omega = self._omega
        adj = self._adj
        K = self._K

        phases = np.empty((steps + 1, N), dtype=np.float64)
        R_arr = np.empty(steps + 1, dtype=np.float64)
        time_arr = np.arange(steps + 1, dtype=np.float64) * dt

        theta = self._theta0.copy()
        phases[0] = theta
        R_arr[0] = _order_parameter(theta)

        _logger.debug(
            "Starting Kuramoto simulation: N=%d, K=%.4f, dt=%.6f, steps=%d, mode=%s",
            N,
            cfg.K,
            dt,
            steps,
            cfg.coupling_mode,
        )

        for k in range(steps):
            theta = _rk4_step(theta, omega, adj, dt, K=K)
            if not np.isfinite(theta).all():
                raise FloatingPointError(f"Non-finite phase values encountered at step={k + 1}.")
            phases[k + 1] = theta
            R_arr[k + 1] = _order_parameter(theta)

        return KuramotoResult(phases=phases, order_parameter=R_arr, time=time_arr, config=cfg)

    @staticmethod
    def _resolve_initial_conditions(
        cfg: KuramotoConfig,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        rng = np.random.default_rng(cfg.seed)

        omega = (
            cfg.omega.astype(np.float64, copy=False)
            if cfg.omega is not None
            else rng.standard_normal(cfg.N)
        )
        theta0 = (
            cfg.theta0.astype(np.float64, copy=False)
            if cfg.theta0 is not None
            else rng.uniform(0.0, 2.0 * np.pi, cfg.N)
        )
        return omega, theta0

    @staticmethod
    def _resolve_adjacency(cfg: KuramotoConfig) -> NDArray[np.float64]:
        """Resolve the topology-only affinity matrix ``A_norm``.

        FP-1: this returns topology/affinity *without* the scalar gain ``K``.
        ``K`` is the single owner of scale and is applied once in the RHS
        (:func:`_dtheta_dt`); folding it in here would hide a second copy of the
        gain and break boundary/RHS agreement. The diagonal is always zeroed so
        no node self-couples.
        """
        N = cfg.N
        if cfg.adjacency is not None:
            adj = cfg.adjacency.astype(np.float64, copy=True)
            np.fill_diagonal(adj, 0.0)
            return adj

        adj = np.full((N, N), 1.0 / N, dtype=np.float64)
        np.fill_diagonal(adj, 0.0)
        return adj

    @staticmethod
    def _validate_runtime_inputs(
        omega: NDArray[np.float64],
        theta0: NDArray[np.float64],
        adj: NDArray[np.float64],
    ) -> None:
        if omega.ndim != 1 or theta0.ndim != 1 or omega.shape != theta0.shape:
            raise ValueError("Runtime vectors omega and theta0 must be 1-D and same shape.")

        n = omega.shape[0]
        if adj.shape != (n, n):
            raise ValueError(f"Runtime adjacency shape must be {(n, n)}, got {adj.shape}.")

        if (
            not np.isfinite(omega).all()
            or not np.isfinite(theta0).all()
            or not np.isfinite(adj).all()
        ):
            raise ValueError("Runtime inputs must all be finite.")


def _dtheta_dt(
    theta: NDArray[np.float64],
    omega: NDArray[np.float64],
    adj: NDArray[np.float64],
    *,
    K: float = 1.0,
) -> NDArray[np.float64]:
    """Evaluate Kuramoto RHS for a single phase vector.

    FP-1: ``K`` is the single scalar coupling gain and is applied here, once,
    over the topology-only affinity ``adj`` (``A_norm``). ``K`` defaults to
    ``1.0`` so callers that pass an already-effective coupling matrix (e.g. a
    Ricci-derived or pre-scaled matrix) get identity scaling and unchanged
    dynamics.
    """
    diff = theta[np.newaxis, :] - theta[:, np.newaxis]
    sin_diff = np.sin(diff)
    coupling = K * (adj * sin_diff).sum(axis=1)
    out: NDArray[np.float64] = omega + coupling
    if not np.isfinite(out).all():
        raise FloatingPointError("Non-finite derivative values produced by Kuramoto RHS.")
    return out


def _rk4_step(
    theta: NDArray[np.float64],
    omega: NDArray[np.float64],
    adj: NDArray[np.float64],
    dt: float,
    *,
    K: float = 1.0,
) -> NDArray[np.float64]:
    """Advance one Runge-Kutta 4 integration step (single ``K`` owner)."""
    k1 = _dtheta_dt(theta, omega, adj, K=K)
    k2 = _dtheta_dt(theta + 0.5 * dt * k1, omega, adj, K=K)
    k3 = _dtheta_dt(theta + 0.5 * dt * k2, omega, adj, K=K)
    k4 = _dtheta_dt(theta + dt * k3, omega, adj, K=K)
    return theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _order_parameter(theta: NDArray[np.float64]) -> float:
    """Compute Kuramoto order parameter ``R = |mean(exp(iθ))|``."""
    z = np.exp(1j * theta).mean()
    # INV-K1: R = |mean(exp(iθ))| ∈ [0,1] by Cauchy-Schwarz
    return float(np.clip(np.abs(z), 0.0, 1.0))


def run_simulation(config: KuramotoConfig) -> KuramotoResult:
    """Convenience one-shot entry point around :class:`KuramotoEngine`."""
    return KuramotoEngine(config).run()
