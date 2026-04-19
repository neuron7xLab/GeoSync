# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Second-order Kuramoto model with inertia and damping.

The classical Kuramoto model is first-order (overdamped).  For power
grid stability analysis, generator rotors have physical inertia:

    m_i · θ̈_i + d_i · θ̇_i = ω_i + K · Σ_j A_ij · sin(θ_j - θ_i)

where:
    m_i = inertia (moment of inertia / angular momentum coefficient)
    d_i = damping (friction / governor response)
    ω_i = natural frequency (power injection - demand)

This is equivalent to the swing equation in power systems:

    M_i · δ̈_i + D_i · δ̇_i = P_m,i - P_e,i(δ)

The second-order system exhibits richer dynamics: oscillatory
transients, frequency nadir, rate of change of frequency (RoCoF),
and inertia-dependent stability margins.

Usage::

    from core.kuramoto.second_order import SecondOrderKuramotoEngine
    config = KuramotoConfig(N=50, K=5.0, dt=0.005, steps=10000)
    engine = SecondOrderKuramotoEngine(config, mass=1.0, damping=0.1)
    result = engine.run()
    print(result.summary)
    # Access velocity: result.metadata["velocities"]  # (steps+1, N)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import KuramotoConfig
from .engine import KuramotoResult, _order_parameter

__all__ = ["SecondOrderKuramotoEngine", "SecondOrderResult"]

_logger = logging.getLogger(__name__)


@dataclass
class SecondOrderResult:
    """Extended result wrapping KuramotoResult with velocity trajectories."""

    base: KuramotoResult
    velocities: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.velocities.size > 0:
            self.base.summary.update(self._compute_frequency_metrics())

    @property
    def phases(self) -> NDArray[np.float64]:
        return self.base.phases

    @property
    def order_parameter(self) -> NDArray[np.float64]:
        return self.base.order_parameter

    @property
    def time(self) -> NDArray[np.float64]:
        return self.base.time

    @property
    def config(self) -> KuramotoConfig:
        return self.base.config

    @property
    def summary(self) -> dict[str, Any]:
        return self.base.summary

    def _compute_frequency_metrics(self) -> dict[str, Any]:
        """Power-grid relevant metrics from velocity trajectories."""
        v = self.velocities
        dt = self.base.config.dt
        return {
            "frequency_nadir": float(np.min(v)),
            "frequency_zenith": float(np.max(v)),
            "max_rocof": float(np.max(np.abs(np.diff(v, axis=0) / dt))),
            "final_frequency_spread": float(np.std(v[-1])),
            "mean_frequency": float(np.mean(v[-1])),
            "settling_time_95pct": self._settling_time(v),
        }

    def _settling_time(self, v: NDArray[np.float64], threshold: float = 0.05) -> float:
        """Time for frequency spread to settle within threshold of final value."""
        t = self.base.time
        final_spread = np.std(v[-1])
        for k in range(len(v) - 1, -1, -1):
            if np.std(v[k]) > final_spread + threshold:
                return float(t[min(k + 1, len(t) - 1)])
        return 0.0


class SecondOrderKuramotoEngine:
    """Second-order Kuramoto (swing equation) integrator.

    Parameters
    ----------
    config : KuramotoConfig
        Standard config.
    mass : float | NDArray
        Inertia coefficient(s).  Scalar for uniform, array(N) for heterogeneous.
    damping : float | NDArray
        Damping coefficient(s).  Scalar or array(N).
    velocity0 : NDArray | None
        Initial angular velocities.  Default: zeros (standstill).
    """

    def __init__(
        self,
        config: KuramotoConfig,
        mass: float | NDArray[np.float64] = 1.0,
        damping: float | NDArray[np.float64] = 0.1,
        velocity0: NDArray[np.float64] | None = None,
    ) -> None:
        self._cfg = config
        N = config.N
        self._omega, self._theta0 = self._resolve_ic(config)
        self._adj = self._resolve_adj(config)

        # Broadcast mass and damping to per-oscillator arrays
        self._mass = np.broadcast_to(np.asarray(mass, dtype=np.float64), (N,)).copy()
        self._damping = np.broadcast_to(np.asarray(damping, dtype=np.float64), (N,)).copy()

        if np.any(self._mass <= 0):
            raise ValueError("Mass must be strictly positive for all oscillators.")
        if np.any(self._damping < 0):
            raise ValueError("Damping must be non-negative for all oscillators.")

        self._v0 = velocity0.copy() if velocity0 is not None else np.zeros(N, dtype=np.float64)

        _logger.info(
            "SecondOrderKuramotoEngine: N=%d, K=%.4f, m=[%.3f, %.3f], d=[%.3f, %.3f]",
            N,
            config.K,
            float(np.min(self._mass)),
            float(np.max(self._mass)),
            float(np.min(self._damping)),
            float(np.max(self._damping)),
        )

    def run(self) -> SecondOrderResult:
        """Integrate second-order system using Störmer-Verlet (symplectic)."""
        cfg = self._cfg
        N = cfg.N
        steps = cfg.steps
        dt = cfg.dt

        phases = np.empty((steps + 1, N), dtype=np.float64)
        velocities = np.empty((steps + 1, N), dtype=np.float64)
        R_arr = np.empty(steps + 1, dtype=np.float64)
        time_arr = np.arange(steps + 1, dtype=np.float64) * dt

        theta = self._theta0.copy()
        v = self._v0.copy()
        phases[0] = theta
        velocities[0] = v
        R_arr[0] = _order_parameter(theta)

        m = self._mass
        d = self._damping
        inv_m = 1.0 / m

        for k in range(steps):
            # Compute acceleration: a = (1/m)(ω + coupling - d·v)
            accel = inv_m * (self._omega + self._coupling(theta) - d * v)

            # Velocity Verlet integration (symplectic, energy-conserving)
            theta_new = theta + v * dt + 0.5 * accel * dt * dt
            accel_new = inv_m * (self._omega + self._coupling(theta_new) - d * v)
            v_new = v + 0.5 * (accel + accel_new) * dt

            theta = theta_new
            v = v_new

            phases[k + 1] = theta
            velocities[k + 1] = v
            R_arr[k + 1] = _order_parameter(theta)

        base = KuramotoResult(
            phases=phases,
            order_parameter=R_arr,
            time=time_arr,
            config=cfg,
        )
        return SecondOrderResult(base=base, velocities=velocities)

    def _coupling(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute coupling forces Σ_j A_ij sin(θ_j - θ_i)."""
        diff = theta[np.newaxis, :] - theta[:, np.newaxis]
        return (self._adj * np.sin(diff)).sum(axis=1)

    @staticmethod
    def _resolve_ic(cfg: KuramotoConfig) -> tuple[NDArray, NDArray]:
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
    def _resolve_adj(cfg: KuramotoConfig) -> NDArray:
        N, K = cfg.N, cfg.K
        if cfg.adjacency is not None:
            adj = K * cfg.adjacency.astype(np.float64, copy=True)
            np.fill_diagonal(adj, 0.0)
            return adj
        adj = np.full((N, N), K / N, dtype=np.float64)
        np.fill_diagonal(adj, 0.0)
        return adj
