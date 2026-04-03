# SPDX-License-Identifier: LicenseRef-TradePulse-Proprietary
"""Kuramoto ODE simulation engine.

Implements a vectorised, numerically stable 4th-order Runge-Kutta (RK4)
integrator for the Kuramoto model of coupled phase oscillators:

    Global coupling (fully connected):
        dθᵢ/dt = ωᵢ + (K/N) · Σⱼ sin(θⱼ − θᵢ)

    Weighted adjacency coupling:
        dθᵢ/dt = ωᵢ + K · Σⱼ Aᵢⱼ sin(θⱼ − θᵢ)

The engine is deterministic and reproducible for a given :class:`KuramotoConfig`.

Typical use
-----------
>>> from core.kuramoto import KuramotoConfig, run_simulation
>>> cfg = KuramotoConfig(N=50, K=2.0, dt=0.01, steps=500, seed=0)
>>> result = run_simulation(cfg)
>>> print(f"Final R = {result.order_parameter[-1]:.4f}")
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


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class KuramotoResult:
    """Typed container for the output of a Kuramoto simulation.

    Attributes
    ----------
    phases:
        Phase trajectories, shape ``(steps + 1, N)`` in radians.
        Row 0 is the initial condition.
    order_parameter:
        Kuramoto order parameter R(t), shape ``(steps + 1,)``.
        R ∈ [0, 1]: 0 → fully desynchronised, 1 → fully synchronised.
    time:
        Time axis, shape ``(steps + 1,)``.  ``time[k] = k * dt``.
    config:
        The :class:`KuramotoConfig` that produced this result.
    summary:
        Dictionary of scalar summary statistics.
    """

    phases: NDArray[np.float64]
    order_parameter: NDArray[np.float64]
    time: NDArray[np.float64]
    config: KuramotoConfig
    summary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.summary = self._compute_summary()

    def _compute_summary(self) -> dict[str, Any]:
        R = self.order_parameter
        return {
            "final_R": float(R[-1]),
            "mean_R": float(R.mean()),
            "max_R": float(R.max()),
            "min_R": float(R.min()),
            "std_R": float(R.std()),
            "N": self.config.N,
            "K": self.config.K,
            "dt": self.config.dt,
            "steps": self.config.steps,
            "total_time": float(self.time[-1]),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Core engine
# ──────────────────────────────────────────────────────────────────────────────


class KuramotoEngine:
    """Deterministic Kuramoto ODE solver using 4th-order Runge-Kutta.

    Parameters
    ----------
    config:
        Validated simulation configuration.  Use :class:`KuramotoConfig` to
        construct and validate parameters before passing them here.

    Examples
    --------
    >>> from core.kuramoto import KuramotoConfig, KuramotoEngine
    >>> cfg = KuramotoConfig(N=10, K=1.5, dt=0.05, steps=200, seed=7)
    >>> engine = KuramotoEngine(cfg)
    >>> result = engine.run()
    >>> result.summary["final_R"]
    """

    def __init__(self, config: KuramotoConfig) -> None:
        self._cfg = config
        self._omega, self._theta0 = self._resolve_initial_conditions(config)
        self._adj = self._resolve_adjacency(config, self._omega)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> KuramotoResult:
        """Execute the simulation and return a :class:`KuramotoResult`.

        Returns
        -------
        KuramotoResult
            Phase trajectories, order parameter time-series, time axis, and
            summary statistics.
        """
        cfg = self._cfg
        N = cfg.N
        steps = cfg.steps
        dt = cfg.dt
        omega = self._omega
        adj = self._adj

        # Pre-allocate output arrays
        phases = np.empty((steps + 1, N), dtype=np.float64)
        R_arr = np.empty(steps + 1, dtype=np.float64)
        time_arr = np.arange(steps + 1, dtype=np.float64) * dt

        theta = self._theta0.copy()
        phases[0] = theta
        R_arr[0] = _order_parameter(theta)

        _logger.debug(
            "Starting Kuramoto simulation: N=%d, K=%.4f, dt=%.6f, steps=%d",
            N,
            cfg.K,
            dt,
            steps,
        )

        for k in range(steps):
            theta = _rk4_step(theta, omega, adj, dt)
            phases[k + 1] = theta
            R_arr[k + 1] = _order_parameter(theta)

        _logger.debug("Simulation complete. Final R=%.4f", R_arr[-1])

        return KuramotoResult(
            phases=phases,
            order_parameter=R_arr,
            time=time_arr,
            config=cfg,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_initial_conditions(
        cfg: KuramotoConfig,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Resolve ``omega`` and ``theta0``, drawing randomly if needed."""
        rng = np.random.default_rng(cfg.seed)

        if cfg.omega is not None:
            omega = cfg.omega.astype(np.float64, copy=False)
        else:
            omega = rng.standard_normal(cfg.N)

        if cfg.theta0 is not None:
            theta0 = cfg.theta0.astype(np.float64, copy=False)
        else:
            theta0 = rng.uniform(0.0, 2.0 * np.pi, cfg.N)

        return omega, theta0

    @staticmethod
    def _resolve_adjacency(
        cfg: KuramotoConfig,
        omega: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Build the effective coupling matrix used at each integration step.

        For fully-connected topology the matrix is ``K/N * (ones - I)``.
        For an explicit adjacency matrix it is ``K * A``.
        """
        N = cfg.N
        K = cfg.K
        if cfg.adjacency is not None:
            return K * cfg.adjacency.astype(np.float64, copy=False)
        # All-to-all coupling (no self-coupling on diagonal)
        adj = np.full((N, N), K / N, dtype=np.float64)
        np.fill_diagonal(adj, 0.0)
        return adj


# ──────────────────────────────────────────────────────────────────────────────
# Vectorised inner-loop helpers (module-level for numba-friendliness)
# ──────────────────────────────────────────────────────────────────────────────


def _dtheta_dt(
    theta: NDArray[np.float64],
    omega: NDArray[np.float64],
    adj: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the Kuramoto RHS: dθ/dt = ω + Σⱼ Aᵢⱼ sin(θⱼ − θᵢ).

    Parameters
    ----------
    theta:
        Current phases, shape ``(N,)``.
    omega:
        Natural frequencies, shape ``(N,)``.
    adj:
        Effective coupling matrix (already scaled by K), shape ``(N, N)``.

    Returns
    -------
    NDArray[np.float64]
        Phase derivatives, shape ``(N,)``.
    """
    # diff[i, j] = θⱼ − θᵢ  →  sin_diff[i, j] = sin(θⱼ − θᵢ)
    diff = theta[np.newaxis, :] - theta[:, np.newaxis]  # (N, N)
    sin_diff = np.sin(diff)
    coupling = (adj * sin_diff).sum(axis=1)  # (N,)
    return omega + coupling


def _rk4_step(
    theta: NDArray[np.float64],
    omega: NDArray[np.float64],
    adj: NDArray[np.float64],
    dt: float,
) -> NDArray[np.float64]:
    """Advance phases by one RK4 step.

    Parameters
    ----------
    theta:
        Current phases, shape ``(N,)``.
    omega:
        Natural frequencies, shape ``(N,)``.
    adj:
        Effective coupling matrix, shape ``(N, N)``.
    dt:
        Integration step size.

    Returns
    -------
    NDArray[np.float64]
        Updated phases, shape ``(N,)``.
    """
    k1 = _dtheta_dt(theta, omega, adj)
    k2 = _dtheta_dt(theta + 0.5 * dt * k1, omega, adj)
    k3 = _dtheta_dt(theta + 0.5 * dt * k2, omega, adj)
    k4 = _dtheta_dt(theta + dt * k3, omega, adj)
    return theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _order_parameter(theta: NDArray[np.float64]) -> float:
    """Kuramoto order parameter R = |mean(exp(iθ))|.

    Parameters
    ----------
    theta:
        Phase array, shape ``(N,)``.

    Returns
    -------
    float
        Order parameter in [0, 1].
    """
    z = np.exp(1j * theta).mean()
    return float(np.clip(np.abs(z), 0.0, 1.0))


# ──────────────────────────────────────────────────────────────────────────────
# Convenience one-shot entry point
# ──────────────────────────────────────────────────────────────────────────────


def run_simulation(config: KuramotoConfig) -> KuramotoResult:
    """Run a Kuramoto simulation from a validated configuration.

    This is the recommended entry point for programmatic use.

    Parameters
    ----------
    config:
        Validated :class:`KuramotoConfig` instance.

    Returns
    -------
    KuramotoResult
        Full simulation output including phase trajectories, order parameter
        time-series, time axis, and summary statistics.

    Examples
    --------
    >>> from core.kuramoto import KuramotoConfig, run_simulation
    >>> cfg = KuramotoConfig(N=20, K=3.0, dt=0.01, steps=500, seed=1)
    >>> result = run_simulation(cfg)
    >>> 0.0 <= result.order_parameter[-1] <= 1.0
    True
    """
    return KuramotoEngine(config).run()
