# SPDX-License-Identifier: MIT
"""T2 — Explosive Synchronization Proximity as Crisis Early-Warning.

Explosive synchronization (ES) = first-order (discontinuous) phase transition
in the order parameter R, as opposed to the smooth second-order transition
in classical Kuramoto.

Detection method (Kim et al. PNAS 2025 framework):
    1. Sweep coupling K from K_low to K_high (forward) and back (backward)
    2. Measure R(K) in both directions
    3. Hysteresis width = K_c_forward - K_c_backward
    4. ES proximity = hysteresis_width / K_range

Signal interpretation:
    R(t) ↑ + hysteresis_width ↑  =  pre-crisis (system near explosive transition)
    R(t) stable + width ≈ 0      =  normal (smooth transition)

Integration: circuit breaker in Risk Manager.
When ES proximity exceeds threshold → escalate FailSafe to RESTRICTED.

References:
    Gómez-Gardeñes et al. "Explosive synchronization transitions" PRL (2011)
    Kim et al. "Explosive synchronization in complex networks" PNAS (2025)
    D'Souza et al. "Explosive phenomena in complex networks" Adv. Phys. (2019)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ESProximityResult:
    """Result of explosive synchronization proximity measurement."""

    R_forward: NDArray[np.float64]   # R(K) forward sweep
    R_backward: NDArray[np.float64]  # R(K) backward sweep
    K_values: NDArray[np.float64]    # coupling values swept
    K_c_forward: float               # critical K (forward)
    K_c_backward: float              # critical K (backward)
    hysteresis_width: float          # K_c_forward - K_c_backward
    proximity: float                 # normalised proximity metric [0, 1]
    is_explosive: bool               # True if significant hysteresis detected


class ExplosiveSyncDetector:
    """Detect proximity to explosive (first-order) synchronization transition.

    Parameters
    ----------
    K_range : tuple[float, float]
        Range of coupling strengths to sweep (default (0.1, 5.0)).
    n_K_steps : int
        Number of coupling values in sweep (default 20).
    kuramoto_steps : int
        Integration steps per K value (default 300).
    R_threshold : float
        Order parameter threshold for "synchronized" (default 0.5).
    hysteresis_threshold : float
        Minimum hysteresis width to declare ES (default 0.3).
    """

    def __init__(
        self,
        K_range: tuple[float, float] = (0.1, 5.0),
        n_K_steps: int = 20,
        kuramoto_steps: int = 300,
        R_threshold: float = 0.5,
        hysteresis_threshold: float = 0.3,
    ) -> None:
        if K_range[0] >= K_range[1]:
            raise ValueError(f"K_range must be (low, high), got {K_range}")
        if n_K_steps < 2:
            raise ValueError(f"n_K_steps must be ≥ 2, got {n_K_steps}")
        self._K_range = K_range
        self._n_K = n_K_steps
        self._steps = kuramoto_steps
        self._R_thresh = R_threshold
        self._hyst_thresh = hysteresis_threshold

    def measure_proximity(
        self,
        adjacency: NDArray[np.float64] | None = None,
        omega: NDArray[np.float64] | None = None,
        N: int = 10,
        seed: int = 42,
    ) -> ESProximityResult:
        """Sweep K forward and backward, measure hysteresis.

        Parameters
        ----------
        adjacency : optional (N, N) adjacency matrix.
        omega : optional (N,) natural frequencies.
        N : int, number of oscillators if omega not given.
        seed : int, RNG seed for reproducibility.

        Returns
        -------
        ESProximityResult with hysteresis analysis.
        """
        from core.kuramoto.config import KuramotoConfig
        from core.kuramoto.engine import KuramotoEngine

        K_values = np.linspace(self._K_range[0], self._K_range[1], self._n_K)

        # Use consistent initial conditions for both sweeps
        rng = np.random.default_rng(seed)
        if omega is None:
            omega = rng.standard_normal(N)
        else:
            N = omega.shape[0]

        theta0_base = rng.uniform(0, 2 * np.pi, N)

        R_forward = np.zeros(self._n_K)
        R_backward = np.zeros(self._n_K)

        # Forward sweep: increasing K, carry final phases forward
        theta_carry = theta0_base.copy()
        for i, K in enumerate(K_values):
            cfg = KuramotoConfig(
                N=N, K=K, omega=omega, adjacency=adjacency,
                theta0=theta_carry, dt=0.01, steps=self._steps, seed=seed,
            )
            result = KuramotoEngine(cfg).run()
            R_forward[i] = result.order_parameter[-1]
            theta_carry = result.phases[-1].copy()

        # Backward sweep: decreasing K, carry final phases backward
        theta_carry_back = theta_carry.copy()
        for i, K in enumerate(reversed(K_values)):
            cfg = KuramotoConfig(
                N=N, K=K, omega=omega, adjacency=adjacency,
                theta0=theta_carry_back, dt=0.01, steps=self._steps, seed=seed,
            )
            result = KuramotoEngine(cfg).run()
            R_backward[self._n_K - 1 - i] = result.order_parameter[-1]
            theta_carry_back = result.phases[-1].copy()

        # Find critical K values
        K_c_fwd = self._find_critical_K(K_values, R_forward)
        K_c_bwd = self._find_critical_K(K_values, R_backward)

        hysteresis = abs(K_c_fwd - K_c_bwd)
        K_span = self._K_range[1] - self._K_range[0]
        proximity = min(hysteresis / K_span, 1.0)

        return ESProximityResult(
            R_forward=R_forward,
            R_backward=R_backward,
            K_values=K_values,
            K_c_forward=K_c_fwd,
            K_c_backward=K_c_bwd,
            hysteresis_width=hysteresis,
            proximity=proximity,
            is_explosive=hysteresis > self._hyst_thresh,
        )

    def _find_critical_K(
        self,
        K_values: NDArray[np.float64],
        R_values: NDArray[np.float64],
    ) -> float:
        """Find K_c where R crosses threshold."""
        crossings = np.where(
            (R_values[:-1] < self._R_thresh) & (R_values[1:] >= self._R_thresh)
        )[0]
        if crossings.size > 0:
            idx = crossings[0]
            # Linear interpolation
            frac = (self._R_thresh - R_values[idx]) / max(
                R_values[idx + 1] - R_values[idx], 1e-12
            )
            return float(K_values[idx] + frac * (K_values[idx + 1] - K_values[idx]))
        # If no crossing found, return endpoint
        if R_values[-1] >= self._R_thresh:
            return float(K_values[0])
        return float(K_values[-1])

    def crisis_signal(
        self,
        prices: NDArray[np.float64],
        window: int = 60,
        correlation_threshold: float = 0.3,
    ) -> ESProximityResult:
        """Compute ES proximity from price data.

        Builds correlation-based adjacency from rolling returns,
        then measures hysteresis.
        """
        prices = np.asarray(prices, dtype=np.float64)
        if prices.ndim != 2 or prices.shape[0] < window:
            raise ValueError(
                f"Need (T≥{window}, N) array, got {prices.shape}"
            )

        returns = np.diff(prices, axis=0) / np.maximum(np.abs(prices[:-1]), 1e-12)
        tail = returns[-window:]
        N = prices.shape[1]

        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(tail, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)

        # Build adjacency
        adj = np.abs(corr)
        adj[adj < correlation_threshold] = 0.0
        np.fill_diagonal(adj, 0.0)

        # Use return-derived natural frequencies
        omega = np.mean(tail, axis=0) * 100  # scaled for Kuramoto dynamics

        return self.measure_proximity(adjacency=adj, omega=omega, N=N)


class ESCircuitBreaker:
    """Circuit breaker that triggers on explosive sync proximity.

    Parameters
    ----------
    proximity_threshold : float
        ES proximity above which to trigger (default 0.15).
    cooldown_steps : int
        Steps to wait before re-arming after trigger (default 10).
    """

    def __init__(
        self,
        proximity_threshold: float = 0.15,
        cooldown_steps: int = 10,
    ) -> None:
        if not 0 < proximity_threshold < 1:
            raise ValueError(f"threshold must be in (0, 1), got {proximity_threshold}")
        self._threshold = proximity_threshold
        self._cooldown = cooldown_steps
        self._triggered = False
        self._cooldown_remaining = 0
        self._trigger_count = 0

    @property
    def is_triggered(self) -> bool:
        return self._triggered

    @property
    def trigger_count(self) -> int:
        return self._trigger_count

    def check(self, proximity: float) -> bool:
        """Check if circuit breaker should trigger.

        Returns True if trading should be HALTED.
        """
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            if self._cooldown_remaining == 0:
                self._triggered = False
            return self._triggered

        if proximity > self._threshold:
            self._triggered = True
            self._cooldown_remaining = self._cooldown
            self._trigger_count += 1
            return True

        self._triggered = False
        return False

    def reset(self) -> None:
        """Reset circuit breaker state."""
        self._triggered = False
        self._cooldown_remaining = 0


__all__ = [
    "ExplosiveSyncDetector",
    "ESProximityResult",
    "ESCircuitBreaker",
]
