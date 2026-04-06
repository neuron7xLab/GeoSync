"""Serotonin ODE System with Lyapunov Stability.

Implements a biologically-grounded two-state ODE for serotonin (5-HT)
dynamics including receptor desensitisation, solved with 4th-order
Runge-Kutta and equipped with a Lyapunov stability certificate.

State variables:
    level          — 5-HT concentration (∈ [0, 1])
    desensitization — receptor adaptation (∈ [0, ∞), practically bounded)

ODE system:
    d(level)/dt  = -α·level + β·stress + γ·(baseline - level) - δ·desens
    d(desens)/dt =  η·max(0, level - threshold) - μ·desens

Lyapunov function:
    V(level, desens) = 0.5·(level - target)² + λ·desens²
    dV/dt < 0  ⟹  asymptotic stability toward (target, 0)

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SerotoninODEParams:
    """Default parameters for the serotonin ODE system."""

    alpha: float = 0.1  # 5-HT decay rate
    beta: float = 0.3  # stress → 5-HT production rate
    gamma: float = 0.05  # homeostatic pull toward baseline
    delta: float = 0.02  # desensitisation → suppression of 5-HT
    eta: float = 0.01  # 5-HT above threshold → desensitisation
    mu: float = 0.005  # desensitisation recovery rate
    baseline: float = 0.3  # homeostatic target for 5-HT level
    threshold: float = 0.5  # level above which desens increases
    target: float = 0.3  # Lyapunov target level
    lambda_: float = 0.5  # Lyapunov weight on desens term


class SerotoninODE:
    """Two-state ODE for serotonin dynamics with RK4 integration.

    Parameters
    ----------
    params : SerotoninODEParams | None
        ODE parameters.  Uses defaults if *None*.
    level : float
        Initial 5-HT concentration (default: baseline).
    desensitization : float
        Initial receptor desensitisation (default 0.0).
    """

    def __init__(
        self,
        params: SerotoninODEParams | None = None,
        level: float | None = None,
        desensitization: float = 0.0,
    ) -> None:
        self.p = params or SerotoninODEParams()
        self.level = level if level is not None else self.p.baseline
        self.desensitization = desensitization

    # ── ODE right-hand side ──────────────────────────────────────────

    def _derivatives(
        self,
        level: float,
        desens: float,
        stress: float,
    ) -> tuple[float, float]:
        """Compute (d_level/dt, d_desens/dt).

        The level equation combines decay and homeostatic pull into a
        single restoring force so that the equilibrium under zero stress
        and zero desensitisation is exactly ``baseline``::

            d(level)/dt = -(alpha + gamma) * (level - baseline)
                          + beta * stress
                          - delta * desens
        """
        p = self.p
        d_level = (
            -(p.alpha + p.gamma) * (level - p.baseline)
            + p.beta * stress
            - p.delta * desens
        )
        d_desens = p.eta * max(0.0, level - p.threshold) - p.mu * desens
        return d_level, d_desens

    # ── RK4 integration step ─────────────────────────────────────────

    def step(self, stress: float, dt: float = 1.0) -> tuple[float, float]:
        """Advance the ODE by *dt* using 4th-order Runge-Kutta.

        Parameters
        ----------
        stress : float
            Current stress input (non-negative).
        dt : float
            Integration time step.

        Returns
        -------
        tuple[float, float]
            (level, desensitization) after the step.
        """
        y1, y2 = self.level, self.desensitization

        k1a, k1b = self._derivatives(y1, y2, stress)
        k2a, k2b = self._derivatives(y1 + 0.5 * dt * k1a, y2 + 0.5 * dt * k1b, stress)
        k3a, k3b = self._derivatives(y1 + 0.5 * dt * k2a, y2 + 0.5 * dt * k2b, stress)
        k4a, k4b = self._derivatives(y1 + dt * k3a, y2 + dt * k3b, stress)

        self.level = y1 + (dt / 6.0) * (k1a + 2 * k2a + 2 * k3a + k4a)
        self.desensitization = y2 + (dt / 6.0) * (k1b + 2 * k2b + 2 * k3b + k4b)

        # Clamp level to [0, 1] for biological plausibility
        self.level = max(
            0.0, min(1.0, self.level)
        )  # INV-5HT2: s(t) ∈ [0,1] — biological 5-HT range
        # Desensitization is non-negative
        self.desensitization = max(
            0.0, self.desensitization
        )  # INV-5HT4: desensitization non-negative — receptor density lower bound

        return self.level, self.desensitization

    # ── Lyapunov stability verification ──────────────────────────────

    def _lyapunov(self, level: float, desens: float) -> float:
        """Compute Lyapunov function V(level, desens)."""
        p = self.p
        return 0.5 * (level - p.target) ** 2 + p.lambda_ * desens**2

    def verify_lyapunov(
        self,
        trajectory: list[tuple[float, float]],
    ) -> bool:
        """Verify Lyapunov stability along a trajectory.

        Checks that V(t+1) < V(t) for all consecutive pairs, which
        certifies monotonic energy decrease (asymptotic stability).

        Parameters
        ----------
        trajectory : list[tuple[float, float]]
            Sequence of (level, desensitization) states.

        Returns
        -------
        bool
            True if dV/dt < 0 along the entire trajectory (within
            numerical tolerance).
        """
        if len(trajectory) < 2:
            return True

        eps = 1e-12  # numerical tolerance
        prev_v = self._lyapunov(*trajectory[0])
        for state in trajectory[1:]:
            curr_v = self._lyapunov(*state)
            if curr_v > prev_v + eps:
                return False
            prev_v = curr_v
        return True
