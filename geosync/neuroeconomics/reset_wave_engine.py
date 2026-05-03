"""Damped phase synchronization system on compact phase manifold.

FACT: dtheta/dt = K * sin(theta* - theta)
MODEL: decentralized reset-wave realignment
ANALOGY: homeostatic stabilizer metaphor
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def wrap_phase(theta: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return ((theta + math.pi) % (2.0 * math.pi)) - math.pi


def phase_distance(a: float, b: float) -> float:
    """Shortest signed angular distance b-a on circle."""
    return wrap_phase(b - a)


def phase_alignment_potential(node_phases: list[float], baseline_phases: list[float]) -> float:
    diffs = [phase_distance(n, b) for n, b in zip(node_phases, baseline_phases)]
    return sum(1.0 - math.cos(d) for d in diffs) / len(diffs)


@dataclass(frozen=True, slots=True)
class ResetWaveConfig:
    coupling_gain: float = 1.0
    dt: float = 0.1
    steps: int = 32
    max_phase_error: float = math.pi
    convergence_tol: float = 0.05
    integrator: str = "rk4_fixed"


@dataclass(frozen=True, slots=True)
class ResetWaveState:
    step: int
    mean_phase_error: float
    phase_alignment_potential: float


@dataclass(frozen=True, slots=True)
class ResetWaveResult:
    converged: bool
    locked: bool
    initial_potential: float
    final_potential: float
    trajectory: tuple[ResetWaveState, ...]


def run_reset_wave(
    node_phases: list[float], baseline_phases: list[float], cfg: ResetWaveConfig
) -> ResetWaveResult:
    if len(node_phases) != len(baseline_phases):
        raise ValueError("node_phases and baseline_phases must have equal length")
    if not node_phases:
        raise ValueError("at least one node phase is required")
    if cfg.coupling_gain <= 0 or cfg.dt <= 0 or cfg.steps <= 0 or cfg.max_phase_error <= 0:
        raise ValueError("invalid reset-wave configuration")
    if cfg.integrator not in {"euler", "rk4_fixed"}:
        raise ValueError("integrator must be one of: euler, rk4_fixed")

    phases = [wrap_phase(p) for p in node_phases]
    baseline = [wrap_phase(p) for p in baseline_phases]
    diffs = [abs(phase_distance(n, b)) for n, b in zip(phases, baseline)]
    if any(d > cfg.max_phase_error for d in diffs):
        p0 = phase_alignment_potential(phases, baseline)
        return ResetWaveResult(
            False, True, p0, p0, (ResetWaveState(0, sum(diffs) / len(diffs), p0),)
        )

    states: list[ResetWaveState] = []
    p0 = phase_alignment_potential(phases, baseline)
    for step in range(cfg.steps + 1):
        diffs_signed = [phase_distance(n, b) for n, b in zip(phases, baseline)]
        mean_err = sum(abs(d) for d in diffs_signed) / len(diffs_signed)
        potential = phase_alignment_potential(phases, baseline)
        states.append(ResetWaveState(step, mean_err, potential))
        if mean_err <= cfg.convergence_tol:
            break

        if cfg.integrator == "euler":
            for i, d in enumerate(diffs_signed):
                phases[i] = wrap_phase(phases[i] + cfg.dt * cfg.coupling_gain * math.sin(d))
        else:
            for i, x in enumerate(phases):
                target = baseline[i]

                def f(v: float, target: float = target) -> float:
                    return cfg.coupling_gain * math.sin(phase_distance(v, target))

                k1 = f(x)
                k2 = f(wrap_phase(x + 0.5 * cfg.dt * k1))
                k3 = f(wrap_phase(x + 0.5 * cfg.dt * k2))
                k4 = f(wrap_phase(x + cfg.dt * k3))
                phases[i] = wrap_phase(x + (cfg.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))

    return ResetWaveResult(
        converged=states[-1].mean_phase_error <= cfg.convergence_tol,
        locked=False,
        initial_potential=p0,
        final_potential=states[-1].phase_alignment_potential,
        trajectory=tuple(states),
    )


def latent_interpretive_forecast_layer(result: ResetWaveResult) -> tuple[str, float]:
    """trajectory -> features -> regime_prediction -> confidence."""
    if result.locked:
        return ("LOCKED", 1.0)
    if len(result.trajectory) < 2:
        return ("UNKNOWN", 0.5)
    init = result.initial_potential
    final = result.final_potential
    slope = final - init
    if result.converged:
        return ("CONVERGING", 0.9)
    if slope > 1e-6:
        return ("DIVERGING", 0.8)
    if abs(slope) < 1e-4:
        return ("OSCILLATORY", 0.6)
    return ("UNSTABLE", 0.7)
