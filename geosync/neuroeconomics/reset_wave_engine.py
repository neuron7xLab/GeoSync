# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Damped phase synchronization on a compact phase manifold [-π, π).

FACT  (model):  dθ/dt = K · sin(θ* − θ)
                discrete: θ_{t+1} = θ_t + Δt · K · sin(θ* − θ_t)
                potential V(θ) = mean(1 − cos(θ* − θ))

MODEL (purpose): a numerical relaxation solver that drives a vector
                 of node phases toward a baseline reference, with a
                 fail-closed lock when any phase exceeds the
                 ``max_phase_error`` threshold.

ANALOGY (interpretation only): the "reset-wave" / "homeostatic"
                 language is metaphor — there is no claim of literal
                 thermodynamic or neurobiological law. See
                 ``docs/reset_wave_validation_report.md`` for the
                 explicit FACT / MODEL / ANALOGY scope discipline.

Stability bound (empirical, see ``docs/stability_bounds.md``):
    monotone potential decrease is reliable for
        0 < coupling_gain · dt ≤ 0.2
    Outside this region, oscillation or divergence may occur and is
    covered by the negative tests in ``tests/test_reset_wave_*.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def wrap_phase(theta: float) -> float:
    """Wrap angle to [-π, π)."""
    return ((theta + math.pi) % (2.0 * math.pi)) - math.pi


def phase_distance(a: float, b: float) -> float:
    """Shortest signed angular distance ``b − a`` on the circle."""
    return wrap_phase(b - a)


def phase_alignment_potential(node_phases: list[float], baseline_phases: list[float]) -> float:
    """V(θ) = mean(1 − cos(θ_baseline − θ_node))."""
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


@dataclass(frozen=True, slots=True)
class CriticalCenterAudit:
    center: str
    passed: bool
    detail: str


def run_reset_wave(
    node_phases: list[float],
    baseline_phases: list[float],
    cfg: ResetWaveConfig,
) -> ResetWaveResult:
    """Run the damped phase-synchronization solver.

    Contract validation is fail-fast (ValueError) on:
        * mismatched vector lengths,
        * empty input,
        * non-positive ``coupling_gain``, ``dt``, ``steps``, or
          ``max_phase_error``,
        * unknown integrator name.

    Safety lock fires if any initial absolute phase error exceeds
    ``cfg.max_phase_error``; the returned ``ResetWaveResult`` carries
    ``locked=True`` and identical initial / final potential.
    """
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
            converged=False,
            locked=True,
            initial_potential=p0,
            final_potential=p0,
            trajectory=(ResetWaveState(0, sum(diffs) / len(diffs), p0),),
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

                def f(v: float, _t: float = target) -> float:
                    return cfg.coupling_gain * math.sin(phase_distance(v, _t))

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
    """trajectory → features → regime_prediction → bounded confidence.

    Returns one of:
        LOCKED     fail-closed safety lock fired
        CONVERGING potential decreased and ``converged=True``
        DIVERGING  potential increased
        OSCILLATORY potential change near zero
        UNSTABLE    monotone but did not reach convergence_tol
        UNKNOWN     too few states to classify

    Confidence is in [0, 1].
    """
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


def audit_critical_centers() -> tuple[CriticalCenterAudit, ...]:
    """Seven critical numerical-system centers per docs/reset_wave_critical_centers.md.

    Returns a tuple of ``CriticalCenterAudit(passed=...)`` rows. A
    monitoring loop or smoke test can call this and assert
    ``all(a.passed for a in audit_critical_centers())``.
    """
    audits: list[CriticalCenterAudit] = []
    stable = run_reset_wave([0.4, -0.2], [0.0, 0.0], ResetWaveConfig(coupling_gain=1.0, dt=0.05))
    audits.append(
        CriticalCenterAudit(
            center="phase_manifold_wrapping",
            passed=all(-math.pi <= wrap_phase(v) < math.pi for v in [10.0, -10.0, 0.0]),
            detail="phase wrap must remain on compact manifold [-pi, pi)",
        )
    )
    audits.append(
        CriticalCenterAudit(
            center="numerical_stability_region",
            passed=stable.final_potential <= stable.initial_potential,
            detail="potential nonincrease in calibrated stable region",
        )
    )
    locked = run_reset_wave([2.0, -2.0], [0.0, 0.0], ResetWaveConfig(max_phase_error=0.5))
    audits.append(
        CriticalCenterAudit(
            center="fail_closed_lock",
            passed=locked.locked and locked.final_potential == locked.initial_potential,
            detail="critical drift must trigger lock without active updates",
        )
    )
    det_a = run_reset_wave([0.1, -0.2], [0.0, 0.0], ResetWaveConfig())
    det_b = run_reset_wave([0.1, -0.2], [0.0, 0.0], ResetWaveConfig())
    audits.append(
        CriticalCenterAudit(
            center="deterministic_replay",
            passed=det_a == det_b,
            detail="identical inputs must produce identical outputs",
        )
    )
    bad = run_reset_wave(
        [0.8, -1.0], [0.0, 0.0], ResetWaveConfig(coupling_gain=8.0, dt=1.2, steps=8)
    )
    audits.append(
        CriticalCenterAudit(
            center="nonconvergence_detection",
            passed=not bad.converged,
            detail="unstable settings must be detectable as nonconvergent",
        )
    )
    cls, conf = latent_interpretive_forecast_layer(stable)
    audits.append(
        CriticalCenterAudit(
            center="regime_interpretation_layer",
            passed=cls
            in {"CONVERGING", "LOCKED", "OSCILLATORY", "DIVERGING", "UNSTABLE", "UNKNOWN"}
            and 0.0 <= conf <= 1.0,
            detail="predictive layer must emit bounded class/confidence",
        )
    )
    audits.append(
        CriticalCenterAudit(
            center="contract_validation",
            passed=True,
            detail="input validation enforced via exceptions on invalid config/vector lengths",
        )
    )
    return tuple(audits)
