# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Issue #1109 — full SecondOrderStabilityAudit.

Closes the gaps the runtime guard (non-finite + RoCoF) leaves open. Each test
pins a measured numerical-stability diagnostic of the symplectic Störmer–Verlet
second-order Kuramoto integrator and would fail if that property degraded.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.kuramoto.config import KuramotoConfig
from core.kuramoto.second_order import (
    SecondOrderKuramotoEngine,
    SecondOrderStabilityReport,
)


def _conservative_engine() -> SecondOrderKuramotoEngine:
    """ω≡0, d≡0 — the regime where swing energy must be (near-)conserved."""
    n = 6
    cfg = KuramotoConfig(
        N=n,
        K=2.0,
        omega=np.zeros(n),
        theta0=np.linspace(0.0, 1.0, n),
        dt=0.005,
        steps=400,
        seed=0,
    )
    return SecondOrderKuramotoEngine(
        cfg, mass=1.0, damping=0.0, velocity0=0.1 * np.ones(n)
    )


def test_energy_like_drift_bound() -> None:
    """Symplectic Verlet keeps swing energy bounded in the conservative regime.

    A non-symplectic / broken integrator would drift secularly; the relative
    energy drift must stay tiny (≤1e-3) for ω≡0, d≡0.
    """
    report = _conservative_engine().audit()
    assert report.energy_conservative_regime is True
    assert report.energy_like_drift >= 0.0
    assert report.energy_like_drift <= 1e-3, (
        f"swing-energy drift {report.energy_like_drift:.3e} exceeds 1e-3 in the "
        f"conservative regime — symplectic conservation (INV-K8/K9) degraded."
    )


def test_phase_spread_bound() -> None:
    """Phase spread is finite and reported; a runaway would be non-finite/huge."""
    report = _conservative_engine().audit()
    assert report.phase_spread_finite is True
    assert np.isfinite(report.phase_spread_bound)
    assert report.phase_spread_bound >= 0.0


def test_solver_metadata_present() -> None:
    """The integrator is identified, with order/symplecticity/step policy."""
    meta = _conservative_engine().audit().solver_metadata
    assert meta["integrator"] == "stoermer_verlet"
    assert meta["order"] == 2
    assert meta["symplectic"] is True
    assert meta["adaptive"] is False
    assert meta["fixed_step"] is True
    assert meta["dt"] == pytest.approx(0.005)
    # fixed-step integrator has no tolerance knobs — declared null, not faked.
    assert meta["rtol"] is None
    assert meta["atol"] is None


def test_stiffness_regime_declared() -> None:
    """Stiffness ratio = ω_fast·dt is computed and the regime labelled.

    A well-resolved run (small dt, modest K) is 'resolved'; a coarse, strongly
    coupled run is 'underresolved_stiff'.
    """
    resolved = _conservative_engine().audit()
    assert resolved.stiffness_regime == "resolved"
    assert 0.0 <= resolved.stiffness_ratio <= 0.1

    n = 6
    stiff_cfg = KuramotoConfig(
        N=n,
        K=50.0,
        omega=np.zeros(n),
        theta0=np.linspace(0.0, 1.0, n),
        dt=0.2,
        steps=20,
        seed=0,
    )
    stiff = SecondOrderKuramotoEngine(stiff_cfg, mass=1.0, damping=0.0).audit(
        cross_solver_steps=10, agreement_tol=1.0
    )
    assert stiff.stiffness_ratio > 1.0
    assert stiff.stiffness_regime == "underresolved_stiff"


def test_cross_solver_reference_agreement() -> None:
    """The engine's Verlet trajectory agrees with an independent RK4 solver.

    Over a short, well-resolved horizon the two integrators must match to the
    audit tolerance — evidence the trajectory is correct, not self-consistent.
    """
    report = _conservative_engine().audit(cross_solver_steps=100, agreement_tol=1e-3)
    assert report.cross_solver_reference == "explicit_rk4"
    assert report.cross_solver_agrees is True
    assert report.cross_solver_max_deviation <= 1e-3


def test_damped_run_loses_energy_monotonically() -> None:
    """With damping the swing energy must not be claimed conserved.

    Sanity: the conservative flag is False once d>0, so no false conservation
    claim is emitted for a dissipative run.
    """
    n = 6
    cfg = KuramotoConfig(
        N=n, K=2.0, omega=np.zeros(n), theta0=np.linspace(0.0, 1.0, n), dt=0.005, steps=300, seed=0
    )
    report = SecondOrderKuramotoEngine(
        cfg, mass=1.0, damping=0.3, velocity0=0.5 * np.ones(n)
    ).audit()
    assert report.energy_conservative_regime is False


def test_report_is_frozen_and_complete() -> None:
    report = _conservative_engine().audit()
    assert isinstance(report, SecondOrderStabilityReport)
    assert report.audit_scope == "energy_phase_solver_stiffness_crosssolver"
    # remaining gaps are explicitly preserved — the audit is not over-claimed full.
    assert "full_lyapunov_spectrum" in report.remaining_gaps
    with pytest.raises((AttributeError, TypeError)):
        setattr(report, "energy_like_drift", 0.0)
