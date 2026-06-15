# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""T18 — P1 Kuramoto invariants: monotone R(K), incoherent scaling, Lyapunov.

Three physically distinct P1 invariants tested against the production
``core.kuramoto.engine.KuramotoEngine``:

* **INV-K4** (conditional / sweep): steady-state R is monotone
  non-decreasing in coupling strength K for the standard all-to-all
  model without frequency-degree correlation.

* **INV-K5** (statistical / ensemble): in the fully incoherent regime
  (K=0), the mean order parameter scales as O(1/sqrt(N)) — the
  finite-size noise floor predicted by the central limit theorem on the
  unit circle.

* **INV-K7** (monotonic / trajectory): the Lyapunov functional
  V = -(K*N/2)*R^2 is non-increasing along trajectories with identical
  natural frequencies (all omega_i = 0), confirming gradient descent
  on the Kuramoto potential landscape.

* **INV-K8** (conservation / trajectory): the *second-order* swing-equation
  total energy E = (1/2)*sum(m*theta_dot^2) + V is conserved under the
  Stoermer-Verlet integrator when undamped (omega=0, d=0) and is monotone
  non-increasing when damped (omega=0, d>0). For the all-to-all model
  V = -(K*N/2)*R^2 + const (the INV-K7 potential), so E is observable
  black-box from velocities and R. Defends the "symplectic, energy-conserving"
  docstring claim in ``core.kuramoto.second_order`` against the two regression
  classes that survive every prior assertion: non-symplectic (Euler) drift
  and damping-sign inversion (anti-damping = energy injection).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.kuramoto.config import KuramotoConfig
from core.kuramoto.engine import KuramotoEngine
from core.kuramoto.second_order import SecondOrderKuramotoEngine

# ---------------------------------------------------------------------------
# INV-K4: R_inf(K1) <= R_inf(K2) when K1 < K2  (sweep / conditional)
# ---------------------------------------------------------------------------


@pytest.mark.heavy_math
def test_steady_state_R_monotone_in_coupling_strength() -> None:
    """INV-K4: R_inf monotonically non-decreasing in K for standard model.

    Sweep K from 0.5 to 5.0 in 10 steps with N=200 oscillators.
    Compute steady-state R as the mean over the last 500 steps of
    each trajectory and assert the resulting sequence is monotone
    non-decreasing up to at most 2 finite-size noise violations.
    """
    # INV-K4: sweep parameters
    n_oscillators = 200  # N=200
    steps = 2000
    seed = 42
    dt = 0.01  # tolerance: standard integration step
    k_values = np.linspace(0.5, 5.0, 10)

    rng = np.random.default_rng(seed)
    omega = rng.standard_normal(n_oscillators)
    theta0 = rng.uniform(0.0, 2.0 * math.pi, n_oscillators)

    r_steady: list[float] = []
    for k_val in k_values:
        cfg = KuramotoConfig(
            N=n_oscillators,
            K=float(k_val),
            omega=omega,
            theta0=theta0,
            dt=dt,
            steps=steps,
            seed=seed,
        )
        result = KuramotoEngine(cfg).run()
        trajectory = result.order_parameter
        # INV-K4: late-time window for steady-state estimate
        r_tail = float(np.mean(trajectory[-500:]))
        r_steady.append(r_tail)

    diffs = np.diff(r_steady)
    # INV-K4: count violations (where R decreased)
    violations = int(np.sum(diffs < 0))
    # tolerance: allow up to 2 violations from finite-size noise
    max_violations = 2

    assert violations <= max_violations, (
        f"INV-K4 VIOLATED: R_inf not monotone in K. "
        f"Expected at most {max_violations} violations, observed {violations}. "
        f"R_steady={[f'{r:.4f}' for r in r_steady]}. "
        f"At N={n_oscillators}, steps={steps}, seed={seed}, "
        f"K=[{k_values[0]:.1f}..{k_values[-1]:.1f}]. "
        f"Physical reasoning: standard Kuramoto R_inf is monotone in K "
        f"absent frequency-degree correlation."
    )


# ---------------------------------------------------------------------------
# INV-K5: <R> ~ O(1/sqrt(N)) in incoherent regime  (ensemble / statistical)
# ---------------------------------------------------------------------------


def test_incoherent_order_parameter_scales_as_inverse_sqrt_N() -> None:
    """INV-K5: mean R at K=0 is O(1/sqrt(N)) over >= 50 realizations.

    With zero coupling (K=0), oscillators are independent and R is the
    magnitude of the mean of N unit-circle random variables. By CLT,
    E[R] ~ 1/sqrt(N). We run 50 seeds, compute mean R for each, then
    check |<R> - 1/sqrt(N)| < 3*sigma/sqrt(n_trials).
    """
    # INV-K5: ensemble parameters
    n_oscillators = 100  # N=100
    steps = 500
    n_trials = 50
    dt = 0.01  # tolerance: standard integration step

    # INV-K5: theoretical prediction
    # epsilon: E[R] = sqrt(pi/(4*N)) for Rayleigh distribution of |mean(exp(i*theta))|
    # The exact finite-N result is sqrt(pi/(4*N)), which is O(1/sqrt(N)).
    expected_r = math.sqrt(math.pi / (4.0 * n_oscillators))

    r_means: list[float] = []
    for trial in range(n_trials):
        seed = trial + 1000
        cfg = KuramotoConfig(
            N=n_oscillators,
            K=0.0,
            dt=dt,
            steps=steps,
            seed=seed,
        )
        result = KuramotoEngine(cfg).run()
        trajectory = result.order_parameter
        # INV-K5: use late-time mean to avoid transient
        r_means.append(float(np.mean(trajectory[-200:])))

    ensemble_mean = float(np.mean(r_means))
    ensemble_std = float(np.std(r_means, ddof=1))
    # INV-K5: tolerance from standard error of the mean
    # tolerance: 3 * sigma / sqrt(n_trials) — 3-sigma confidence bound
    tolerance = 3.0 * ensemble_std / math.sqrt(n_trials)

    deviation = abs(ensemble_mean - expected_r)

    assert deviation < tolerance, (
        f"INV-K5 VIOLATED: incoherent <R> deviates from sqrt(pi/(4*N)). "
        f"Expected <R> ~ {expected_r:.4f}, observed <R>={ensemble_mean:.4f}, "
        f"|deviation|={deviation:.4f} >= tolerance={tolerance:.4f}. "
        f"At N={n_oscillators}, K=0.0, steps={steps}, n_trials={n_trials}. "
        f"Physical reasoning: at K=0 oscillators are independent; "
        f"R = |mean(exp(i*theta))| ~ sqrt(pi/(4*N)) by Rayleigh statistics."
    )


# ---------------------------------------------------------------------------
# INV-K7: Lyapunov V = -(K*N/2)*R^2 non-increasing for omega=0
# ---------------------------------------------------------------------------


def test_lyapunov_functional_non_increasing_identical_frequencies() -> None:
    """INV-K7: V = -(K*N/2)*R^2 is non-increasing when all omega_i = 0.

    With identical frequencies the Kuramoto system is a gradient flow
    on the potential V. We compute V at each integration step and assert
    it never increases by more than a floating-point tolerance of 1e-10.
    """
    # INV-K7: trajectory parameters
    n_oscillators = 50  # N=50
    k_coupling = 2.0  # K=2.0
    steps = 1000
    seed = 42
    dt = 0.01  # tolerance: standard integration step

    omega = np.zeros(n_oscillators)
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(0.0, 2.0 * math.pi, n_oscillators)

    cfg = KuramotoConfig(
        N=n_oscillators,
        K=k_coupling,
        omega=omega,
        theta0=theta0,
        dt=dt,
        steps=steps,
        seed=seed,
    )
    result = KuramotoEngine(cfg).run()
    r_trajectory = result.order_parameter

    # INV-K7: Lyapunov functional V = -(K*N/2) * R^2
    coeff = -(k_coupling * n_oscillators) / 2.0
    v_trajectory = coeff * r_trajectory**2

    dv = np.diff(v_trajectory)
    # tolerance: 1e-10 for floating-point accumulation in RK4
    fp_tolerance = 1e-10  # epsilon: floating-point tolerance for Lyapunov check
    violations = dv[dv > fp_tolerance]
    n_violations = len(violations)

    assert n_violations == 0, (
        f"INV-K7 VIOLATED: Lyapunov V = -(K*N/2)*R^2 increased {n_violations} times. "
        f"Expected V non-increasing (dV <= {fp_tolerance}), "
        f"observed max dV={float(np.max(dv)):.2e}. "
        f"At N={n_oscillators}, K={k_coupling}, steps={steps}, seed={seed}. "
        f"Physical reasoning: with omega_i=0 for all i, "
        f"Kuramoto is a gradient system and V is a strict Lyapunov function."
    )


# ---------------------------------------------------------------------------
# INV-K8: second-order swing-equation energy balance
#   E = (1/2) sum(m*theta_dot^2) + V,   V_all-to-all = -(K*N/2)*R^2 + const
#   omega=0, d=0  ->  dE/dt = 0          (conservation, Stoermer-Verlet)
#   omega=0, d>0  ->  dE/dt = -sum(d*theta_dot^2) <= 0   (dissipation)
# ---------------------------------------------------------------------------


def _swing_energy(
    velocities: np.ndarray, r_trajectory: np.ndarray, mass: float, k_coupling: float, n: int
) -> np.ndarray:
    """Total swing energy per timestep for the all-to-all second-order model.

    E(t) = (1/2) * mass * sum_i v_i(t)^2  +  V(t),  with the all-to-all
    potential V(t) = -(K*N/2) * R(t)^2 (the INV-K7 potential, up to an
    additive constant K/2 that cancels in every drift / monotonicity check).
    """
    kinetic = 0.5 * mass * np.sum(velocities**2, axis=1)
    potential = -0.5 * k_coupling * n * r_trajectory**2
    return kinetic + potential


def test_swing_energy_conserved_undamped() -> None:
    """INV-K8 (conservation): undamped swing energy is constant under Verlet.

    With omega_i = 0 and d = 0 the autonomous swing equation is conservative,
    so a symplectic (Stoermer-Verlet) integrator keeps E within a bounded
    O(dt^2) oscillation with no secular trend. We force a genuine transient
    (unlocked theta0 + non-zero velocity0) because the conservation law is
    sharpest exactly at this degenerate (omega=0, d=0) slice. The bound
    range/E_scale < 1e-2 sits ~300x above the observed Verlet drift (~3e-5)
    yet ~40x below the non-symplectic Euler drift (~0.39): it is a true
    falsifier for loss of symplecticity, not a magic number.
    """
    # INV-K8: undamped trajectory parameters
    n_oscillators = 64  # N=64
    k_coupling = 4.0  # K=4.0
    mass = 1.0  # uniform inertia
    steps = 6000
    seed = 0
    dt = 0.01  # tolerance: standard integration step

    rng = np.random.default_rng(seed)
    omega = np.zeros(n_oscillators)  # INV-K8 condition: zero natural frequency
    theta0 = rng.uniform(0.0, 2.0 * math.pi, n_oscillators)  # unlocked -> transient
    velocity0 = 0.3 * rng.standard_normal(n_oscillators)  # kinetic seed

    cfg = KuramotoConfig(
        N=n_oscillators,
        K=k_coupling,
        omega=omega,
        theta0=theta0,
        dt=dt,
        steps=steps,
        seed=seed,
    )
    result = SecondOrderKuramotoEngine(
        cfg, mass=mass, damping=0.0, velocity0=velocity0
    ).run()

    energy = _swing_energy(
        result.velocities, result.order_parameter, mass, k_coupling, n_oscillators
    )
    e_scale = 0.5 * k_coupling * n_oscillators  # potential well depth scale
    bounded_drift = float(energy.max() - energy.min())
    rel_drift = bounded_drift / e_scale
    # tolerance: empirically Verlet gives ~3e-5, Euler ~0.39 over this horizon
    drift_bound = 1e-2  # epsilon: symplectic bounded-drift tolerance

    assert rel_drift < drift_bound, (
        f"INV-K8 VIOLATED: undamped swing energy drifted range={bounded_drift:.3e} "
        f"(range/E_scale={rel_drift:.3e}) > tolerance={drift_bound:.1e}. "
        f"Expected bounded O(dt^2) oscillation with no secular trend; this "
        f"magnitude indicates loss of symplecticity (e.g. Euler integration). "
        f"At N={n_oscillators}, K={k_coupling}, d=0, steps={steps}, seed={seed}. "
        f"Physical reasoning: omega=0, d=0 -> dE/dt=0, conservative dynamics."
    )

    # Secular-trend guard: a symplectic scheme has zero net energy drift.
    initial = float(energy[0])
    final = float(energy[-1])
    step_index = np.arange(energy.size, dtype=np.float64)
    secular_slope = float(np.polyfit(step_index, energy, 1)[0])
    secular_total = abs(secular_slope) * energy.size
    rel_secular = secular_total / e_scale
    # tolerance: Verlet secular drift ~2e-4/E_scale; Euler is orders larger
    secular_bound = 1e-2  # epsilon: net (secular) energy drift tolerance

    assert rel_secular < secular_bound, (
        f"INV-K8 VIOLATED: undamped swing energy shows a secular trend "
        f"total={secular_total:.3e} (total/E_scale={rel_secular:.3e}) > "
        f"tolerance={secular_bound:.1e}. Expected a stationary E(t) with zero "
        f"net drift (initial={initial:.3e}, final={final:.3e}); a monotone trend "
        f"indicates non-symplectic integration. "
        f"At N={n_oscillators}, K={k_coupling}, d=0, steps={steps}, seed={seed}. "
        f"Physical reasoning: conservative swing flow -> E(t) is stationary."
    )


def test_swing_energy_dissipated_damped() -> None:
    """INV-K8 (dissipation): damped swing energy is monotone non-increasing.

    With omega_i = 0 and d > 0, dE/dt = -sum(d*theta_dot^2) <= 0, so E must
    not increase along the trajectory and must net-decrease while there is
    motion to dissipate. This clause is the falsifier for a damping-sign
    inversion (-d*theta_dot -> +d*theta_dot), which converts dissipation
    into energy injection and makes E diverge (~10^6 over this horizon).
    """
    # INV-K8: damped trajectory parameters
    n_oscillators = 64  # N=64
    k_coupling = 4.0  # K=4.0
    mass = 1.0  # uniform inertia
    damping = 0.2  # d>0: strictly dissipative
    steps = 6000
    seed = 0
    dt = 0.01  # tolerance: standard integration step

    rng = np.random.default_rng(seed)
    omega = np.zeros(n_oscillators)  # INV-K8 condition: zero natural frequency
    theta0 = rng.uniform(0.0, 2.0 * math.pi, n_oscillators)  # unlocked -> transient
    velocity0 = 0.3 * rng.standard_normal(n_oscillators)  # kinetic seed to dissipate

    cfg = KuramotoConfig(
        N=n_oscillators,
        K=k_coupling,
        omega=omega,
        theta0=theta0,
        dt=dt,
        steps=steps,
        seed=seed,
    )
    result = SecondOrderKuramotoEngine(
        cfg, mass=mass, damping=damping, velocity0=velocity0
    ).run()

    energy = _swing_energy(
        result.velocities, result.order_parameter, mass, k_coupling, n_oscillators
    )
    e_scale = 0.5 * k_coupling * n_oscillators  # potential well depth scale
    de = np.diff(energy)
    # tolerance: observed max per-step change is negative; allow tiny float slack
    fp_tolerance = 1e-6 * e_scale  # epsilon: floating-point monotonicity slack
    increases = de[de > fp_tolerance]
    n_increases = len(increases)

    assert n_increases == 0, (
        f"INV-K8 VIOLATED: damped swing energy increased {n_increases} times "
        f"(max step dE={float(np.max(de)):.3e} > tol={fp_tolerance:.3e}). "
        f"Expected dE/dt = -sum(d*theta_dot^2) <= 0; a positive step indicates "
        f"a damping-sign inversion (anti-damping = energy injection). "
        f"At N={n_oscillators}, K={k_coupling}, d={damping}, steps={steps}, seed={seed}. "
        f"Physical reasoning: positive damping can only remove kinetic energy."
    )

    # Net dissipation must actually occur (guards a no-op / near-zero damping path).
    initial = float(energy[0])
    final = float(energy[-1])
    net_change = final - initial
    assert net_change < -fp_tolerance, (
        f"INV-K8 VIOLATED: damped swing energy did not net-decrease "
        f"(observed final-initial={net_change:.3e} >= -tol={-fp_tolerance:.3e}). "
        f"With d={damping}>0 and a seeded transient, dissipation must reduce E. "
        f"At N={n_oscillators}, K={k_coupling}, steps={steps}, seed={seed}. "
        f"Physical reasoning: sustained motion under positive damping bleeds energy."
    )
