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
"""

from __future__ import annotations

import math

import numpy as np

from core.kuramoto.config import KuramotoConfig
from core.kuramoto.engine import KuramotoEngine

# ---------------------------------------------------------------------------
# INV-K4: R_inf(K1) <= R_inf(K2) when K1 < K2  (sweep / conditional)
# ---------------------------------------------------------------------------


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
