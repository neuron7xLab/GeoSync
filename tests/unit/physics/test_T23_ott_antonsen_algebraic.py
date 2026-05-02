# SPDX-License-Identifier: MIT
"""T23 (algebraic companion) — Ott-Antonsen fixed-point at machine epsilon.

The companion test file (`test_T23_ott_antonsen_chimera.py`) verifies
INV-OA2 by integrating the ODE for T=100 and comparing the long-time
limit with `√(1 − 2Δ/K)`. Because that test goes through RK4 at
``dt=0.01``, its tolerance is bounded by the integrator (≈ 1e-3) — a
loosening of the underlying invariant which is, per CLAUDE.md
("INV-OA2 ... Analytical — CAN test to float precision"), exact at
the algebraic level.

These tests close that gap. They do *not* integrate. They evaluate
the Ott-Antonsen ODE right-hand side directly at the analytical
fixed point ``z = R_∞ + 0j`` (with ω₀ = 0) and assert
``|dz/dt| < 1e-13`` — i.e., the formula and the implementation
agree to within IEEE-754 double-precision arithmetic noise on a
single ``_dz_dt`` evaluation.

Why this is the strongest possible test for INV-OA2
---------------------------------------------------

INV-OA2 is an *algebraic* invariant: at the supercritical fixed
point ``z = R_∞`` (real, ω₀ = 0), the right-hand side reduces to::

    dz/dt = -Δ·R_∞ + (K/2)·(R_∞ − R_∞³)
          = R_∞·[(K/2)·(1 − R_∞²) − Δ]
          = R_∞·[(K/2)·(2Δ/K) − Δ]    (substitute R_∞² = 1 − 2Δ/K)
          = R_∞·[Δ − Δ]
          = 0.

So at the analytical R_∞, ``_dz_dt`` should return exactly zero
modulo IEEE-754 round-off. Any deviation larger than ~10·eps
proves the implementation has drifted from the formula. The
existing 1e-3 RK4 test would not catch a sign error in one of the
RHS terms; this one would.
"""

from __future__ import annotations

import math

import pytest

from core.kuramoto.ott_antonsen import OttAntonsenEngine

# Machine epsilon ≈ 2.22e-16; allow ~50·eps for accumulated round-off
# across the multiplication / subtraction chain in _dz_dt. Empirically
# the residual stays under 1e-14 across the parameter sweep.
_ALGEBRAIC_TOL: float = 1e-13


@pytest.mark.parametrize(
    ("K", "delta"),
    [
        (1.5, 0.5),  # K/K_c = 1.5  → R_∞ ≈ 0.577
        (2.0, 0.5),  # K/K_c = 2.0  → R_∞ ≈ 0.707
        (3.0, 0.5),  # K/K_c = 3.0  → R_∞ ≈ 0.816
        (5.0, 0.5),  # K/K_c = 5.0  → R_∞ ≈ 0.894
        (10.0, 0.5),  # K/K_c = 10.0 → R_∞ ≈ 0.949
        (100.0, 0.5),  # extreme: R_∞ → 1
        (1.01, 0.5),  # near critical: R_∞ ≈ 0.099 (tight test of the formula)
        (4.0, 1.0),  # different Δ
        (0.4, 0.1),  # smaller scale
        (200.0, 0.05),  # extreme: R_∞ → 1, very small Δ
    ],
)
def test_inv_oa2_fixed_point_residual_at_machine_epsilon(K: float, delta: float) -> None:
    """INV-OA2 (algebraic): _dz_dt(R_∞ + 0j) == 0 to machine precision.

    Evaluates the right-hand side at the analytical fixed point and
    asserts the residual is below ``1e-13``. This is the strongest
    possible test for INV-OA2 because it does not depend on a
    numerical integrator — only on the formula's algebraic
    correctness.
    """
    engine = OttAntonsenEngine(K=K, delta=delta, omega0=0.0)
    R_inf = math.sqrt(1.0 - 2.0 * delta / K)
    z_fixed = complex(R_inf, 0.0)
    dz = engine._dz_dt(z_fixed)  # noqa: SLF001 — algebraic invariant probe
    residual = abs(dz)
    assert residual < _ALGEBRAIC_TOL, (
        f"INV-OA2 ALGEBRAIC VIOLATED: |dz/dt| at the analytical "
        f"fixed point R_∞ = √(1 − 2Δ/K) is {residual:.3e}, "
        f"expected < {_ALGEBRAIC_TOL:.0e}. "
        f"Observed at K={K}, Δ={delta}, K_c={2 * delta}, R_∞={R_inf:.16f}, "
        f"dz/dt = {dz!r}. "
        "Physical reasoning: at the supercritical fixed point the "
        "drift terms exactly cancel — any residual above the "
        "round-off floor implies a sign or factor drift in _dz_dt "
        "that the integration-tolerance test (1e-3) would miss."
    )


def test_inv_oa2_subcritical_zero_is_also_a_fixed_point() -> None:
    """INV-OA2 / INV-OA3: z = 0 is *always* a fixed point (sub- and supercritical).

    For both K < K_c and K > K_c the incoherent state z = 0 is a
    fixed point of the ODE. Subcritically it is the *only* stable
    one; supercritically it is unstable but still a fixed point.
    Either way, ``_dz_dt(0)`` must be exactly ``0+0j``.
    """
    for K, delta in [(0.5, 0.5), (1.0, 0.5), (2.0, 0.5), (10.0, 0.5)]:
        engine = OttAntonsenEngine(K=K, delta=delta)
        dz = engine._dz_dt(complex(0.0, 0.0))  # noqa: SLF001
        assert dz == 0j, (
            f"INV-OA2/3 VIOLATED: z=0 must yield dz/dt == 0+0j exactly; "
            f"got {dz!r} at K={K}, delta={delta}. "
            "Any non-zero result would imply spurious drift terms in _dz_dt."
        )


def test_inv_oa2_critical_point_zero_residual() -> None:
    """INV-OA2 (boundary) — at K = K_c the only fixed point is z = 0; R_∞ formula returns 0.

    Verifies that :attr:`OttAntonsenEngine.R_steady` returns exactly
    ``0.0`` at the bifurcation point (no pseudo-supercritical drift),
    and that ``_dz_dt(0)`` agrees.
    """
    delta = 0.7
    K_c = 2.0 * delta
    engine = OttAntonsenEngine(K=K_c, delta=delta)
    assert engine.R_steady == 0.0, (
        f"INV-OA2 boundary: at K = K_c the steady-state magnitude must "
        f"be exactly 0.0; got R_steady={engine.R_steady!r} at K={K_c}, "
        f"delta={delta}. The formula √(1 − K_c/K) evaluates to 0 here, "
        "but a sloppy implementation could return √eps."
    )
    dz = engine._dz_dt(complex(0.0, 0.0))  # noqa: SLF001
    assert dz == 0j, (
        f"INV-OA2 boundary VIOLATED: _dz_dt(0) at K = K_c "
        f"observed = {dz!r}, expected = 0+0j. "
        f"Parameters: K={K_c}, delta={delta}, K_c=2·delta={2 * delta}. "
        "Physical reasoning: at the bifurcation point K = K_c the "
        "incoherent fixed point z = 0 is marginally stable; the ODE "
        "right-hand side must still vanish there exactly by symmetry."
    )


def test_inv_oa2_residual_grows_with_perturbation_size() -> None:
    """INV-OA2 (discriminative-power probe) — verify the algebraic test is non-vacuous.

    Demonstrates that ``_dz_dt`` is sensitive to perturbations off
    the fixed point, so the at-fixed-point residual being below
    ``1e-13`` is a real falsification of any drift in INV-OA2 — not
    an artifact of a degenerate (e.g., constantly-zero) function.
    The residual at ``z = R_∞ + ε`` must grow at least linearly
    with ε for small ε.
    """
    K, delta = 3.0, 0.5
    engine = OttAntonsenEngine(K=K, delta=delta)
    R_inf = math.sqrt(1.0 - 2.0 * delta / K)
    base = abs(engine._dz_dt(complex(R_inf, 0.0)))  # noqa: SLF001
    perturbed = abs(engine._dz_dt(complex(R_inf + 1e-6, 0.0)))  # noqa: SLF001
    assert perturbed > base * 1e6, (
        f"INV-OA2 discriminative-power probe failed: "
        f"observed perturbed-residual = {perturbed:.3e}, "
        f"expected ≥ base × 1e6 = {base * 1e6:.3e} (base = {base:.3e}). "
        f"Parameters: K={K}, delta={delta}, R_inf={R_inf:.6f}, perturbation=1e-6. "
        "Physical reasoning: if _dz_dt is suspiciously flat near the "
        "fixed point, the algebraic test cannot catch implementation "
        "drift and INV-OA2 would be vacuously satisfied."
    )
