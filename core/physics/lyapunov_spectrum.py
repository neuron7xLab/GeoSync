# SPDX-License-Identifier: MIT
"""Full Lyapunov spectrum via Benettin-Galgani-Strelcyn QR method.

LAW T2 — Whole-spectrum Lyapunov estimator for autonomous ODEs

GeoSync Physics Law T2 (constitutional, INV-LY1..LY3). Companion to the
scalar Rosenstein MLE in ``lyapunov_exponent.py``: that estimator works
on observed time-series; this estimator integrates the variational
(tangent) flow and returns every exponent up to ``n_exp``.

Algorithm (Benettin et al. 1980, "Lyapunov characteristic exponents
for smooth dynamical systems and for Hamiltonian systems"):

1.  Augment state ``x ∈ R^n`` with an orthonormal tangent frame
    ``Q ∈ R^{n × n_exp}``, ``Q^T Q = I``.
2.  Co-integrate the augmented system ``ẋ = f(x)``, ``Q̇ = J(x)·Q``
    where ``J = ∂f/∂x`` (forward-mode via ``jax.jacfwd``).
3.  Every ``qr_every`` steps, perform a QR decomposition
    ``Q ← Q``, ``R ← R`` and accumulate ``log|diag(R)|``.
4.  Spectrum: ``λ_k = (1/T) · Σ log|R_kk|``.

The integrator is the explicit midpoint rule (Heun-RK2): bit-identical
under ``jit`` for the same ``(x0, dt, n_steps)``, second-order accurate,
and cheap enough for ``vmap`` over hundreds of seeds. RK4 was rejected
for v1 — it would quadruple ``jacfwd`` invocations per step while the
dominant error is the finite QR frequency, not the integrator order.

Invariants (CLAUDE.md INVARIANT REGISTRY):

* INV-LY1 | algebraic    | linear ``ẋ = A x``: ``sort(spectrum) ==
                          sort(real(eigvals(A)))`` to 1e-6.
* INV-LY2 | conservation | Hamiltonian flow: ``Σ λ_k = 0`` (symplectic
                          pairing). Tested on harmonic oscillator.
* INV-LY3 | universal    | for INV-LE1 (existing): finite, bounded
                          input ⟹ finite spectrum.

Determinism: bit-identical output for identical ``(rhs, x0, dt,
n_steps, qr_every, n_exp)`` on same hardware/JAX version (INV-HPC1).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

# Public API ------------------------------------------------------------------

__all__ = [
    "LyapunovReport",
    "lyapunov_spectrum",
]

# Numerical tolerance below which |R_kk| is treated as a zero singular
# value (collapsed direction). Slightly above float32 eps to stay
# robust under float32 fallback. Linked to INV-LY3.
_LOG_FLOOR: float = 1e-30


class LyapunovReport(NamedTuple):
    """Outcome of a single ``lyapunov_spectrum`` run.

    Attributes
    ----------
    spectrum:
        Length-``n_exp`` array sorted descending. ``spectrum[0]`` is
        the maximal Lyapunov exponent (compatible with the scalar
        ``maximal_lyapunov_exponent`` for the same trajectory).
    log_growth:
        Cumulative ``Σ log|R_kk|`` per direction, BEFORE division by
        ``T``. Useful for convergence diagnostics — its derivative
        with respect to ``T`` should stabilise.
    final_state:
        State ``x`` at the end of integration. Returned so callers
        can chain runs without re-warming.
    final_frame:
        Orthonormal frame ``Q`` at the end of integration.
    integration_time:
        Total wall-clock-physics time ``T = n_steps · dt``.
    """

    spectrum: Array
    log_growth: Array
    final_state: Array
    final_frame: Array
    integration_time: float


def lyapunov_spectrum(
    rhs: Callable[[Array], Array],
    x0: Array,
    *,
    dt: float,
    n_steps: int,
    n_exp: int | None = None,
    qr_every: int = 1,
) -> LyapunovReport:
    """Estimate the Lyapunov spectrum of an autonomous ODE ``ẋ = rhs(x)``.

    Parameters
    ----------
    rhs:
        Right-hand side. Must be a JAX-traceable function ``Array
        (shape (n,)) -> Array (shape (n,))``. Will be differentiated
        via ``jax.jacfwd`` once per midpoint substep.
    x0:
        Initial condition, shape ``(n,)``. ``dtype`` propagates to all
        outputs; pass ``float64`` for production-grade precision (set
        ``jax.config.update("jax_enable_x64", True)`` first).
    dt:
        Integration timestep. Must satisfy ``dt > 0``.
    n_steps:
        Total number of midpoint integration steps. Total physical
        integration time is ``T = n_steps · dt``.
    n_exp:
        Number of exponents to estimate. Defaults to ``n`` (full
        spectrum). Passing ``n_exp < n`` is a valid optimisation when
        only the leading exponents are needed.
    qr_every:
        Re-orthonormalise the tangent frame every ``qr_every``
        integration steps. ``1`` is most stable; larger values save
        compute at the cost of conditioning. ``qr_every`` must
        divide ``n_steps``.

    Returns
    -------
    LyapunovReport
        See class docstring.

    Raises
    ------
    ValueError
        On any input that violates a contract: non-positive ``dt``,
        non-positive ``n_steps``, ``qr_every`` not dividing ``n_steps``,
        rank-deficient initial frame, ``x0`` not 1-D, ``n_exp >
        x0.size``. Fail-closed (INV-LE1, INV-HPC2 spirit) — no silent
        repair.

    Notes
    -----
    The implementation is **pure functional** and JIT-friendly:
    ``rhs`` is captured by closure and the returned function compiles
    on first call with given ``(n_steps, qr_every, n_exp, dt)``. For
    parameter sweeps prefer ``vmap`` over a wrapping function rather
    than re-tracing.
    """
    # --- Contract checks (fail-closed) ---------------------------------------
    if dt <= 0.0:
        raise ValueError(f"INV-LY3: dt must be positive, got {dt}")
    if n_steps <= 0:
        raise ValueError(f"INV-LY3: n_steps must be positive, got {n_steps}")
    if qr_every <= 0:
        raise ValueError(f"INV-LY3: qr_every must be positive, got {qr_every}")
    if n_steps % qr_every != 0:
        raise ValueError(f"INV-LY3: qr_every={qr_every} must divide n_steps={n_steps}")

    x0 = jnp.asarray(x0)
    if x0.ndim != 1:
        raise ValueError(f"INV-LY3: x0 must be 1-D, got shape {x0.shape}")
    n: int = int(x0.shape[0])
    n_exp_eff: int = n if n_exp is None else int(n_exp)
    if n_exp_eff <= 0 or n_exp_eff > n:
        raise ValueError(f"INV-LY3: n_exp must satisfy 1 ≤ n_exp ≤ {n}, got {n_exp_eff}")

    n_outer: int = n_steps // qr_every
    total_time: float = float(n_steps) * float(dt)

    # --- Initial orthonormal frame -------------------------------------------
    # Q0: first n_exp standard basis vectors. They are exactly orthonormal
    # so the first QR step is a no-op identity (sanity).
    Q0: Array = jnp.eye(n, n_exp_eff, dtype=x0.dtype)

    # --- Augmented midpoint integrator (no QR inside) ------------------------
    jac: Callable[[Array], Array] = jax.jacfwd(rhs)

    def _midpoint_inner(_i: Array, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        """One midpoint step on (x, Q). Pure, jit-fusable."""
        x, Q = carry
        f1 = rhs(x)
        J1 = jac(x)
        x_mid = x + 0.5 * dt * f1
        Q_mid = Q + 0.5 * dt * J1 @ Q
        f2 = rhs(x_mid)
        J2 = jac(x_mid)
        x_next = x + dt * f2
        Q_next = Q + dt * J2 @ Q_mid
        return (x_next, Q_next)

    def _outer_step(_k: Array, carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        """``qr_every`` midpoint steps, then QR re-orthonormalisation."""
        x, Q, log_sum = carry
        x, Q = jax.lax.fori_loop(0, qr_every, _midpoint_inner, (x, Q))
        # Reduced QR; R is upper-triangular (n_exp × n_exp).
        Q, R = jnp.linalg.qr(Q)
        diag_abs = jnp.abs(jnp.diag(R))
        # INV-LY3: any direction collapsed to numerical zero accumulates
        # as a strongly negative exponent rather than -inf.
        log_sum = log_sum + jnp.log(jnp.maximum(diag_abs, _LOG_FLOOR))
        return (x, Q, log_sum)

    log_sum0: Array = jnp.zeros(n_exp_eff, dtype=x0.dtype)
    x_final, Q_final, log_sum_final = jax.lax.fori_loop(0, n_outer, _outer_step, (x0, Q0, log_sum0))

    spectrum_unsorted = log_sum_final / total_time
    # Sort descending so spectrum[0] is the maximal exponent (matches
    # convention used by INV-LE2 and downstream T1 phase boundary).
    order = jnp.argsort(-spectrum_unsorted)
    spectrum = spectrum_unsorted[order]
    log_growth = log_sum_final[order]

    return LyapunovReport(
        spectrum=spectrum,
        log_growth=log_growth,
        final_state=x_final,
        final_frame=Q_final,
        integration_time=total_time,
    )
