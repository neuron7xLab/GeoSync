# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Sparse signed coupling inference ``K_ij`` (protocol M1.3).

Given a :class:`~core.kuramoto.contracts.PhaseMatrix` we solve, for each
oscillator ``i`` independently, the penalised regression

.. math::

    \\dot\\theta_i(t) - \\bar\\omega_i \\;=\\; \\sum_{j \\neq i}
        \\beta_{ij}\\, \\sin\\bigl(\\theta_j(t) - \\theta_i(t)\\bigr) + \\varepsilon_i

where ``β_{ij} ≡ K_{ij}`` is the signed coupling we are after. The
design matrix is composed of phase-difference sines and the target
is the instantaneous frequency deviation from its temporal median.

This module ships a **pure-numpy** Minimax Concave Penalty (MCP),
SCAD, and L1 (Lasso) proximal-gradient solver — the methodology
explicitly warns against the deprecated ``stability-selection`` PyPI
package (broken on sklearn ≥ 1.3) and notes that ``skglm`` is optional.
Keeping the core path dependency-free guarantees the identification
stack runs on a bare scipy install.

On top of the row solver we ship a **complementary-pairs stability
selection** wrapper (Shah & Samworth, 2013) that returns per-edge
selection probabilities. Edges whose maximum selection probability
across the regularisation grid exceeds the user-chosen threshold are
retained; all others are zeroed out.

The output is a contract-compliant :class:`CouplingMatrix` with
optional per-edge ``stability_scores``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .contracts import CouplingMatrix, PhaseMatrix

__all__ = [
    "CouplingEstimationConfig",
    "CouplingEstimator",
    "PersistentExcitationError",
    "SwingCouplingEstimate",
    "estimate_coupling",
    "estimate_swing_coupling",
    "mcp_prox",
    "scad_prox",
    "soft_threshold",
    "complementary_pairs_stability",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CouplingEstimationConfig:
    """Hyperparameters for row-wise sparse coupling regression.

    Attributes
    ----------
    penalty : str
        ``"mcp"`` (default), ``"scad"``, or ``"lasso"``.
    lambda_reg : float
        Regularisation strength used by :func:`CouplingEstimator.estimate`
        when stability selection is disabled.
    gamma : float
        Concavity parameter for MCP/SCAD. ``γ = 3.0`` is the standard
        choice. For MCP the prox is well-defined as long as
        ``γ * L > 1``, where ``L`` is the Lipschitz constant of the
        smooth part; the row solver enforces this automatically.
    dt : float
        Sampling interval used for the finite-difference derivative of
        the unwrapped phase. Matches the ``timestamps`` units.
    max_iter : int
        Proximal-gradient iteration cap per row.
    tol : float
        Relative convergence tolerance on the coefficient infinity-norm.
    standardize : bool
        If ``True`` the design-matrix columns are rescaled to unit
        standard deviation before solving and the coefficients are
        back-transformed afterwards. Strongly recommended — it removes
        the dependence of ``λ`` on the phase-difference variance.
    stability_selection : bool
        If ``True``, call :func:`complementary_pairs_stability` inside
        :meth:`CouplingEstimator.estimate` and zero out edges below
        ``stability_threshold``.
    lambda_grid : tuple[float, ...] | None
        Regularisation grid for stability selection. If ``None`` a
        log-spaced grid on ``[1e-3, 1e-1]`` is used.
    n_subsamples : int
        Number of complementary-pair subsamples (each pair produces
        two fits, so the total number of fits per lambda is
        ``2 * n_subsamples``).
    subsample_fraction : float
        Fraction of rows drawn into each half of a complementary pair.
    stability_threshold : float
        Minimum selection probability for an edge to survive.
    random_state : int | None
        Seed for stability-selection resampling.
    """

    penalty: str = "mcp"
    lambda_reg: float = 0.01
    gamma: float = 3.0
    dt: float = 1.0
    max_iter: int = 500
    tol: float = 1e-5
    standardize: bool = True
    stability_selection: bool = False
    lambda_grid: tuple[float, ...] | None = None
    n_subsamples: int = 40
    subsample_fraction: float = 0.5
    stability_threshold: float = 0.6
    random_state: int | None = None

    _ALLOWED_PENALTIES: tuple[str, ...] = field(default=("mcp", "scad", "lasso"), repr=False)

    def __post_init__(self) -> None:
        if self.penalty not in self._ALLOWED_PENALTIES:
            raise ValueError(
                f"penalty must be one of {self._ALLOWED_PENALTIES}; got {self.penalty!r}"
            )
        if self.lambda_reg <= 0:
            raise ValueError("lambda_reg must be > 0")
        if self.gamma <= 1.0:
            raise ValueError("gamma must be > 1 for MCP/SCAD")
        if self.dt <= 0:
            raise ValueError("dt must be > 0")
        if self.max_iter < 1:
            raise ValueError("max_iter must be ≥ 1")
        if not 0.0 < self.subsample_fraction < 1.0:
            raise ValueError("subsample_fraction must lie in (0, 1)")
        if not 0.0 <= self.stability_threshold <= 1.0:
            raise ValueError("stability_threshold must lie in [0, 1]")


# ---------------------------------------------------------------------------
# Proximal operators
# ---------------------------------------------------------------------------


def soft_threshold(x: np.ndarray, t: float) -> np.ndarray:
    """Component-wise soft-thresholding operator (L1 prox)."""
    return np.asarray(np.sign(x) * np.maximum(np.abs(x) - t, 0.0), dtype=np.float64)


def mcp_prox(z: np.ndarray, lam: float, gamma: float, step: float) -> np.ndarray:
    """Proximal operator of the MCP penalty (Breheny & Huang, 2011).

    The MCP is

        P_λ(t) = λ|t| − t² / (2γ)  for |t| ≤ γλ,
                 γλ² / 2            otherwise.

    For step-size ``step = 1/L`` the proximal map has the closed form

    - ``|z| ≤ step·λ``:              ``0``
    - ``step·λ < |z| ≤ γλ``:         ``sign(z)·(|z| − step·λ) / (1 − step/γ)``
    - ``|z| > γλ``:                  ``z``

    and requires ``step < γ`` for the middle branch to stay contractive.
    """
    if step >= gamma:
        # Fall back to soft-thresholding on the offending coordinates;
        # in practice the row solver always picks ``step = 1/L`` small enough.
        return soft_threshold(z, step * lam)
    abs_z = np.abs(z)
    out = np.where(
        abs_z <= step * lam,
        0.0,
        np.where(
            abs_z <= gamma * lam,
            np.sign(z) * (abs_z - step * lam) / (1.0 - step / gamma),
            z,
        ),
    )
    return np.asarray(out, dtype=np.float64)


def scad_prox(z: np.ndarray, lam: float, gamma: float, step: float) -> np.ndarray:
    """Proximal operator of the SCAD penalty (Fan & Li, 2001).

    Three-region closed form; ``γ > 2`` is required for the middle
    branch to be contractive. Outside that assumption the operator
    degrades gracefully to soft-thresholding.
    """
    if gamma <= 2.0 or step >= (gamma - 1.0):
        return soft_threshold(z, step * lam)
    abs_z = np.abs(z)
    sign_z = np.sign(z)
    out = np.where(
        abs_z <= step * lam + lam,
        soft_threshold(z, step * lam),
        np.where(
            abs_z <= gamma * lam,
            sign_z * (abs_z - step * gamma * lam / (gamma - 1.0)) / (1.0 - step / (gamma - 1.0)),
            z,
        ),
    )
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# Row solver
# ---------------------------------------------------------------------------


def _apply_prox(z: np.ndarray, lam: float, step: float, penalty: str, gamma: float) -> np.ndarray:
    if penalty == "lasso":
        return soft_threshold(z, step * lam)
    if penalty == "mcp":
        return mcp_prox(z, lam, gamma, step)
    if penalty == "scad":
        return scad_prox(z, lam, gamma, step)
    raise ValueError(f"Unknown penalty {penalty!r}")


def _proximal_gradient_row(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    *,
    penalty: str = "mcp",
    gamma: float = 3.0,
    max_iter: int = 500,
    tol: float = 1e-5,
) -> np.ndarray:
    """Solve ``min_β (1/(2n))||y − Xβ||² + P_λ(β)`` via proximal gradient.

    Uses the ISTA update with a constant step size set from the
    spectral-radius upper bound ``L = ‖XᵀX‖₂ / n``. For small matrices
    (``N−1 ≲ 500``) we compute ``L`` exactly via ``numpy.linalg.svd``;
    for larger designs the row solver is still fine but you may wish to
    substitute a power-iteration estimate.
    """
    n, p = X.shape
    # Exact Lipschitz constant of the smooth gradient
    # ∇f(β) = (1/n) X^T (X β − y), so L = λ_max(X^T X) / n
    if p == 0:
        return np.zeros(0, dtype=np.float64)
    # Use SVD for a numerically robust singular value
    sigma_max = float(np.linalg.svd(X, compute_uv=False)[0])
    L = (sigma_max**2) / n
    if L <= 0:
        return np.zeros(p, dtype=np.float64)
    step = 1.0 / L

    beta = np.zeros(p, dtype=np.float64)
    Xt = X.T
    for _ in range(max_iter):
        grad = (Xt @ (X @ beta - y)) / n
        z = beta - step * grad
        beta_new = _apply_prox(z, lam, step, penalty, gamma)
        diff = float(np.max(np.abs(beta_new - beta)))
        scale = max(
            1.0, float(np.max(np.abs(beta)))
        )  # bounds: normalisation floor avoids division by near-zero beta
        beta = beta_new
        if diff < tol * scale:
            break
    return beta


# ---------------------------------------------------------------------------
# Design-matrix construction
# ---------------------------------------------------------------------------


def _build_target_and_design(
    theta: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-oscillator target ``y`` and shared design ``X``.

    ``theta`` is the *wrapped* phase matrix from
    :class:`PhaseMatrix` in ``[0, 2π)``. We unwrap along the time axis,
    differentiate once, and subtract the temporal median to remove the
    natural frequency.

    Returns
    -------
    y : np.ndarray, shape ``(T, N)``
        Frequency-deviation target per oscillator.
    sin_diff : np.ndarray, shape ``(T, N, N)``
        ``sin(θ_j − θ_i)`` with ``j`` on the last axis. Used to form the
        row-specific design ``X_i`` via slicing.
    omega_nat : np.ndarray, shape ``(N,)``
        Natural-frequency estimate per oscillator (median instantaneous
        frequency).
    """
    unwrapped = np.unwrap(theta, axis=0)
    omega_inst = np.gradient(unwrapped, dt, axis=0)  # (T, N)
    omega_nat = np.median(omega_inst, axis=0)  # (N,)
    y = omega_inst - omega_nat[np.newaxis, :]  # (T, N)
    # sin_diff[t, i, j] = sin(θ_j[t] − θ_i[t])
    diff = theta[:, np.newaxis, :] - theta[:, :, np.newaxis]
    sin_diff = np.sin(diff)
    return (
        y.astype(np.float64),
        sin_diff.astype(np.float64),
        omega_nat.astype(np.float64),
    )


def _row_design(sin_diff: np.ndarray, i: int) -> np.ndarray:
    """Extract ``X_i`` — columns ``sin(θ_j − θ_i)`` for ``j ≠ i``."""
    N = sin_diff.shape[1]
    cols = [j for j in range(N) if j != i]
    return sin_diff[:, i, cols]  # (T, N-1)


def _standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale columns of ``X`` to unit standard deviation.

    Returns the scaled matrix and the per-column scale factors so that
    the solved coefficients can be back-transformed (``β_original =
    β_scaled / scale``).
    """
    scale = X.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return X / scale, scale


# ---------------------------------------------------------------------------
# Single-shot estimation
# ---------------------------------------------------------------------------


def _estimate_K_single(
    theta: np.ndarray,
    lam: float,
    cfg: CouplingEstimationConfig,
) -> np.ndarray:
    """Fit coupling rows at a fixed ``λ`` without stability selection."""
    y_all, sin_diff, _ = _build_target_and_design(theta, cfg.dt)
    T, N = y_all.shape
    K = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        Xi = _row_design(sin_diff, i)
        yi = y_all[:, i]
        if cfg.standardize:
            Xi_s, x_scale = _standardise(Xi)
            y_scale = float(np.std(yi, ddof=0))
            if y_scale < 1e-12:
                y_scale = 1.0
            yi_s = (yi - float(np.mean(yi))) / y_scale
        else:
            Xi_s, x_scale = Xi, np.ones(Xi.shape[1])
            y_scale = 1.0
            yi_s = yi
        beta = _proximal_gradient_row(
            Xi_s,
            yi_s,
            lam,
            penalty=cfg.penalty,
            gamma=cfg.gamma,
            max_iter=cfg.max_iter,
            tol=cfg.tol,
        )
        # Back-transform: β_physical = β_scaled * std(y) / std(X_j)
        beta = beta * y_scale / x_scale
        # Scatter back into K with the diagonal hole preserved
        idx = 0
        for j in range(N):
            if j == i:
                continue
            K[i, j] = beta[idx]
            idx += 1
    return K


# ---------------------------------------------------------------------------
# Complementary-pairs stability selection
# ---------------------------------------------------------------------------


def complementary_pairs_stability(
    theta: np.ndarray,
    cfg: CouplingEstimationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Complementary-pairs stability selection (Shah & Samworth, 2013).

    For each λ in ``cfg.lambda_grid`` we draw ``cfg.n_subsamples``
    complementary pairs (two disjoint halves of the time axis) and fit
    the row regression on each half independently. The selection
    probability for edge ``(i, j)`` at lambda ``λ`` is

        π_{ij}(λ) = (# halves where |K̂_{ij}| > 0) / (2 * n_subsamples)

    The score returned is the supremum of ``π_{ij}`` over the grid.

    Returns
    -------
    K_median : np.ndarray
        Median of the non-zero edge estimates across the full set of
        fits (useful as a bias-reduced weight estimate).
    stability : np.ndarray
        Per-edge selection probability in ``[0, 1]``. Diagonal is zero.
    """
    T, N = theta.shape
    rng = np.random.default_rng(cfg.random_state)
    if cfg.lambda_grid is None:
        lam_grid: np.ndarray = np.logspace(-3, -1, 10)
    else:
        lam_grid = np.asarray(cfg.lambda_grid, dtype=np.float64)

    half = int(cfg.subsample_fraction * T)
    if half < 10:
        raise ValueError(f"Subsample half size {half} too small; increase T or subsample_fraction")

    selection_count = np.zeros((len(lam_grid), N, N), dtype=np.int64)
    weight_sum = np.zeros((N, N), dtype=np.float64)
    weight_n = np.zeros((N, N), dtype=np.int64)
    n_halves = 0  # count of independent halves processed

    for _ in range(cfg.n_subsamples):
        idx = rng.permutation(T)
        half_a = np.sort(idx[:half])
        half_b = np.sort(idx[half : 2 * half])
        for half_idx in (half_a, half_b):
            theta_sub = theta[half_idx]
            n_halves += 1
            for li, lam in enumerate(lam_grid):
                K_hat = _estimate_K_single(theta_sub, float(lam), cfg)
                active = np.abs(K_hat) > 1e-10
                selection_count[li] += active
                weight_sum += np.where(active, K_hat, 0.0)
                weight_n += active
    # Selection probability per (λ, edge): fraction of halves in which
    # the edge was selected at that regularisation level
    probs = selection_count / max(n_halves, 1)
    stability = probs.max(axis=0).astype(np.float64)
    np.fill_diagonal(stability, 0.0)

    K_median = np.where(
        weight_n > 0,
        weight_sum / np.maximum(weight_n, 1),
        0.0,
    )
    np.fill_diagonal(K_median, 0.0)
    return K_median, stability


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


class CouplingEstimator:
    """Sparse signed coupling estimator for Sakaguchi–Kuramoto identification.

    Example
    -------
    >>> cfg = CouplingEstimationConfig(penalty="mcp", lambda_reg=0.02)
    >>> est = CouplingEstimator(cfg)
    >>> coupling = est.estimate(phase_matrix)
    >>> coupling.K.shape
    (N, N)
    """

    def __init__(self, config: CouplingEstimationConfig | None = None) -> None:
        self.config = config or CouplingEstimationConfig()

    def estimate(self, phases: PhaseMatrix) -> CouplingMatrix:
        """Run the full estimation pipeline and return a validated
        :class:`CouplingMatrix`.

        If ``config.stability_selection`` is enabled, the returned
        matrix uses the median of the bootstrap weight estimates and
        carries per-edge stability scores on ``stability_scores``.
        Otherwise the matrix contains the single-``λ`` ISTA solution.
        """
        cfg = self.config
        theta = np.asarray(phases.theta, dtype=np.float64)

        if cfg.stability_selection:
            K_median, stability = complementary_pairs_stability(theta, cfg)
            K = np.where(stability >= cfg.stability_threshold, K_median, 0.0)
            np.fill_diagonal(K, 0.0)
            stability_scores: np.ndarray | None = stability
        else:
            K = _estimate_K_single(theta, cfg.lambda_reg, cfg)
            np.fill_diagonal(K, 0.0)
            stability_scores = None

        n_off_diag = K.size - K.shape[0]
        sparsity = float(1.0 - np.count_nonzero(K) / n_off_diag) if n_off_diag else 1.0

        # Sign consistency warning (methodology M1.3 Step 5)
        nz_mask = K != 0.0
        antisym = nz_mask & nz_mask.T & (np.sign(K) != np.sign(K.T))
        n_antisym_pairs = int(antisym.sum() // 2)
        n_bidirectional_pairs = int((nz_mask & nz_mask.T).sum() // 2)
        if n_bidirectional_pairs and n_antisym_pairs > 0.3 * n_bidirectional_pairs:
            logger.warning(
                "High antisymmetric ratio: %d of %d bidirectional pairs have opposite signs — "
                "verify data quality or confirm competitive dynamics are expected",
                n_antisym_pairs,
                n_bidirectional_pairs,
            )

        return CouplingMatrix(
            K=K,
            asset_ids=phases.asset_ids,
            sparsity=sparsity,
            method=cfg.penalty,
            stability_scores=stability_scores,
        )


def estimate_coupling(
    phases: PhaseMatrix,
    config: CouplingEstimationConfig | None = None,
) -> CouplingMatrix:
    """Functional shortcut around :class:`CouplingEstimator`."""
    return CouplingEstimator(config).estimate(phases)


# ---------------------------------------------------------------------------
# Second-order (swing) identification path — CALIB-GRID-001 R1
# ---------------------------------------------------------------------------
#
# The first-order path above solves
#
#     θ̇_i − ω̄_i = Σ_{j≠i} K_ij sin(θ_j − θ_i) + ε_i .
#
# Power-grid rotors are *second-order* (swing equation):
#
#     m_i θ̈_i + d_i θ̇_i = P_i − Σ_{j≠i} K_ij sin(θ_i − θ_j) .
#
# Applying the first-order identifier to swing data folds the unmodelled
# inertial term ``m_i θ̈_i`` into the residual, biasing ``K̂``. The path
# below regresses the *full* swing identity instead. With known per-node
# inertia ``m_i`` and damping ``d_i`` the model is linear in the unknowns
# ``(K_i·, P_i)``:
#
#     y_i(t) ≡ m_i θ̈_i(t) + d_i θ̇_i(t)
#            = P_i + Σ_{j≠i} (−K_ij) · sin(θ_i(t) − θ_j(t)) .
#
# A column of ones recovers the injection ``P_i`` jointly (so the natural
# frequency ``ω_i = P_i / d_i`` is *not* assumed known), and the negated
# phase-difference sines recover the signed, symmetric coupling.


class PersistentExcitationError(RuntimeError):
    """Raised when the swing design matrix is rank-deficient.

    A phase-locked trajectory has ``θ_i − θ_j → const`` so the regressor
    columns ``sin(θ_i − θ_j)`` become near-collinear: the per-node design
    Gram matrix loses rank and the least-squares solution is dominated by
    noise. Emitting a ``K̂`` from such a design would be a misleading
    instrument output, so the swing path fails closed with this typed
    diagnostic instead (instrument-honesty invariant, not a tuning knob).

    Attributes
    ----------
    node : int
        Index of the oscillator whose row design first failed the test.
    singular_ratio : float
        Smallest-to-largest singular-value ratio of the standardised
        design at ``node`` (the persistent-excitation diagnostic).
    threshold : float
        The configured minimum acceptable ``singular_ratio``.
    """

    def __init__(self, node: int, singular_ratio: float, threshold: float) -> None:
        self.node = node
        self.singular_ratio = singular_ratio
        self.threshold = threshold
        super().__init__(
            f"persistent-excitation guard tripped at node {node}: "
            f"design singular-value ratio {singular_ratio:.3e} < "
            f"threshold {threshold:.3e} — the trajectory is phase-locked / "
            f"rank-deficient and no trustworthy K̂ can be identified "
            f"(fail-closed, no silent biased estimate)"
        )


@dataclass(frozen=True, slots=True)
class SwingCouplingEstimate:
    """Result of the second-order (swing) identification path.

    Attributes
    ----------
    K : np.ndarray
        Shape ``(N, N)``, signed coupling with zero diagonal. The grid
        coupling is physically symmetric; the raw row solution is *not*
        symmetrised here (the caller decides), but the swing identity
        removes the inertial bias that produced the large antisymmetric
        residual under the first-order path.
    injection : np.ndarray
        Shape ``(N,)``, recovered per-node net power injection ``P_i``.
    omega : np.ndarray
        Shape ``(N,)``, natural frequency ``ω_i = P_i / d_i`` consistent
        with the over-damped reduction (Dörfler–Bullo SI Eq. (S15)).
    min_singular_ratio : float
        Worst per-node persistent-excitation diagnostic over all rows
        (smallest-to-largest singular-value ratio of the standardised
        design). Reported for transparency even when the guard passes.
    """

    K: NDArray[np.float64]
    injection: NDArray[np.float64]
    omega: NDArray[np.float64]
    min_singular_ratio: float


def _finite_difference_derivatives(
    theta: NDArray[np.float64], dt: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Second-order central finite differences of the unwrapped phase.

    Returns ``(θ̇, θ̈)`` with shape ``(T, N)``. The phase is unwrapped
    along the time axis first so the derivative is taken on the
    continuous lift, not the wrapped ``[0, 2π)`` representation. Both
    derivatives use :func:`numpy.gradient` (second-order accurate in the
    interior, first-order at the two endpoints).
    """
    unwrapped = np.unwrap(theta, axis=0)
    theta_dot = np.gradient(unwrapped, dt, axis=0)
    theta_ddot = np.gradient(theta_dot, dt, axis=0)
    return (
        np.asarray(theta_dot, dtype=np.float64),
        np.asarray(theta_ddot, dtype=np.float64),
    )


def _singular_ratio(design: NDArray[np.float64]) -> float:
    """Smallest-to-largest singular-value ratio of a design matrix.

    This is the reciprocal condition number; it is ``0`` for a
    rank-deficient design and ``1`` for a perfectly conditioned one. It
    is the principled persistent-excitation diagnostic (it does not
    depend on an absolute scale once the columns are standardised).
    """
    if design.shape[1] == 0:
        return 0.0
    sv = np.linalg.svd(design, compute_uv=False)
    s_max = float(sv[0])
    if s_max <= 0.0:
        return 0.0
    return float(sv[-1] / s_max)


def estimate_swing_coupling(
    phases: PhaseMatrix,
    inertia: NDArray[np.float64],
    damping: NDArray[np.float64],
    *,
    dt: float,
    pe_min_singular_ratio: float = 1e-3,
    pe_guard: bool = True,
) -> SwingCouplingEstimate:
    r"""Identify ``K_ij``, ``P_i`` and ``ω_i`` from second-order swing data.

    Solves, per oscillator ``i``, the ordinary least-squares regression

    .. math::

        m_i\,\ddot\theta_i(t) + d_i\,\dot\theta_i(t)
            \;=\; P_i \;+\; \sum_{j\neq i} (-K_{ij})\,
                \sin\!\bigl(\theta_i(t) - \theta_j(t)\bigr) ,

    where the derivatives are obtained by second-order central finite
    differences of the unwrapped phase. The design is standardised per
    column before the solve and the coefficients are back-transformed,
    so the persistent-excitation diagnostic is scale-free.

    Parameters
    ----------
    phases : PhaseMatrix
        Wrapped phase trajectory (the contract guarantees ``[0, 2π)``).
    inertia, damping : np.ndarray
        Per-node ``m_i > 0`` and ``d_i ≥ 0`` (length ``N``). These are
        the *known* machine constants of the swing model; the coupling
        and injection are the unknowns.
    dt : float
        Sampling interval matching ``phases.timestamps`` units.
    pe_min_singular_ratio : float
        Minimum acceptable smallest-to-largest singular-value ratio of
        any per-node standardised design. Below this the design is
        treated as rank-deficient (phase-locked input).
    pe_guard : bool
        If ``True`` (default) a sub-threshold design raises
        :class:`PersistentExcitationError` (fail-closed). If ``False``
        the diagnostic is still reported on the result but the
        (untrustworthy) estimate is returned anyway — for diagnostic
        sweeps only.

    Returns
    -------
    SwingCouplingEstimate
        Signed ``K``, recovered injection ``P``, natural frequency
        ``ω = P / d`` and the worst persistent-excitation ratio.

    Raises
    ------
    ValueError
        On contract violations (shape / sign / ``dt``) — fail-closed,
        no silent repair.
    PersistentExcitationError
        When ``pe_guard`` is set and any row design is rank-deficient.
    """
    if dt <= 0.0:
        raise ValueError("dt must be > 0")
    if not 0.0 < pe_min_singular_ratio < 1.0:
        raise ValueError("pe_min_singular_ratio must lie in (0, 1)")

    theta = np.asarray(phases.theta, dtype=np.float64)
    _, n = theta.shape
    m = np.asarray(inertia, dtype=np.float64)
    d = np.asarray(damping, dtype=np.float64)
    if m.shape != (n,) or d.shape != (n,):
        raise ValueError(f"inertia and damping must both have shape ({n},)")
    if not np.all(np.isfinite(m)) or not np.all(np.isfinite(d)):
        raise ValueError("inertia and damping must be finite")
    if np.any(m <= 0.0):
        raise ValueError("inertia must be strictly positive")
    if np.any(d < 0.0):
        raise ValueError("damping must be non-negative")

    theta_dot, theta_ddot = _finite_difference_derivatives(theta, dt)

    k_hat = np.zeros((n, n), dtype=np.float64)
    injection = np.zeros(n, dtype=np.float64)
    worst_ratio = 1.0

    for i in range(n):
        # Target: y_i = m_i θ̈_i + d_i θ̇_i
        y = m[i] * theta_ddot[:, i] + d[i] * theta_dot[:, i]
        # Regressors: [sin(θ_i − θ_j)]_{j≠i} and a constant for P_i.
        others = [j for j in range(n) if j != i]
        sin_cols = np.sin(theta[:, i][:, None] - theta[:, others])  # (T, n-1)
        design = np.column_stack([sin_cols, np.ones(theta.shape[0])])

        # Standardise the sine columns (the intercept column is left as
        # a unit constant so its coefficient stays in physical units).
        sin_scale = sin_cols.std(axis=0, ddof=0)
        # bounds: a near-constant regressor (phase-locked pair) gets a
        # unit scale so the PE diagnostic — not a divide-by-zero — flags
        # the degeneracy.
        sin_scale_safe = np.where(sin_scale > 1e-12, sin_scale, 1.0)
        design_std = design.copy()
        design_std[:, :-1] = sin_cols / sin_scale_safe

        ratio = _singular_ratio(design_std)
        worst_ratio = min(worst_ratio, ratio)
        if pe_guard and ratio < pe_min_singular_ratio:
            raise PersistentExcitationError(i, ratio, pe_min_singular_ratio)

        coef_std, *_ = np.linalg.lstsq(design_std, y, rcond=None)
        # Back-transform the standardised sine coefficients.
        coef = coef_std.copy()
        coef[:-1] = coef_std[:-1] / sin_scale_safe

        # y = P_i + Σ (−K_ij) sin(θ_i − θ_j)  ⇒  K_ij = −coef_j .
        for col, j in enumerate(others):
            k_hat[i, j] = -coef[col]
        injection[i] = coef[-1]

    # ω_i = P_i / d_i (over-damped reduction). Guard d_i = 0 explicitly.
    with np.errstate(divide="ignore", invalid="ignore"):
        omega = np.where(d > 0.0, injection / np.where(d > 0.0, d, 1.0), 0.0)

    return SwingCouplingEstimate(
        K=np.asarray(k_hat, dtype=np.float64),
        injection=np.asarray(injection, dtype=np.float64),
        omega=np.asarray(omega, dtype=np.float64),
        min_singular_ratio=float(worst_ratio),
    )
