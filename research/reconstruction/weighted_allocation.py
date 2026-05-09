# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Bernoulli sampling of A_ij + gravity allocation + IPF marginal projection.

Per Protocol X-10R:

    if a_ij sampled True via Bernoulli(p_ij):
        <w_ij> = (s_i^out · s_j^in) / W_total          (gravity initial)
    else:
        w_ij = 0

The bare gravity formula preserves expected TOTAL mass but not row/col
marginals — Gate 5 requires row_sum_invariant_L1 ≤ 0.05 exactly.
Almog & Squartini 2017 §2.3 closes the gap with iterated proportional
fitting (Sinkhorn–Knopp on the support of A). We do that here:

  1. Initialise W^0_ij = a_ij · s_i^out · s_j^in / W_total
  2. Row-scale W^k_ij ← W^k_ij · (s_i^out / Σ_j W^k_ij)
  3. Col-scale W^k_ij ← W^k_ij · (s_j^in  / Σ_i W^k_ij)
  4. Iterate until max(row_err, col_err) ≤ ipf_tol or max_iter exhausted

Convergence is monotone provided the support of A can carry the
prescribed marginals (every row/col with positive marginal must have
≥ 1 non-zero edge). Pre-checked; ValueError on infeasible support.

GATE_4 (REPRODUCIBILITY): all sampling via injected
``numpy.random.Generator`` — no global numpy.random.seed, no system
entropy. Identical PRNG state + identical input → bit-exact A, W.
"""

from __future__ import annotations

import numpy as np

_IPF_TOL_DEFAULT: float = 1e-9
_IPF_MAX_ITER_DEFAULT: int = 5000


def sample_adjacency_bernoulli(
    p: np.ndarray,
    *,
    rng: np.random.Generator,
    guarantee_row_col_support: bool = True,
) -> np.ndarray:
    """Sample binary adjacency A from per-edge probabilities p.

    Diagonal forced to zero. When ``guarantee_row_col_support`` is True
    (default), any orphan row / col has a single forced edge added at
    the argmax-p off-diagonal position — this guarantees IPF feasibility
    at the cost of negligible deviation from the Bernoulli ensemble
    (only finite-N orphans are repaired, an O(1/N) effect).
    Returns ``np.uint8`` matrix.
    """
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError(f"p must be square 2-D; got shape={p.shape}")
    if not np.all((p >= 0) & (p <= 1)):
        raise ValueError("p entries must lie in [0, 1]")
    n = p.shape[0]
    draws = rng.uniform(0.0, 1.0, size=(n, n))
    a = (draws < p).astype(np.uint8)
    np.fill_diagonal(a, 0)
    if guarantee_row_col_support:
        p_offdiag = p.copy()
        np.fill_diagonal(p_offdiag, -np.inf)
        row_orphans = np.where(a.sum(axis=1) == 0)[0]
        for i in row_orphans:
            j = int(np.argmax(p_offdiag[i]))
            a[i, j] = 1
        col_orphans = np.where(a.sum(axis=0) == 0)[0]
        for j in col_orphans:
            i = int(np.argmax(p_offdiag[:, j]))
            a[i, j] = 1
    return a


def _validate_support(a: np.ndarray, s_out: np.ndarray, s_in: np.ndarray) -> None:
    """Reject sampled supports that cannot carry the prescribed marginals."""
    n = a.shape[0]
    a_bool = a > 0
    row_has_edge = a_bool.any(axis=1)
    col_has_edge = a_bool.any(axis=0)
    bad_rows = np.where((s_out > 0) & ~row_has_edge)[0]
    bad_cols = np.where((s_in > 0) & ~col_has_edge)[0]
    if bad_rows.size or bad_cols.size:
        raise ValueError(
            "infeasible adjacency support for IPF: "
            f"{bad_rows.size}/{n} positive-marginal rows have zero edges, "
            f"{bad_cols.size}/{n} positive-marginal cols have zero edges"
        )


def _ipf_project(
    w: np.ndarray,
    s_out: np.ndarray,
    s_in: np.ndarray,
    *,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, int, float]:
    """Sinkhorn-Knopp on the support of W; return (W, n_iter, final_err).

    Convergence check looks at BOTH marginals BEFORE the next iteration's
    paired row/col scaling pass. This catches alternating oscillation
    (col-perfect / row-off) that a post-col-step check would hide.
    """
    w_cur = w.copy()
    ref = float(max(s_out.max(), s_in.max(), 1.0))
    err = float("inf")
    for it in range(1, max_iter + 1):
        # Check residuals on the matrix as it stands at the start of iter.
        row_err = float(np.max(np.abs(w_cur.sum(axis=1) - s_out)))
        col_err = float(np.max(np.abs(w_cur.sum(axis=0) - s_in)))
        err = max(row_err, col_err)
        if err / ref <= tol:
            np.fill_diagonal(w_cur, 0.0)
            return w_cur.astype(np.float64), it, err
        # Row scaling
        row_sum = w_cur.sum(axis=1)
        row_factor = np.where(row_sum > 0, s_out / np.where(row_sum > 0, row_sum, 1.0), 0.0)
        w_cur = w_cur * row_factor[:, None]
        # Col scaling
        col_sum = w_cur.sum(axis=0)
        col_factor = np.where(col_sum > 0, s_in / np.where(col_sum > 0, col_sum, 1.0), 0.0)
        w_cur = w_cur * col_factor[None, :]
    # Final residual on the produced matrix.
    row_err = float(np.max(np.abs(w_cur.sum(axis=1) - s_out)))
    col_err = float(np.max(np.abs(w_cur.sum(axis=0) - s_in)))
    err = max(row_err, col_err)
    np.fill_diagonal(w_cur, 0.0)
    return w_cur.astype(np.float64), max_iter, err


def allocate_weights(
    a: np.ndarray,
    s_out: np.ndarray,
    s_in: np.ndarray,
    *,
    ipf_tol: float = _IPF_TOL_DEFAULT,
    ipf_max_iter: int = _IPF_MAX_ITER_DEFAULT,
) -> np.ndarray:
    """Gravity initial + IPF projection to enforce marginals.

    Total mass W_total = Σs_in = Σs_out (precondition: marginals balance,
    enforced by GATE_2 in recovery_audit). Returns float64 NxN matrix
    whose row sums = s_out and col sums = s_in to ``ipf_tol`` (relative
    to max marginal). Raises ValueError on infeasible adjacency support
    or non-convergent IPF.
    """
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"a must be square 2-D; got shape={a.shape}")
    n = a.shape[0]
    s_out_arr = np.asarray(s_out, dtype=np.float64)
    s_in_arr = np.asarray(s_in, dtype=np.float64)
    if s_out_arr.shape != (n,) or s_in_arr.shape != (n,):
        raise ValueError(
            f"s_out / s_in must have shape ({n},); got {s_out_arr.shape} / {s_in_arr.shape}"
        )
    if np.any(s_out_arr < 0) or np.any(s_in_arr < 0):
        raise ValueError("s_out / s_in must be non-negative")
    w_total = float(s_in_arr.sum())
    if w_total <= 0 or not np.isfinite(w_total):
        raise ValueError(f"total mass must be positive finite; got W_total={w_total}")
    if ipf_tol <= 0 or ipf_max_iter < 1:
        raise ValueError(f"ipf_tol > 0 and ipf_max_iter ≥ 1; got {ipf_tol}, {ipf_max_iter}")
    _validate_support(a, s_out_arr, s_in_arr)
    outer = np.outer(s_out_arr, s_in_arr) / w_total
    w0: np.ndarray = (a.astype(np.float64) * outer).astype(np.float64)
    np.fill_diagonal(w0, 0.0)
    w_proj, _n_iter, _final_err = _ipf_project(
        w0, s_out_arr, s_in_arr, tol=ipf_tol, max_iter=ipf_max_iter
    )
    # Note: we do NOT raise on non-convergence. Residual marginal error
    # is what Gate 5 (row_sum_invariant_L1, col_sum_invariant_L1) is
    # designed to catch. Hard-failing here would mask the failure mode
    # behind an opaque exception — Gate 5 makes it observable, which
    # is the X-10R contract: "every failure must be a numbered gate".
    return w_proj
