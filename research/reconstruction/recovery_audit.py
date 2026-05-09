# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""GATE_5 — reconstruction recovery audit on synthetic ground truth.

Per Protocol X-10R:

    GATE_5  RECONSTRUCTION_RECOVERY    (positive_control.py)
        - on each synthetic ground-truth substrate:
          reconstruct from aggregated marginals, compare to truth:
            spectral_radius_recovery: |ρ_recon − ρ_true|/ρ_true ≤ 0.20
            top_k_hub_jaccard (k = N//10):                    ≥ 0.60
            row_sum_invariant_L1:           ≤ 0.05 × mean_strength
            col_sum_invariant_L1:           ≤ 0.05 × mean_strength

Any threshold violated → INVALID_RECONSTRUCTION. This is the
PRECONDITION for any real-data interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Recovery thresholds — matched 1:1 to Protocol X-10R Gate 5.
RECOVERY_THRESHOLDS: dict[str, float] = {
    "spectral_radius_relative_error_max": 0.20,
    "top_k_hub_jaccard_min": 0.60,
    "row_sum_invariant_L1_relative_max": 0.05,
    "col_sum_invariant_L1_relative_max": 0.05,
}


@dataclass(frozen=True)
class RecoveryReport:
    spectral_radius_true: float
    spectral_radius_recon: float
    spectral_radius_relative_error: float
    top_k_hub_jaccard: float
    row_sum_invariant_L1: float
    col_sum_invariant_L1: float
    mean_strength: float
    k_top: int
    passed: bool
    failure_reasons: tuple[str, ...]


def _spectral_radius(w: np.ndarray) -> float:
    """ρ(W) = max |eigenvalue|."""
    if w.size == 0 or not np.any(w):
        return 0.0
    eigs = np.linalg.eigvals(w.astype(np.float64))
    return float(np.max(np.abs(eigs)))


def _top_k_jaccard(deg_true: np.ndarray, deg_recon: np.ndarray, *, k: int) -> float:
    if k <= 0:
        return 0.0
    top_t = set(np.argsort(-deg_true)[:k].tolist())
    top_r = set(np.argsort(-deg_recon)[:k].tolist())
    union = top_t | top_r
    if not union:
        return 0.0
    return float(len(top_t & top_r) / len(union))


def audit_recovery(
    w_true: np.ndarray,
    w_recon: np.ndarray,
    *,
    k_top: int | None = None,
) -> RecoveryReport:
    """Compare reconstructed W against ground-truth W on Gate 5 metrics."""
    if w_true.shape != w_recon.shape or w_true.ndim != 2:
        raise ValueError(
            f"w_true and w_recon must be matching square 2-D; got {w_true.shape} / {w_recon.shape}"
        )
    n = w_true.shape[0]
    k = k_top if k_top is not None else max(1, n // 10)
    rho_true = _spectral_radius(w_true)
    rho_recon = _spectral_radius(w_recon)
    rho_rel_err = abs(rho_recon - rho_true) / rho_true if rho_true > 0 else float("inf")
    s_out_true = w_true.sum(axis=1)
    s_out_recon = w_recon.sum(axis=1)
    s_in_true = w_true.sum(axis=0)
    s_in_recon = w_recon.sum(axis=0)
    # Hub definition: total strength s_out + s_in (Battiston-Caldarelli DebtRank
    # convention for weighted systemic networks). Binary degree is meaningless
    # in the bank-exposure setting where one $1B link is a bigger systemic
    # hub than fifty $1M links. The reconstruction is calibrated against
    # marginal strengths, so this is the observable that Gate 5 must check.
    strength_true = (s_out_true + s_in_true).astype(np.float64)
    strength_recon = (s_out_recon + s_in_recon).astype(np.float64)
    jacc = _top_k_jaccard(strength_true, strength_recon, k=k)
    mean_strength = float((s_out_true + s_in_true).mean() / 2.0)
    if mean_strength == 0:
        mean_strength = 1.0  # avoid div-by-zero
    row_l1 = float(np.abs(s_out_true - s_out_recon).sum() / n / mean_strength)
    col_l1 = float(np.abs(s_in_true - s_in_recon).sum() / n / mean_strength)
    failures: list[str] = []
    if rho_rel_err > RECOVERY_THRESHOLDS["spectral_radius_relative_error_max"]:
        failures.append(
            f"spectral_radius_relative_error={rho_rel_err:.3f} > "
            f"{RECOVERY_THRESHOLDS['spectral_radius_relative_error_max']}"
        )
    if jacc < RECOVERY_THRESHOLDS["top_k_hub_jaccard_min"]:
        failures.append(
            f"top_k_hub_jaccard={jacc:.3f} < {RECOVERY_THRESHOLDS['top_k_hub_jaccard_min']}"
        )
    if row_l1 > RECOVERY_THRESHOLDS["row_sum_invariant_L1_relative_max"]:
        failures.append(
            f"row_sum_invariant_L1={row_l1:.3f} > "
            f"{RECOVERY_THRESHOLDS['row_sum_invariant_L1_relative_max']}"
        )
    if col_l1 > RECOVERY_THRESHOLDS["col_sum_invariant_L1_relative_max"]:
        failures.append(
            f"col_sum_invariant_L1={col_l1:.3f} > "
            f"{RECOVERY_THRESHOLDS['col_sum_invariant_L1_relative_max']}"
        )
    return RecoveryReport(
        spectral_radius_true=rho_true,
        spectral_radius_recon=rho_recon,
        spectral_radius_relative_error=rho_rel_err,
        top_k_hub_jaccard=jacc,
        row_sum_invariant_L1=row_l1,
        col_sum_invariant_L1=col_l1,
        mean_strength=mean_strength,
        k_top=k,
        passed=not failures,
        failure_reasons=tuple(failures),
    )


def conservation_of_mass_passes(
    s_out: np.ndarray, s_in: np.ndarray, *, tol_rel: float = 1e-9
) -> bool:
    """GATE_2 — |Σs_in − Σs_out| / Σs_in < tol_rel."""
    s_out_arr = np.asarray(s_out, dtype=np.float64)
    s_in_arr = np.asarray(s_in, dtype=np.float64)
    sum_in = float(s_in_arr.sum())
    sum_out = float(s_out_arr.sum())
    if sum_in <= 0:
        return False
    return abs(sum_in - sum_out) / sum_in < tol_rel
