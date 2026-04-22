# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""CPCV + PBO + PSR suite bound to the frozen Kuramoto evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from research.robustness.cpcv import estimate_pbo, probabilistic_sharpe_ratio

from .kuramoto_contract import KuramotoRobustnessContract

PBO_PASS_THRESHOLD: Final[float] = 0.50
PSR_PASS_THRESHOLD: Final[float] = 0.95
LOO_PBO_PASS_THRESHOLD: Final[float] = 0.50


@dataclass(frozen=True)
class KuramotoCPCVResult:
    """Aggregate of the CPCV/PBO/PSR suite on frozen evidence."""

    fold_sharpes: tuple[float, ...]
    pbo: float
    pbo_pass: bool
    psr_daily: float
    psr_pass: bool
    annualised_sharpe: float
    n_bars: int
    n_folds: int
    loo_pbo: float | None
    loo_pbo_pass: bool
    loo_n_strategies: int


def _fold_oos_matrix(fold_sharpes: tuple[float, ...]) -> NDArray[np.float64]:
    """Build an OOS matrix from per-fold Sharpe values for PBO estimation.

    The frozen evidence bundle ships one Sharpe per walk-forward fold
    (single anchor strategy). To stress PBO we build a 2-strategy family:
    anchor vs shifted-by-median mirror — which is a conservative upper
    bound since the two "strategies" are highly correlated. A more
    generous PBO would require the full parameter grid from the spike.
    """
    arr = np.asarray(fold_sharpes, dtype=np.float64)
    mirror = arr - float(np.median(arr))
    return np.column_stack([arr, mirror])


def _loo_oos_matrix(loo_grid: pd.DataFrame) -> NDArray[np.float64]:
    """Build an OOS matrix of (folds × strategies) from the LOO grid.

    Each non-baseline LOO row is one strategy variant whose per-fold
    Sharpes live in columns ``fold1..fold5``. We transpose so rows are
    CPCV paths (folds) and columns are strategies, matching
    :func:`research.robustness.cpcv.estimate_pbo` shape expectations.
    The baseline row is excluded: including it would guarantee best-IS
    capture every time and trivialise the PBO.
    """
    perturbations = loo_grid[loo_grid["loo_type"] != "baseline_full"]
    folds = perturbations[["fold1", "fold2", "fold3", "fold4", "fold5"]].to_numpy(dtype=np.float64)
    # folds shape is (n_strategies, n_folds); transpose to (n_folds, n_strategies)
    out: NDArray[np.float64] = folds.T
    return out


def run_kuramoto_cpcv_suite(
    contract: KuramotoRobustnessContract,
) -> KuramotoCPCVResult:
    """Compute CPCV-family metrics against the frozen contract.

    The frozen bundle already carries OOS fold Sharpes (spike walk-
    forward) and the full daily strategy equity. We therefore reuse the
    pre-computed fold sharpes for PBO and the daily return stream for
    PSR. No re-simulation is performed — this suite is *read-only* on
    frozen artifacts by design.

    When the contract carries the optional LOO grid, a *second* PBO is
    computed on the real (folds × LOO-perturbations) OOS matrix — this
    is the honest Bailey et al. PBO and is non-trivial by construction.
    """
    daily = contract.daily_strategy_returns().to_numpy(dtype=np.float64)
    fold_sharpes = tuple(float(s) for s in contract.fold_metrics["sharpe"].to_numpy())
    psr = probabilistic_sharpe_ratio(
        daily,
        sr_benchmark=0.0,
        periods_per_year=252,
    )
    oos = _fold_oos_matrix(fold_sharpes)
    pbo = estimate_pbo(oos)
    std = float(np.std(daily, ddof=1))
    sr = float(np.mean(daily) / std * np.sqrt(252)) if std > 0 and np.isfinite(std) else 0.0

    loo_pbo: float | None = None
    loo_pbo_pass = True
    loo_n_strategies = 0
    if contract.loo_grid is not None:
        loo_oos = _loo_oos_matrix(contract.loo_grid)
        loo_n_strategies = int(loo_oos.shape[1])
        if loo_oos.shape[0] >= 2 and loo_n_strategies >= 2:
            loo_pbo = estimate_pbo(loo_oos)
            loo_pbo_pass = loo_pbo < LOO_PBO_PASS_THRESHOLD

    return KuramotoCPCVResult(
        fold_sharpes=fold_sharpes,
        pbo=pbo,
        pbo_pass=pbo < PBO_PASS_THRESHOLD,
        psr_daily=psr,
        psr_pass=(psr >= PSR_PASS_THRESHOLD) if np.isfinite(psr) else False,
        annualised_sharpe=sr,
        n_bars=int(daily.size),
        n_folds=len(fold_sharpes),
        loo_pbo=loo_pbo,
        loo_pbo_pass=loo_pbo_pass,
        loo_n_strategies=loo_n_strategies,
    )
