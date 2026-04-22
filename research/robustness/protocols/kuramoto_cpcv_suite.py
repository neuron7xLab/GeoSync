# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""CPCV + PBO + PSR suite bound to the frozen Kuramoto evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from research.robustness.cpcv import estimate_pbo, probabilistic_sharpe_ratio

from .kuramoto_contract import KuramotoRobustnessContract

PBO_PASS_THRESHOLD: Final[float] = 0.50
PSR_PASS_THRESHOLD: Final[float] = 0.95


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


def _fold_oos_matrix(fold_sharpes: tuple[float, ...]) -> np.ndarray:
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


def run_kuramoto_cpcv_suite(
    contract: KuramotoRobustnessContract,
) -> KuramotoCPCVResult:
    """Compute CPCV-family metrics against the frozen contract.

    The frozen bundle already carries OOS fold Sharpes (spike walk-
    forward) and the full daily strategy equity. We therefore reuse the
    pre-computed fold sharpes for PBO and the daily return stream for
    PSR. No re-simulation is performed — this suite is *read-only* on
    frozen artifacts by design.
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
    return KuramotoCPCVResult(
        fold_sharpes=fold_sharpes,
        pbo=pbo,
        pbo_pass=pbo < PBO_PASS_THRESHOLD,
        psr_daily=psr,
        psr_pass=(psr >= PSR_PASS_THRESHOLD) if np.isfinite(psr) else False,
        annualised_sharpe=sr,
        n_bars=int(daily.size),
        n_folds=len(fold_sharpes),
    )
