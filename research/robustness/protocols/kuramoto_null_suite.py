# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Frozen-returns null audit bound to the Kuramoto contract.

The four-family null primitive (:mod:`research.robustness.null_audit`)
expects separate ``signal`` and ``target`` streams. The frozen evidence
bundle only ships *realised* strategy returns (no raw position trace),
so the generic primitive's three "signal-side" families degenerate.

This suite implements the two null families that are meaningful given
only a realised return stream:

1. **iid_permutation** — reshuffle the returns i.i.d.; tests for any
   deterministic information encoded in time order.
2. **stationary_bootstrap** — Politis & Romano block bootstrap with
   geometric block length (mean = 21 bars); tests information beyond
   short-horizon autocorrelation.

Positive observed Sharpe with a low p-value on both families is the
minimum evidence that the realised return stream is not a re-ordering
artefact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from .kuramoto_contract import KuramotoRobustnessContract

FrozenNullFamily = Literal["iid_permutation", "stationary_bootstrap"]
NULL_PASS_P_THRESHOLD: Final[float] = 0.05


@dataclass(frozen=True)
class FrozenNullResult:
    """One null family's bootstrap distribution and p-value."""

    family: FrozenNullFamily
    observed_sharpe: float
    null_sharpes: tuple[float, ...]
    p_value: float
    n_bootstrap: int
    p_value_pass: bool


@dataclass(frozen=True)
class KuramotoNullSuiteResult:
    """Aggregate of the frozen-returns null audit."""

    families: tuple[FrozenNullResult, ...]
    all_families_pass: bool


def _sharpe(returns: NDArray[np.float64], periods_per_year: int) -> float:
    if returns.size < 2:
        return 0.0
    std = returns.std(ddof=1)
    if std <= 0 or not np.isfinite(std):
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def _stationary_bootstrap_indices(
    n: int,
    mean_block: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=np.int64)
    current = int(rng.integers(0, n))
    for i in range(n):
        idx[i] = current
        if rng.random() < p:
            current = int(rng.integers(0, n))
        else:
            current = (current + 1) % n
    return idx


def _p_value(observed: float, null: NDArray[np.float64]) -> float:
    exceedances = int((null >= observed).sum())
    return float((exceedances + 1) / (null.size + 1))


def run_kuramoto_null_suite(
    contract: KuramotoRobustnessContract,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
    mean_block: int = 21,
    periods_per_year: int = 252,
) -> KuramotoNullSuiteResult:
    """Run the two-family frozen-returns null audit.

    Parameters
    ----------
    contract
        Verified frozen-artifact contract.
    n_bootstrap
        Null resamples per family.
    seed
        Seed for the shared PCG64 stream (deterministic audit).
    mean_block
        Mean geometric block length for stationary bootstrap.
    periods_per_year
        Sharpe annualisation factor.
    """
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    returns = contract.daily_strategy_returns().to_numpy(dtype=np.float64)
    observed = _sharpe(returns, periods_per_year)
    rng = np.random.default_rng(seed)

    null_iid = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        null_iid[b] = _sharpe(rng.permutation(returns), periods_per_year)

    null_sb = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = _stationary_bootstrap_indices(returns.size, mean_block, rng)
        null_sb[b] = _sharpe(returns[idx], periods_per_year)

    families: list[FrozenNullResult] = []
    family_name: FrozenNullFamily
    for family_name, null_dist in (
        ("iid_permutation", null_iid),
        ("stationary_bootstrap", null_sb),
    ):
        p = _p_value(observed, null_dist)
        families.append(
            FrozenNullResult(
                family=family_name,
                observed_sharpe=observed,
                null_sharpes=tuple(float(x) for x in null_dist),
                p_value=p,
                n_bootstrap=n_bootstrap,
                p_value_pass=p <= NULL_PASS_P_THRESHOLD,
            )
        )
    return KuramotoNullSuiteResult(
        families=tuple(families),
        all_families_pass=all(f.p_value_pass for f in families),
    )
