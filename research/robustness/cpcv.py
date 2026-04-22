# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Combinatorial Purged Cross-Validation + PBO + Probabilistic Sharpe.

Canonical references:

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*,
  Wiley. Chapter 7 (CPCV), Chapter 11 (PBO), Chapter 14 (PSR/DSR).
- Bailey, Borwein, Lopez de Prado, Zhu (2017) "The Probability of
  Backtest Overfitting", *Journal of Computational Finance*.

All functions are pure and deterministic. Inputs are numpy arrays or
pandas Series; outputs are dataclasses or scalars. No I/O.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CPCVSplit:
    """One training / testing index pair from a CPCV fold combination."""

    train_index: NDArray[np.int64]
    test_index: NDArray[np.int64]
    embargo_mask: NDArray[np.bool_]


def _purge_and_embargo(
    n: int,
    test_blocks: tuple[tuple[int, int], ...],
    embargo: int,
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Compute the train index and embargo mask for given test blocks.

    Any training sample within ``embargo`` bars of a test block is dropped
    (purged) to prevent leakage from overlapping labels.
    """
    mask = np.ones(n, dtype=bool)
    for lo, hi in test_blocks:
        mask[max(0, lo - embargo) : min(n, hi + embargo)] = False
    train_index = np.nonzero(mask)[0].astype(np.int64)
    return train_index, ~mask


def cpcv_splits(
    n_samples: int,
    n_groups: int,
    n_test_groups: int,
    embargo: int = 0,
) -> Iterator[CPCVSplit]:
    """Yield every ``C(n_groups, n_test_groups)`` purged split.

    ``n_groups`` contiguous buckets of (roughly) equal size partition the
    time axis; every combination of ``n_test_groups`` of them is a
    disjoint test set, the remainder is the training set after purging
    ``embargo`` bars around each test block.

    Parameters
    ----------
    n_samples
        Total number of timesteps.
    n_groups
        Total number of time-ordered groups (paper convention: N).
    n_test_groups
        Number of groups forming the test set each combination (k).
    embargo
        Embargo bars applied around each test block on both sides.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    if not 2 <= n_groups <= n_samples:
        raise ValueError(f"n_groups must be in [2, n_samples={n_samples}], got {n_groups}")
    if not 1 <= n_test_groups < n_groups:
        raise ValueError(
            f"n_test_groups must be in [1, n_groups-1={n_groups - 1}], got {n_test_groups}"
        )
    if embargo < 0:
        raise ValueError(f"embargo must be >= 0, got {embargo}")

    edges = np.linspace(0, n_samples, n_groups + 1, dtype=np.int64)
    group_ranges: tuple[tuple[int, int], ...] = tuple(
        (int(edges[i]), int(edges[i + 1])) for i in range(n_groups)
    )

    for test_group_ids in combinations(range(n_groups), n_test_groups):
        test_blocks = tuple(group_ranges[g] for g in test_group_ids)
        train_idx, embargo_mask = _purge_and_embargo(n_samples, test_blocks, embargo)
        test_idx = np.concatenate([np.arange(lo, hi, dtype=np.int64) for lo, hi in test_blocks])
        yield CPCVSplit(
            train_index=train_idx,
            test_index=test_idx,
            embargo_mask=embargo_mask,
        )


def estimate_pbo(oos_matrix: NDArray[np.float64]) -> float:
    """Probability of Backtest Overfitting.

    Bailey et al. (2017) logit-rank estimator. ``oos_matrix`` is shape
    (n_paths, n_strategies); each row is the OOS performance of all
    strategies on one CPCV combinatorial path. The PBO is the fraction
    of paths where the *in-sample best* strategy is below median OOS.

    Returns a scalar in [0, 1]. Lower is better; >= 0.5 is no better
    than a random selection from the strategy family.
    """
    if oos_matrix.ndim != 2:
        raise ValueError(
            f"oos_matrix must be 2-D (paths × strategies), got shape {oos_matrix.shape}"
        )
    n_paths, n_strategies = oos_matrix.shape
    if n_paths < 2 or n_strategies < 2:
        raise ValueError(
            f"PBO requires at least 2 paths and 2 strategies, got {n_paths} × {n_strategies}"
        )

    overfits = 0
    for path_ix in range(n_paths):
        is_scores = np.delete(oos_matrix, path_ix, axis=0).mean(axis=0)
        best_is = int(np.argmax(is_scores))
        oos_rank = (oos_matrix[path_ix] <= oos_matrix[path_ix, best_is]).sum()
        # Rank is 1-indexed: N_strats worst, 1 best. Below median → overfit.
        if oos_rank <= n_strategies / 2:
            overfits += 1
    return float(overfits / n_paths)


def probabilistic_sharpe_ratio(
    returns: NDArray[np.float64],
    sr_benchmark: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Probabilistic Sharpe Ratio (Lopez de Prado 2018, Eq. 14.1).

    PSR = Φ( (SR − SR*) · √(T − 1) / √(1 − γ₃·SR + (γ₄ − 1)/4 · SR²) )

    Corrects the observed Sharpe for its own sampling distribution under
    non-normal returns (skewness γ₃, kurtosis γ₄). Returns a probability
    in [0, 1] that the true Sharpe exceeds ``sr_benchmark``.

    Degenerate cases (n ≤ 1, zero variance, non-finite inputs) return NaN
    rather than raising, so aggregate reports do not crash on empty folds.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 1:
        raise ValueError(f"returns must be 1-D, got shape {r.shape}")
    n = r.size
    if n < 2 or not np.all(np.isfinite(r)):
        return math.nan
    std = r.std(ddof=1)
    if std <= 0:
        return math.nan

    mean = r.mean()
    sr = mean / std * math.sqrt(periods_per_year)
    sr_star = float(sr_benchmark)

    centered = r - mean
    m2 = float((centered**2).mean())
    if m2 <= 0:
        return math.nan
    m3 = float((centered**3).mean())
    m4 = float((centered**4).mean())
    skew = m3 / (m2**1.5)
    kurt = m4 / (m2**2)

    denom_sq = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom_sq <= 0 or not math.isfinite(denom_sq):
        return math.nan
    z = (sr - sr_star) * math.sqrt(n - 1) / math.sqrt(denom_sq)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def rolling_probabilistic_sharpe(
    returns: NDArray[np.float64],
    window: int,
    sr_benchmark: float = 0.0,
    periods_per_year: int = 252,
) -> NDArray[np.float64]:
    """Rolling PSR over a fixed window.

    Returns an array of length ``len(returns)`` with NaN for the first
    ``window - 1`` entries (analogous to ``pandas.Series.rolling``).
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 1:
        raise ValueError(f"returns must be 1-D, got shape {r.shape}")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    n = r.size
    out = np.full(n, math.nan, dtype=np.float64)
    if n < window:
        return out
    for end in range(window, n + 1):
        out[end - 1] = probabilistic_sharpe_ratio(
            r[end - window : end],
            sr_benchmark=sr_benchmark,
            periods_per_year=periods_per_year,
        )
    return out
