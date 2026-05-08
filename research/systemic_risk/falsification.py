# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Pre-registered falsification battery for the interbank-Kuramoto hypothesis.

The hypothesis under test (``HYPOTHESIS`` tier per ``CLAIMS.md``):

    The early-warning score (``research.systemic_risk.early_warning``)
    is *elevated* in pre-event windows preceding banking-crisis dates
    compared to null windows drawn from the same series at safe
    distance from any event.

Decision rule (pre-registered, written *before* any data is run)::

    For each crisis c with at least one valid pre-event window:
        AUC_c   = ROC AUC of (pre-event scores vs null-window scores)
        p_c     = one-sided permutation p-value
                  (alternative: pre-event > null)

    Across crises:
        p_BH    = Benjamini-Hochberg corrected p-values (FDR control)

    HARD FAIL  : ∃ c with AUC_c <= ``fail_auc`` (default 0.55)
                  → archive negative; close the hypothesis.
    HARD PASS  : at least 2 crises with AUC_c >= ``pass_auc`` AND
                  p_BH_c <= ``pass_alpha`` (defaults 0.70, 0.01).
    UNDECIDED  : neither — collect more data, do not promote tier.

The verdict is *encoded once* and never edited after seeing data — any
subsequent change to ``fail_auc``, ``pass_auc`` or ``pass_alpha`` is a
new pre-registration on a fresh branch.

Pure-function API. No I/O. Determinism via explicit ``seed``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .event_ledger import BankingCrisisEvent, BankingCrisisLedger

__all__ = [
    "FalsificationConfig",
    "CrisisOutcome",
    "FalsificationReport",
    "auc_mann_whitney",
    "benjamini_hochberg",
    "run_falsification",
]


Verdict = Literal["HARD_FAIL", "HARD_PASS", "UNDECIDED"]


# ---------------------------------------------------------------------------
# Statistics primitives
# ---------------------------------------------------------------------------


def auc_mann_whitney(positives: NDArray[np.float64], negatives: NDArray[np.float64]) -> float:
    """ROC AUC via the Mann-Whitney U statistic.

    AUC = P(X_pos > X_neg) + 0.5 * P(X_pos == X_neg).
    Returns 0.5 when either array is empty (uninformative, by convention).
    """
    if positives.size == 0 or negatives.size == 0:
        return 0.5
    pos = positives[np.isfinite(positives)]
    neg = negatives[np.isfinite(negatives)]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    combined = np.concatenate([pos, neg])
    # Average ranks over ties (rankdata midrank).
    _, inv, counts = np.unique(combined, return_inverse=True, return_counts=True)
    raw_ranks = combined.argsort().argsort().astype(np.float64) + 1.0
    sums = np.bincount(inv, weights=raw_ranks)
    avg = sums / counts
    ranks_avg = avg[inv]
    rank_sum_pos = float(ranks_avg[: pos.size].sum())
    n_pos = pos.size
    n_neg = neg.size
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _permutation_p(
    positives: NDArray[np.float64],
    negatives: NDArray[np.float64],
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    """One-sided permutation p-value: P(AUC_perm >= AUC_observed | exchangeable)."""
    pos = positives[np.isfinite(positives)]
    neg = negatives[np.isfinite(negatives)]
    if pos.size == 0 or neg.size == 0:
        return 1.0
    observed = auc_mann_whitney(pos, neg)
    pool = np.concatenate([pos, neg])
    n_pos = pos.size
    exceedances = 0
    for _ in range(n_permutations):
        perm = rng.permutation(pool)
        a = auc_mann_whitney(perm[:n_pos], perm[n_pos:])
        if a >= observed:
            exceedances += 1
    # Davison & Hinkley (1997) +1 continuity correction.
    return float((exceedances + 1) / (n_permutations + 1))


def benjamini_hochberg(p_values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Benjamini-Hochberg FDR-adjusted p-values (1995).

    Adjusted p_(i) = min_{k>=i} (m / k) * p_(k), clipped to [0, 1].
    Order is preserved relative to the input.
    """
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"p_values must be 1-D, got shape={p.shape}")
    if p.size == 0:
        return p.copy()
    if np.any(p < 0) or np.any(p > 1) or not np.isfinite(p).all():
        raise ValueError("p_values must be finite values in [0, 1]")
    m = p.size
    order = p.argsort()
    ranked = p[order]
    factors = np.arange(1, m + 1, dtype=np.float64)
    adjusted_sorted = ranked * (m / factors)
    # Enforce monotone non-decreasing along reverse-sorted indices.
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    out = np.empty_like(p)
    out[order] = adjusted_sorted
    return out


# ---------------------------------------------------------------------------
# Window-extraction helpers
# ---------------------------------------------------------------------------


def _date_to_index(target: date, dates: tuple[date, ...]) -> int | None:
    """Return the index of the *latest* date <= ``target``, or ``None`` if all later."""
    out: int | None = None
    for i, d in enumerate(dates):
        if d <= target:
            out = i
        else:
            break
    return out


def _pre_event_window(
    score: NDArray[np.float64],
    dates: tuple[date, ...],
    event_start: date,
    window_days: int,
) -> NDArray[np.float64]:
    """Extract the score values strictly *before* the event, length ``window_days``."""
    end_idx = _date_to_index(event_start - timedelta(days=1), dates)
    if end_idx is None:
        return np.empty(0, dtype=np.float64)
    start_idx = max(0, end_idx - window_days + 1)
    return score[start_idx : end_idx + 1]


def _null_windows(
    score: NDArray[np.float64],
    dates: tuple[date, ...],
    ledger: BankingCrisisLedger,
    *,
    window_days: int,
    min_distance_days: int,
    rng: np.random.Generator,
    n_windows: int,
    country_filter: str | None,
) -> NDArray[np.float64]:
    """Sample non-overlapping null windows at safe distance from any event."""
    if score.size != len(dates):
        raise ValueError(f"score length {score.size} != dates length {len(dates)}")
    if score.size < window_days:
        return np.empty(0, dtype=np.float64)

    relevant_events: tuple[BankingCrisisEvent, ...]
    if country_filter is None:
        relevant_events = ledger.events
    else:
        relevant_events = ledger.by_country(country_filter)

    def _too_close(window_end_idx: int) -> bool:
        window_start_date = dates[window_end_idx - window_days + 1]
        window_end_date = dates[window_end_idx]
        for ev in relevant_events:
            buffer = timedelta(days=min_distance_days)
            ev_lo = ev.start - buffer
            ev_hi = ev.end + buffer
            if not (window_end_date < ev_lo or window_start_date > ev_hi):
                return True
        return False

    valid_ends: list[int] = []
    for end_idx in range(window_days - 1, score.size):
        if not _too_close(end_idx):
            valid_ends.append(end_idx)
    if not valid_ends:
        return np.empty(0, dtype=np.float64)

    take = min(n_windows, len(valid_ends))
    chosen = rng.choice(np.asarray(valid_ends, dtype=np.int64), size=take, replace=False)
    flat = np.concatenate([score[int(c) - window_days + 1 : int(c) + 1] for c in chosen])
    return flat


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FalsificationConfig:
    """Pre-registered falsification protocol.

    Attributes
    ----------
    pre_event_window_days
        Length of the pre-event window in days (default 60).
    null_window_count
        Number of null windows to draw per run (default 30).
    min_distance_from_event_days
        Buffer between any null window and any event interval
        (default 365: a full year of separation).
    n_permutations
        Permutations for the AUC p-value (default 5000).
    fail_auc
        Hard-fail threshold (default 0.55).
    pass_auc
        Hard-pass AUC threshold (default 0.70).
    pass_alpha
        Hard-pass BH-adjusted p threshold (default 0.01).
    seed
        Deterministic RNG seed (default 42).
    """

    pre_event_window_days: int = 60
    null_window_count: int = 30
    min_distance_from_event_days: int = 365
    n_permutations: int = 5000
    fail_auc: float = 0.55
    pass_auc: float = 0.70
    pass_alpha: float = 0.01
    seed: int = 42

    def __post_init__(self) -> None:
        if self.pre_event_window_days < 4:
            raise ValueError(
                f"pre_event_window_days must be >= 4, got {self.pre_event_window_days}"
            )
        if self.null_window_count < 2:
            raise ValueError(f"null_window_count must be >= 2, got {self.null_window_count}")
        if self.min_distance_from_event_days < 0:
            raise ValueError(
                f"min_distance_from_event_days must be >= 0, "
                f"got {self.min_distance_from_event_days}"
            )
        if self.n_permutations < 100:
            raise ValueError(f"n_permutations must be >= 100, got {self.n_permutations}")
        if not 0.0 < self.fail_auc < self.pass_auc <= 1.0:
            raise ValueError(
                f"thresholds must satisfy 0 < fail_auc < pass_auc <= 1; "
                f"got fail={self.fail_auc}, pass={self.pass_auc}"
            )
        if not 0.0 < self.pass_alpha <= 0.1:
            raise ValueError(f"pass_alpha must be in (0, 0.1], got {self.pass_alpha}")


@dataclass(frozen=True, slots=True)
class CrisisOutcome:
    """Per-crisis outcome of the falsification run."""

    label: str
    country: str
    event_start: date
    n_pre_event: int
    n_null: int
    auc: float
    p_value: float
    p_bh: float


@dataclass(frozen=True, slots=True)
class FalsificationReport:
    """Top-level falsification verdict + per-crisis evidence."""

    outcomes: tuple[CrisisOutcome, ...]
    verdict: Verdict
    config: FalsificationConfig


def run_falsification(
    score: NDArray[np.float64],
    dates: tuple[date, ...],
    ledger: BankingCrisisLedger,
    *,
    config: FalsificationConfig | None = None,
    country_filter: str | None = None,
) -> FalsificationReport:
    """Run the pre-registered falsification battery.

    Parameters
    ----------
    score
        Time-resolved early-warning score, shape ``(T,)``.
    dates
        Length-``T`` tuple of calendar dates, monotonically increasing.
    ledger
        :class:`BankingCrisisLedger` of crisis events.
    config
        Optional :class:`FalsificationConfig`; defaults to the
        pre-registered settings.
    country_filter
        If given (ISO-3 code), restrict events and null-distance
        masking to that country only.
    """
    cfg = config if config is not None else FalsificationConfig()
    s = np.asarray(score, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError(f"score must be 1-D, got shape={s.shape}")
    if s.size != len(dates):
        raise ValueError(f"score length {s.size} != dates length {len(dates)}")
    for i in range(1, len(dates)):
        if dates[i] <= dates[i - 1]:
            raise ValueError(
                f"dates must be strictly increasing; "
                f"dates[{i - 1}]={dates[i - 1]} >= dates[{i}]={dates[i]}"
            )

    rng = np.random.default_rng(cfg.seed)

    if country_filter is None:
        candidate_events = ledger.events
    else:
        candidate_events = ledger.by_country(country_filter)

    outcomes_in_order: list[CrisisOutcome] = []
    p_per_crisis: list[float] = []

    for ev in candidate_events:
        pre = _pre_event_window(s, dates, ev.start, cfg.pre_event_window_days)
        pre = pre[np.isfinite(pre)]
        if pre.size < 4:
            continue
        nulls = _null_windows(
            s,
            dates,
            ledger,
            window_days=cfg.pre_event_window_days,
            min_distance_days=cfg.min_distance_from_event_days,
            rng=rng,
            n_windows=cfg.null_window_count,
            country_filter=country_filter,
        )
        nulls = nulls[np.isfinite(nulls)]
        if nulls.size < 4:
            continue
        a = auc_mann_whitney(pre, nulls)
        p = _permutation_p(pre, nulls, n_permutations=cfg.n_permutations, rng=rng)
        outcomes_in_order.append(
            CrisisOutcome(
                label=ev.label,
                country=ev.country,
                event_start=ev.start,
                n_pre_event=int(pre.size),
                n_null=int(nulls.size),
                auc=a,
                p_value=p,
                p_bh=float("nan"),
            )
        )
        p_per_crisis.append(p)

    if not outcomes_in_order:
        return FalsificationReport(
            outcomes=tuple(),
            verdict="UNDECIDED",
            config=cfg,
        )

    p_bh = benjamini_hochberg(np.asarray(p_per_crisis, dtype=np.float64))
    finalised = tuple(
        CrisisOutcome(
            label=o.label,
            country=o.country,
            event_start=o.event_start,
            n_pre_event=o.n_pre_event,
            n_null=o.n_null,
            auc=o.auc,
            p_value=o.p_value,
            p_bh=float(p_bh[i]),
        )
        for i, o in enumerate(outcomes_in_order)
    )

    if any(o.auc <= cfg.fail_auc for o in finalised):
        verdict: Verdict = "HARD_FAIL"
    else:
        passing = [o for o in finalised if o.auc >= cfg.pass_auc and o.p_bh <= cfg.pass_alpha]
        verdict = "HARD_PASS" if len(passing) >= 2 else "UNDECIDED"

    return FalsificationReport(outcomes=finalised, verdict=verdict, config=cfg)
