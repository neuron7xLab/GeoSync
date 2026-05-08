# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Detection-side metrics — precision / recall / FPR / FNR / lead-time.

§ 14 of the canonical R&D checklist forbids AUC-only reporting.
This module ships the conventional confusion-matrix metrics plus
a strict pre-event-window lead-time aggregator. Every undefined
denominator yields NaN — never silently zero — to refuse fake
quality on empty input.

Pure-function API. No I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ClassificationMetrics",
    "LeadTimeConfig",
    "LeadTimeMetrics",
    "compute_classification_metrics",
    "compute_lead_time_metrics",
]


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Confusion-matrix-derived metrics.

    NaN policy (per § 8 of the agent calibration spec):

    * ``precision``  → NaN when ``tp + fp == 0`` (no predictions).
    * ``recall``     → NaN when ``tp + fn == 0`` (no positives).
    * ``false_positive_rate``  → NaN when ``fp + tn == 0``.
    * ``false_negative_rate``  → NaN when ``fn + tp == 0``.

    NaN, not 0.0 — the absence of denominators must propagate.
    """

    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    false_positive_rate: float
    false_negative_rate: float


def compute_classification_metrics(
    y_true: NDArray[np.bool_],
    y_pred: NDArray[np.bool_],
) -> ClassificationMetrics:
    """Confusion-matrix metrics from boolean truth / prediction arrays."""
    t = np.asarray(y_true, dtype=np.bool_)
    p = np.asarray(y_pred, dtype=np.bool_)
    if t.shape != p.shape or t.ndim != 1:
        raise ValueError(
            f"y_true and y_pred must be 1-D with matching shape; "
            f"got y_true={t.shape}, y_pred={p.shape}"
        )
    tp = int(np.logical_and(t, p).sum())
    fp = int(np.logical_and(~t, p).sum())
    tn = int(np.logical_and(~t, ~p).sum())
    fn = int(np.logical_and(t, ~p).sum())
    pred_pos = tp + fp
    cond_pos = tp + fn
    cond_neg = fp + tn
    precision = float(tp / pred_pos) if pred_pos > 0 else float("nan")
    recall = float(tp / cond_pos) if cond_pos > 0 else float("nan")
    fpr = float(fp / cond_neg) if cond_neg > 0 else float("nan")
    fnr = float(fn / cond_pos) if cond_pos > 0 else float("nan")
    return ClassificationMetrics(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
    )


@dataclass(frozen=True, slots=True)
class LeadTimeConfig:
    """Pre-registered lead-time parameters.

    A signal counts as a *valid early warning* iff it fires inside
    the pre-event window ``[event - max_lead_days, event - min_lead_days]``.
    Signals on or after the event date never count
    (post-event contamination is a hard fail).

    Attributes
    ----------
    min_lead_days
        Minimum gap (days) between alarm and event for the alarm
        to count. ``min_lead_days = 1`` excludes same-day signals.
        ``min_lead_days = 0`` includes same-day signals — discouraged
        for pre-event detection but supported for completeness.
    max_lead_days
        Maximum gap (days). The pre-event window is closed at
        both ends; an alarm older than ``max_lead_days`` does not
        count as a warning *for this event*.
    event_exclusion_days_after
        Buffer days *after* the event during which alarms are
        ignored entirely (post-event contamination guard).
        Defaults to 0 — the strict pre-event-only contract.
    """

    min_lead_days: int
    max_lead_days: int
    event_exclusion_days_after: int = 0

    def __post_init__(self) -> None:
        if self.min_lead_days < 0:
            raise ValueError(f"min_lead_days must be >= 0, got {self.min_lead_days}")
        if self.max_lead_days < self.min_lead_days:
            raise ValueError(
                f"max_lead_days ({self.max_lead_days}) must be >= "
                f"min_lead_days ({self.min_lead_days})"
            )
        if self.max_lead_days < 1:
            raise ValueError(f"max_lead_days must be >= 1, got {self.max_lead_days}")
        if self.event_exclusion_days_after < 0:
            raise ValueError(
                f"event_exclusion_days_after must be >= 0, got {self.event_exclusion_days_after}"
            )


@dataclass(frozen=True, slots=True)
class LeadTimeMetrics:
    """Aggregate lead-time profile across a set of events."""

    event_count: int
    detected_event_count: int
    lead_times: tuple[int, ...]
    median_lead_time: float
    min_lead_time: int | None
    max_lead_time: int | None


def _first_valid_lead(
    score: NDArray[np.float64],
    dates: tuple[date, ...],
    *,
    threshold: float,
    event_start: date,
    config: LeadTimeConfig,
) -> int | None:
    """Return the earliest valid pre-event lead time in days, or None."""
    s = np.asarray(score, dtype=np.float64)
    if s.shape != (len(dates),):
        raise ValueError(f"score length {s.size} != dates length {len(dates)}")
    window_lo = event_start - timedelta(days=config.max_lead_days)
    window_hi = event_start - timedelta(days=config.min_lead_days)
    # Iterate in chronological order; first match is the earliest
    # valid pre-event alarm — aligns with the "use first valid signal"
    # contract.
    for i, d in enumerate(dates):
        if d < window_lo:
            continue
        if d > window_hi:
            break
        if not np.isfinite(s[i]):
            continue
        if s[i] >= threshold:
            return int((event_start - d).days)
    return None


def compute_lead_time_metrics(
    score: NDArray[np.float64],
    dates: tuple[date, ...],
    *,
    threshold: float,
    event_dates: tuple[date, ...],
    config: LeadTimeConfig,
) -> LeadTimeMetrics:
    """Aggregate lead-time profile across a set of pre-registered events.

    ``event_dates`` is the canonical labelled set; dates outside
    ``dates`` (or with no valid pre-event window inside the score
    range) count as *undetected* events. Same-day and post-event
    alarms never count regardless of how high they spike.
    """
    if config.min_lead_days < 0 or config.max_lead_days < 1:
        raise ValueError("LeadTimeConfig invariants must already be satisfied")
    leads: list[int] = []
    for ev in event_dates:
        lead = _first_valid_lead(
            score,
            dates,
            threshold=threshold,
            event_start=ev,
            config=config,
        )
        if lead is not None:
            leads.append(lead)
    n_events = len(event_dates)
    n_detected = len(leads)
    median_lt = float(np.median(leads)) if leads else float("nan")
    return LeadTimeMetrics(
        event_count=n_events,
        detected_event_count=n_detected,
        lead_times=tuple(sorted(leads)),
        median_lead_time=median_lt,
        min_lead_time=min(leads) if leads else None,
        max_lead_time=max(leads) if leads else None,
    )
