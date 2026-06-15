# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Calibrate a Lyapunov prefetch threshold with replayable metadata.

Input CSV columns:
- step (int)
- lyapunov (float)
- regime_label (str)

The calibration protocol is descriptor-only. It selects a bounded threshold from a
declared candidate grid using only the calibration split, then records holdout
validation metrics and a sensitivity sweep. The output is not financial advice,
not predictive evidence, and not a trading instruction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class Point:
    step: int
    lyapunov: float
    label: str


@dataclass(frozen=True, slots=True)
class ThresholdMetrics:
    mean_lead: float
    coverage: int
    score: float


@dataclass(frozen=True, slots=True)
class SensitivityRow:
    candidate: float
    calibration_metric: float
    validation_metric: float
    passes: bool
    failure_reason: str


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    method: str
    parameter_name: str
    candidate_values: tuple[float, ...]
    selected_value: float
    metric_name: str
    metric_values: dict[str, float]
    calibration_count: int
    validation_count: int
    invalid_count: int
    seed: int
    tie_policy: str
    sensitivity_band: tuple[float, float]
    failure_policy: str
    not_financial_advice: bool
    not_predictive_claim: bool
    calibration_dataset: str
    validation_dataset: str
    objective: str
    allowed_range: tuple[float, float]
    selection_rule: str
    sensitivity_sweep: tuple[SensitivityRow, ...]


def load_points(path: Path) -> list[Point]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"step", "lyapunov", "regime_label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        points: list[Point] = []
        for row in reader:
            step = int(row["step"])
            lyapunov = float(row["lyapunov"])
            if not math.isfinite(lyapunov):
                raise ValueError("lyapunov must be finite")
            points.append(
                Point(
                    step=step,
                    lyapunov=lyapunov,
                    label=str(row["regime_label"]),
                )
            )
        return points


def transition_steps(points: list[Point]) -> list[int]:
    out: list[int] = []
    for prev, cur in zip(points, points[1:]):
        if prev.label != cur.label:
            out.append(cur.step)
    return out


def evaluate_threshold(points: list[Point], threshold: float) -> tuple[float, int]:
    metrics = threshold_metrics(points, threshold)
    return metrics.mean_lead, metrics.coverage


def threshold_metrics(points: list[Point], threshold: float) -> ThresholdMetrics:
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")

    transitions = transition_steps(points)
    if not transitions:
        return ThresholdMetrics(mean_lead=0.0, coverage=0, score=0.0)

    lead_times: list[int] = []
    for transition in transitions:
        candidates = [
            p.step for p in points if p.step < transition and p.lyapunov >= threshold
        ]
        if not candidates:
            continue
        lead_times.append(transition - max(candidates))

    if not lead_times:
        return ThresholdMetrics(mean_lead=0.0, coverage=0, score=0.0)

    mean_lead = sum(lead_times) / len(lead_times)
    return ThresholdMetrics(
        mean_lead=mean_lead,
        coverage=len(lead_times),
        score=mean_lead * len(lead_times),
    )


def build_candidate_grid(
    min_threshold: float,
    max_threshold: float,
    step_size: float,
) -> tuple[float, ...]:
    if not all(
        math.isfinite(value) for value in (min_threshold, max_threshold, step_size)
    ):
        raise ValueError("candidate range values must be finite")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if min_threshold > max_threshold:
        raise ValueError("min_threshold must be <= max_threshold")

    values: list[float] = []
    current = min_threshold
    while current <= max_threshold + 1e-12:
        values.append(round(current, 12))
        current += step_size
    if not values:
        raise ValueError("candidate grid must not be empty")
    return tuple(values)


def validate_candidates(
    candidates: Iterable[float],
    allowed_range: tuple[float, float],
) -> tuple[float, ...]:
    lower, upper = allowed_range
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ValueError("allowed_range must be finite and ordered")

    seen: set[float] = set()
    out: list[float] = []
    for candidate in candidates:
        if not math.isfinite(candidate):
            raise ValueError("candidate threshold must be finite")
        rounded = round(float(candidate), 12)
        if rounded < lower or rounded > upper:
            raise ValueError("candidate threshold outside allowed range")
        if rounded not in seen:
            seen.add(rounded)
            out.append(rounded)
    if not out:
        raise ValueError("candidate grid must not be empty")
    return tuple(sorted(out))


def split_points(
    points: list[Point],
    *,
    calibration_fraction: float,
) -> tuple[list[Point], list[Point], str, str]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")
    if len(points) < 4:
        raise ValueError("at least four points are required for split calibration")

    ordered = sorted(points, key=lambda point: point.step)
    split_at = int(len(ordered) * calibration_fraction)
    split_at = max(1, min(split_at, len(ordered) - 1))

    calibration = ordered[:split_at]
    validation = ordered[split_at:]
    if not calibration or not validation:
        raise ValueError("calibration and validation splits must be non-empty")
    if calibration[-1].step >= validation[0].step:
        raise ValueError("calibration split must precede validation split")

    calibration_id = f"steps:{calibration[0].step}-{calibration[-1].step}"
    validation_id = f"steps:{validation[0].step}-{validation[-1].step}"
    return calibration, validation, calibration_id, validation_id


def calibrate_threshold(
    points: list[Point],
    *,
    candidates: Iterable[float],
    allowed_range: tuple[float, float] = (0.0, 1.0),
    calibration_fraction: float = 0.6,
    seed: int = 0,
) -> CalibrationResult:
    """Select a descriptor threshold using calibration split metrics only."""

    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    candidate_values = validate_candidates(candidates, allowed_range)
    calibration, validation, calibration_id, validation_id = split_points(
        points,
        calibration_fraction=calibration_fraction,
    )

    calibration_scores: dict[float, ThresholdMetrics] = {
        candidate: threshold_metrics(calibration, candidate)
        for candidate in candidate_values
    }
    if all(metrics.coverage == 0 for metrics in calibration_scores.values()):
        raise ValueError("calibration split has no threshold coverage")

    selected_value = min(
        candidate_values,
        key=lambda candidate: (-calibration_scores[candidate].score, candidate),
    )
    selected_calibration = calibration_scores[selected_value]
    selected_validation = threshold_metrics(validation, selected_value)

    sweep: list[SensitivityRow] = []
    for candidate in candidate_values:
        calibration_metric = calibration_scores[candidate].score
        validation_metric = threshold_metrics(validation, candidate).score
        passes = calibration_scores[candidate].coverage > 0
        sweep.append(
            SensitivityRow(
                candidate=candidate,
                calibration_metric=calibration_metric,
                validation_metric=validation_metric,
                passes=passes,
                failure_reason="" if passes else "no_calibration_coverage",
            )
        )

    return CalibrationResult(
        method="fixed_grid_holdout_sensitivity",
        parameter_name="lyapunov_prefetch_threshold",
        candidate_values=candidate_values,
        selected_value=selected_value,
        metric_name="mean_lead_times_coverage",
        metric_values={
            "selected_calibration_metric": selected_calibration.score,
            "selected_validation_metric": selected_validation.score,
        },
        calibration_count=len(calibration),
        validation_count=len(validation),
        invalid_count=0,
        seed=seed,
        tie_policy="lowest_threshold_on_equal_calibration_score",
        sensitivity_band=(candidate_values[0], candidate_values[-1]),
        failure_policy="raise_value_error_fail_closed",
        not_financial_advice=True,
        not_predictive_claim=True,
        calibration_dataset=calibration_id,
        validation_dataset=validation_id,
        objective="maximize calibration split mean lead times covered transitions",
        allowed_range=allowed_range,
        selection_rule="select highest calibration metric; validation is reported only",
        sensitivity_sweep=tuple(sweep),
    )


def result_to_dict(result: CalibrationResult) -> dict[str, object]:
    data = asdict(result)
    data["sensitivity_sweep"] = [asdict(row) for row in result.sensitivity_sweep]
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--min", dest="min_threshold", type=float, default=0.4)
    parser.add_argument("--max", dest="max_threshold", type=float, default=0.9)
    parser.add_argument("--step", dest="step_size", type=float, default=0.01)
    parser.add_argument("--calibration-fraction", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit calibration metadata as JSON",
    )
    args = parser.parse_args()

    points = load_points(args.csv)
    candidates = build_candidate_grid(
        args.min_threshold, args.max_threshold, args.step_size
    )
    result = calibrate_threshold(
        points,
        candidates=candidates,
        allowed_range=(args.min_threshold, args.max_threshold),
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )

    if args.json:
        print(json.dumps(result_to_dict(result), sort_keys=True))
        return

    metric = result.metric_values["selected_calibration_metric"]
    print(
        f"best_threshold={result.selected_value:.3f} "
        f"coverage={result.metric_values['selected_validation_metric']:.3f} "
        f"score={metric:.3f}"
    )


if __name__ == "__main__":
    main()
