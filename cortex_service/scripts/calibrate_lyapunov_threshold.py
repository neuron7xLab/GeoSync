# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Calibrate Lyapunov prefetch threshold from historical traces.

Input CSV columns:
- step (int)
- lyapunov (float)
- regime_label (str)

The script searches thresholds and reports lead-time statistics (how many
steps before actual label transition the threshold fires).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Point:
    step: int
    lyapunov: float
    label: str


def load_points(path: Path) -> list[Point]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"step", "lyapunov", "regime_label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        return [
            Point(
                step=int(row["step"]),
                lyapunov=float(row["lyapunov"]),
                label=str(row["regime_label"]),
            )
            for row in reader
        ]


def transition_steps(points: list[Point]) -> list[int]:
    out: list[int] = []
    for prev, cur in zip(points, points[1:]):
        if prev.label != cur.label:
            out.append(cur.step)
    return out


def evaluate_threshold(points: list[Point], threshold: float) -> tuple[float, int]:
    transitions = transition_steps(points)
    if not transitions:
        return 0.0, 0

    lead_times: list[int] = []
    for transition in transitions:
        candidates = [
            p.step
            for p in points
            if p.step < transition and p.lyapunov >= threshold
        ]
        if not candidates:
            continue
        lead_times.append(transition - max(candidates))

    if not lead_times:
        return 0.0, 0
    return sum(lead_times) / len(lead_times), len(lead_times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--min", dest="min_threshold", type=float, default=0.4)
    parser.add_argument("--max", dest="max_threshold", type=float, default=0.9)
    parser.add_argument("--step", dest="step_size", type=float, default=0.01)
    args = parser.parse_args()

    points = load_points(args.csv)
    threshold = args.min_threshold
    best_threshold = threshold
    best_score = -1.0
    best_coverage = 0

    while threshold <= args.max_threshold + 1e-12:
        mean_lead, covered = evaluate_threshold(points, threshold)
        score = mean_lead * covered
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_coverage = covered
        threshold += args.step_size

    print(
        f"best_threshold={best_threshold:.3f} coverage={best_coverage} score={best_score:.3f}"
    )


if __name__ == "__main__":
    main()
