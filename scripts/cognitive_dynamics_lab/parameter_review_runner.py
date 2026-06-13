#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ParamTable = dict[str, float]
ReviewRow = dict[str, str | float]
ReviewSummary = dict[str, str | int | float | list[str] | dict[str, str | float | int]]

DEFAULT_OUT = "artifacts/cognitive_dynamics_lab"
CLAIM_TIER = "simulation-only"
REVIEW_MODE = "deterministic_parameter_review"
VERIFICATION = "deterministic_replay_only"
UNIT_SCALE = "0..1"
SIGNAL_REFERENCE = 4.5
LIMIT_REFERENCE = 1.0
CALIBRATION_PROFILE = {
    "version": "calibration-v0.1",
    "sensitivity": "balanced",
    "low_upper": 0.25,
    "high_lower": 0.75,
    "precision_digits": 6,
    "signal_pass_min": 1.5,
    "limit_pass_max": 0.5,
}
QUANTIZED_STATES = ["low", "medium", "high"]
QUANTIZATION_BINS = {
    "low": [0.0, CALIBRATION_PROFILE["low_upper"]],
    "medium": [CALIBRATION_PROFILE["low_upper"], CALIBRATION_PROFILE["high_lower"]],
    "high": [CALIBRATION_PROFILE["high_lower"], 1.0],
}
CSV_FIELDS = [
    "parameter",
    "value",
    "normalized_value",
    "quantized_value",
    "signal_delta",
    "normalized_signal_delta",
    "quantized_signal_delta",
    "limit_delta",
    "normalized_limit_delta",
    "quantized_limit_delta",
    "calibration",
    "verdict",
]

PARAMS: ParamTable = {
    "alpha_error": 0.35,
    "alpha_explore": 0.20,
    "alpha_gain": 0.20,
    "alpha_novelty": 0.25,
    "beta_energy": 0.25,
    "beta_invalid": 0.25,
    "beta_sparsity": 0.20,
    "beta_stability": 0.30,
    "w_accuracy": 0.25,
    "w_diversity": 0.15,
    "w_energy": 0.10,
    "w_realism": 0.30,
    "w_stability": 0.20,
}


def clamp_unit(value: float) -> float:
    digits = int(CALIBRATION_PROFILE["precision_digits"])
    return round(max(0.0, min(1.0, value)), digits)


def quantize_unit(value: float) -> str:
    if value <= float(CALIBRATION_PROFILE["low_upper"]):
        return "low"
    if value < float(CALIBRATION_PROFILE["high_lower"]):
        return "medium"
    return "high"


def normalize_value(value: float) -> float:
    return clamp_unit(value)


def normalize_signal(value: float) -> float:
    return clamp_unit(value / SIGNAL_REFERENCE)


def normalize_limit(value: float) -> float:
    return clamp_unit(value / LIMIT_REFERENCE)


def quantize_value(value: float) -> str:
    return quantize_unit(normalize_value(value))


def quantize_signal(value: float) -> str:
    return quantize_unit(normalize_signal(value))


def quantize_limit(value: float) -> str:
    return quantize_unit(normalize_limit(value))


def score_baseline(value: float) -> float:
    return round(value * 10.0, 6)


def score_raised(value: float) -> float:
    return round(value * 15.0, 6)


def choose_verdict(signal_delta: float, limit_delta: float) -> str:
    signal_pass_min = float(CALIBRATION_PROFILE["signal_pass_min"])
    limit_pass_max = float(CALIBRATION_PROFILE["limit_pass_max"])
    if signal_delta >= signal_pass_min and limit_delta <= limit_pass_max:
        return "pass"
    return "review"


def review_param(name: str, value: float) -> ReviewRow:
    baseline = score_baseline(value)
    muted = 0.0
    raised = score_raised(value)
    signal_delta = round(baseline - muted, 6)
    limit_delta = round(max(0.0, raised - SIGNAL_REFERENCE), 6)
    return {
        "parameter": name,
        "value": value,
        "normalized_value": normalize_value(value),
        "quantized_value": quantize_value(value),
        "baseline_score": baseline,
        "muted_score": muted,
        "raised_score": raised,
        "signal_delta": signal_delta,
        "normalized_signal_delta": normalize_signal(signal_delta),
        "quantized_signal_delta": quantize_signal(signal_delta),
        "limit_delta": limit_delta,
        "normalized_limit_delta": normalize_limit(limit_delta),
        "quantized_limit_delta": quantize_limit(limit_delta),
        "metric_scale": UNIT_SCALE,
        "quantization_bins": str(QUANTIZATION_BINS),
        "calibration": str(CALIBRATION_PROFILE["version"]),
        "verdict": choose_verdict(signal_delta, limit_delta),
        "verification": VERIFICATION,
    }


def build_rows(params: ParamTable) -> list[ReviewRow]:
    return [review_param(name, value) for name, value in sorted(params.items())]


def build_summary(rows: list[ReviewRow]) -> ReviewSummary:
    return {
        "mode": REVIEW_MODE,
        "claim_tier": CLAIM_TIER,
        "metric_scale": UNIT_SCALE,
        "calibration_profile": CALIBRATION_PROFILE,
        "quantized_states": QUANTIZED_STATES,
        "parameters_tested": len(rows),
        "scheduled_for_review": sum(row["verdict"] != "pass" for row in rows),
        "entropy_residual": 0.0,
        "reproducibility": 1.0,
        "normalized_fields": [
            "normalized_value",
            "normalized_signal_delta",
            "normalized_limit_delta",
        ],
        "quantized_fields": [
            "quantized_value",
            "quantized_signal_delta",
            "quantized_limit_delta",
        ],
        "outputs": ["parameter_review.json", "parameter_review.csv"],
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def csv_row(row: ReviewRow) -> list[str | float]:
    return [
        row["parameter"],
        row["value"],
        row["normalized_value"],
        row["quantized_value"],
        row["signal_delta"],
        row["normalized_signal_delta"],
        row["quantized_signal_delta"],
        row["limit_delta"],
        row["normalized_limit_delta"],
        row["quantized_limit_delta"],
        row["calibration"],
        row["verdict"],
    ]


def write_csv(path: Path, rows: list[ReviewRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_FIELDS)
        for row in rows:
            writer.writerow(csv_row(row))


def write_outputs(out: Path, rows: list[ReviewRow]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "parameter_review.json", rows)
    write_csv(out / "parameter_review.csv", rows)
    summary = build_summary(rows)
    write_json(out / "parameter_review_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_rows(PARAMS)
    write_outputs(Path(args.out), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
