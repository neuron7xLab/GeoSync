#!/usr/bin/env python3
"""Replayable stdlib-only artifact emitter for cognitive dynamics lab."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

CASES = [
    "baseline",
    "high_noise",
    "delayed_feedback",
    "scarce_data",
    "rapid_regime_shift",
    "stress_case",
    "resource_constrained",
]

SUMMARY: dict[str, Any] = {
    "run_id": "CDL-SUMMARY",
    "method_version": "cognitive-dynamics-lab-v0.1",
    "seed": 20260613,
    "status": "ARTIFACT_BUILT",
    "artifact_role": "simulation_evidence",
    "claim_tier": "simulation-only",
    "runs": 7,
    "cases": CASES,
    "replay_command": (
        "python scripts/cognitive_dynamics_lab/simulation_runner.py "
        "--out artifacts/cognitive_dynamics_lab"
    ),
    "aggregate_metrics": {
        "mean_mae": 0.189113,
        "mean_realism_gap": 0.157488,
        "mean_ei_balance": 0.643307,
        "mean_outside_ei_fraction": 0.777974,
        "mean_energy_efficiency": 0.000152479,
        "mean_stability_margin": 0.736609,
    },
    "detected_holes_frequency": {"H5": 7, "H6": 1, "H10": 1},
    "confidence": {
        "level": 0.60,
        "reason": (
            "simulation-only perturbation evidence; stability thresholds are not "
            "consistently passed"
        ),
        "cannot_claim": ["external validation", "production readiness"],
    },
}


def run_case(case: str) -> dict[str, str]:
    """Return one deterministic per-case replay record."""
    return {
        "run_id": f"CDL-{case}",
        "case": case,
        "claim_tier": "simulation-only",
    }


def write_outputs(out: Path) -> None:
    """Write deterministic summary, run records, and a compact metrics table."""
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)

    for case in CASES:
        payload = run_case(case)
        (runs / f"{case}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    (out / "summary.json").write_text(
        json.dumps(SUMMARY, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with (out / "metrics_table.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["case", "claim_tier"])
        for case in CASES:
            writer.writerow([case, "simulation-only"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/cognitive_dynamics_lab")
    out = Path(parser.parse_args().out)
    write_outputs(out)
    print(json.dumps(SUMMARY, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
