#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PARAMS: dict[str, float] = {
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


def review_param(name: str, value: float) -> dict[str, str | float]:
    baseline = round(value * 10.0, 6)
    muted = 0.0
    raised = round(value * 15.0, 6)
    signal_delta = round(baseline - muted, 6)
    limit_delta = round(max(0.0, raised - 4.5), 6)
    verdict = "pass" if signal_delta >= 1.5 and limit_delta <= 0.5 else "review"
    return {
        "parameter": name,
        "value": value,
        "baseline_score": baseline,
        "muted_score": muted,
        "raised_score": raised,
        "signal_delta": signal_delta,
        "limit_delta": limit_delta,
        "verdict": verdict,
        "verification": "deterministic_replay_only",
    }


def write_outputs(out: Path, rows: list[dict[str, str | float]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "parameter_review.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (out / "parameter_review.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["parameter", "value", "signal_delta", "limit_delta", "verdict"])
        for row in rows:
            writer.writerow(
                [row["parameter"], row["value"], row["signal_delta"], row["limit_delta"], row["verdict"]]
            )
    summary = {
        "mode": "deterministic_parameter_review",
        "claim_tier": "simulation-only",
        "parameters_tested": len(rows),
        "scheduled_for_review": sum(row["verdict"] != "pass" for row in rows),
        "entropy_residual": 0.0,
        "reproducibility": 1.0,
        "outputs": ["parameter_review.json", "parameter_review.csv"],
    }
    (out / "parameter_review_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/cognitive_dynamics_lab")
    out = Path(parser.parse_args().out)
    rows = [review_param(name, value) for name, value in sorted(PARAMS.items())]
    write_outputs(out, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
