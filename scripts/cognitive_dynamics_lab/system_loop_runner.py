#!/usr/bin/env python3
"""Small runner that combines replay, parameter review, and timing summaries."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

DEFAULT_OUT = "artifacts/cognitive_dynamics_lab_system"
LOOP_PROFILE = {
    "version": "system-loop-v0.1",
    "metric_scale": "0..1",
    "stages": ["calibrate", "optimize", "measure", "integrate"],
    "claim_tier": "simulation-only",
}
WEIGHTS = {
    "simulation_score": 0.40,
    "parameter_score": 0.35,
    "timing_score": 0.25,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clamp_unit(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 6)


def timing_score(summary: dict[str, Any]) -> float:
    rows = summary.get("benchmarks", [])
    if not isinstance(rows, list) or not rows:
        return 0.0
    ratios = [float(row.get("relative_to_json_baseline", 1.0)) for row in rows]
    return clamp_unit(1.0 / max(1.0, max(ratios)))


def state(score: float) -> str:
    if score < 0.60:
        return "optimize"
    if score < 0.75:
        return "watch"
    return "stable"


def build_summary(samples: int, warmups: int) -> dict[str, Any]:
    root = repo_root()
    sim = load_module(root / "scripts/cognitive_dynamics_lab/simulation_runner.py", "cdl_sim")
    params = load_module(root / "scripts/cognitive_dynamics_lab/parameter_review_runner.py", "cdl_params")
    bench = load_module(root / "scripts/cognitive_dynamics_lab/benchmark_runner.py", "cdl_bench")
    sim_summary = sim.build_summary()
    rows = params.build_rows(params.PARAMS)
    param_summary = params.build_summary(rows)
    bench_summary = bench.run_suite(samples=samples, warmups=warmups)
    scores = {
        "simulation_score": float(sim_summary["objective_score"]),
        "parameter_score": float(param_summary["objective_score"]),
        "timing_score": timing_score(bench_summary),
    }
    integrated_score = clamp_unit(sum(scores[name] * WEIGHTS[name] for name in scores))
    return {
        "loop_profile": LOOP_PROFILE,
        "weights": WEIGHTS,
        "scores": scores,
        "integrated_score": integrated_score,
        "integrated_state": state(integrated_score),
        "calibration_profiles": {
            "simulation": sim_summary["calibration_profile"],
            "parameters": param_summary["calibration_profile"],
        },
        "optimization_profiles": {
            "simulation": sim_summary["optimization_profile"],
            "parameters": param_summary["optimization_profile"],
        },
        "measurements": {
            "runs": sim_summary["runs"],
            "parameters": param_summary["parameters_tested"],
            "benchmarks": bench_summary["total_benchmarks"],
            "samples": samples,
            "warmups": warmups,
        },
        "actions": param_summary["recommended_actions"],
        "status": "SYSTEM_LOOP_BUILT",
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, scores: dict[str, float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["score", "value", "weight"])
        for name, value in scores.items():
            writer.writerow([name, value, WEIGHTS[name]])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = build_summary(samples=args.samples, warmups=args.warmups)
    write_json(out / "system_loop_summary.json", summary)
    write_csv(out / "system_loop_scores.csv", summary["scores"])
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
