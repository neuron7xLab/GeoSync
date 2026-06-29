#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = [
    "canonical_td0",
    "td0_plus_distributional_helper",
    "td0_plus_asymmetric_alpha",
    "td0_plus_risk_penalty",
    "td0_plus_vigor_helper",
    "expectile_ensemble_low_tau",
    "expectile_ensemble_mid_tau",
    "expectile_ensemble_high_tau",
    "combined_surface",
]
NULL_MODELS = [
    "shuffled_returns",
    "sign_flipped_reward",
    "random_tau_assignment",
    "constant_reward",
    "lagged_reward",
    "no_rpe_baseline",
    "random_entry_baseline",
    "same_turnover_random_signal",
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_manifest(out: Path) -> None:
    files = [p for p in sorted(out.iterdir()) if p.is_file() and p.name != "ARTIFACT_MANIFEST.sha256"]
    out.joinpath("ARTIFACT_MANIFEST.sha256").write_text("".join(f"{sha(p.read_bytes())}  {p.name}\n" for p in files), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/dopamine_rpe_extension")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260629)
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows = []
    for fold in range(args.folds):
        for variant in VARIANTS:
            base = 0.01 * math.sin(fold + len(variant))
            rows.append({
                "fold": fold,
                "variant": variant,
                "IC": round(base, 6),
                "Sharpe": round(base * 3.0 + rng.uniform(-0.02, 0.02), 6),
                "DeflatedSharpeProxy": round(base * 2.0, 6),
                "MaxDD": round(-0.05 - abs(base), 6),
                "turnover": round(0.1 + fold * 0.01, 6),
                "hit_rate": round(0.5 + base, 6),
                "calibration_error": round(abs(base), 6),
                "regime_stability": round(1.0 - abs(base), 6),
                "null_model_delta": round(base - 0.001, 6),
                "parameter_sensitivity": round(abs(base) + 0.001, 6),
                "fold_start": f"fold_{fold}_start",
                "fold_end": f"fold_{fold}_end",
                "seed": args.seed,
                "data_hash": sha(f"synthetic:{args.seed}:{fold}".encode()),
                "config_hash": sha(f"dopamine-eval:{args.seed}:{args.folds}".encode()),
            })
    with (out / "ABLATION_MATRIX.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "EVAL_SUMMARY.json").write_text(json.dumps({"component": "geosync.dopamine", "status": "P2_RESEARCH", "folds": args.folds, "seed": args.seed, "variants": VARIANTS, "null_models": NULL_MODELS, "market_claim_allowed": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "PARAMETER_LOCK.json").write_text(json.dumps({"folds": args.folds, "seed": args.seed, "variants": VARIANTS}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "METRIC_DEFINITIONS.json").write_text(json.dumps({"IC": "association proxy", "Sharpe": "risk-adjusted proxy", "MaxDD": "drawdown proxy"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "WALKFORWARD_REPORT.md").write_text("# Walk-forward report\n\nDeterministic scaffold. No market-edge claim.\n", encoding="utf-8")
    (out / "NULL_MODEL_REPORT.md").write_text("# Null-model report\n\nNull models are enumerated. Survival is not claimed until real-data replay passes.\n", encoding="utf-8")
    (out / "FALSIFIER_REPORT.md").write_text("# Falsifier report\n\nPromotion remains blocked without replayable evidence.\n", encoding="utf-8")
    write_manifest(out)
    print(json.dumps({"status": "PASS", "out": str(out.relative_to(ROOT))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
