# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""CALIB-GRID-001 runner — emits the machine-readable result ledger.

Usage::

    PYTHONPATH=. python -m research.calibration.grid_kuramoto.run \
        --system wscc9 --out research/calibration/grid_kuramoto/RESULTS.json

The verdict is computed *only* from the frozen gates in
:mod:`research.calibration.grid_kuramoto.gates`. No threshold is defined
here; this module just orchestrates and serialises (fail-closed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

from .calibration import SimConfig, run_calibration
from .gates import (
    NOISELESS_GATES,
    NOISY_GATES,
    evaluate_gates,
    overall_verdict,
)
from .grid_data import GridSystem, ieee_39_new_england, wscc_9_bus

__all__ = ["build_ledger", "build_r1_ledger", "main"]

_SYSTEMS: dict[str, Any] = {
    "wscc9": wscc_9_bus,
    "ieee39": ieee_39_new_england,
}


def _branch_sha() -> str:
    """Best-effort current commit sha for the ledger provenance field."""
    head = Path(__file__).resolve().parents[3] / ".git" / "HEAD"
    try:
        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref:"):
            ref_path = head.parent / ref.split(" ", 1)[1]
            return ref_path.read_text(encoding="utf-8").strip()
        return ref
    except OSError:
        return "unknown"


def build_ledger(system: GridSystem, cfg: SimConfig) -> dict[str, Any]:
    """Run both regimes, evaluate the frozen gates, return the ledger dict."""
    noiseless = run_calibration(system, cfg, noisy=False)
    noisy = run_calibration(system, cfg, noisy=True)

    nl_gates = evaluate_gates(noiseless, NOISELESS_GATES)
    ny_gates = evaluate_gates(noisy, NOISY_GATES)
    all_gates = nl_gates + ny_gates
    verdict = overall_verdict(all_gates)

    failed = [g.to_dict() for g in all_gates if not g.passed]

    ledger: dict[str, Any] = {
        "artifact": "CALIB-GRID-001",
        "kind": "external-ground-truth-calibration",
        "is_hypothesis": False,
        "system": system.name,
        "citation": system.citation,
        "branch_sha": _branch_sha(),
        "python": platform.python_version(),
        "config": {
            "coupling_scale": cfg.coupling_scale,
            "dt": cfg.dt,
            "steps": cfg.steps,
            "keep_frac": cfg.keep_frac,
            "theta0_perturb": cfg.theta0_perturb,
            "seed": cfg.seed,
            "noise_sigma": cfg.noise_sigma,
            "lambda_reg": cfg.lambda_reg,
            "penalty": cfg.penalty,
            "topology_rel_threshold": cfg.topology_rel_threshold,
        },
        "metrics": {
            "noiseless": noiseless.to_dict(),
            "noisy": noisy.to_dict(),
        },
        "gates": [g.to_dict() for g in all_gates],
        "verdict": verdict,
        "failed_gates": failed,
        "localized_refinement_targets": sorted({g.localises_to for g in all_gates if not g.passed}),
    }
    payload = json.dumps(ledger, sort_keys=True).encode("utf-8")
    ledger["ledger_sha256"] = hashlib.sha256(payload).hexdigest()
    return ledger


_FROZEN_PREREG_SHA = "d170d48afa5066c13edeb40b2c1904b3fd708516"
_PARENT_LEDGER_SHA256 = "ed8d409b7b222eb053572d6bf9ab6e98c5f4918be1cae384864733a2b4d72aaf"


def build_r1_ledger(system: GridSystem, cfg: SimConfig) -> dict[str, Any]:
    """CALIB-GRID-001 **R1** ledger — same frozen gates, swing-aware path.

    Refinement-lineage R1 changes *only* the estimator (the second-order
    swing-identification path with the symmetric joint solver and
    Savitzky–Golay derivatives). The pre-registered configuration, gate
    thresholds, seeds, σ, θ₀ perturbation and decision rule are the
    frozen ground truth from PR #749 and are *not* touched here.
    """
    noiseless = run_calibration(system, cfg, noisy=False, estimator_path="swing")
    noisy = run_calibration(system, cfg, noisy=True, estimator_path="swing")

    nl_gates = evaluate_gates(noiseless, NOISELESS_GATES)
    ny_gates = evaluate_gates(noisy, NOISY_GATES)
    all_gates = nl_gates + ny_gates
    verdict = overall_verdict(all_gates)
    failed = [g.to_dict() for g in all_gates if not g.passed]

    ledger: dict[str, Any] = {
        "artifact": "CALIB-GRID-001",
        "lineage": "R1",
        "kind": "external-ground-truth-calibration",
        "is_hypothesis": False,
        "r1_scope": "estimator-only change (second-order swing path)",
        "frozen_preregistration_sha": _FROZEN_PREREG_SHA,
        "parent_ledger_sha256": _PARENT_LEDGER_SHA256,
        "system": system.name,
        "citation": system.citation,
        "estimator": (
            "core.kuramoto.coupling_estimator.estimate_swing_coupling "
            "(symmetric joint solve, Savitzky-Golay window=7 polyorder=4, "
            "persistent-excitation guard active)"
        ),
        "branch_sha": _branch_sha(),
        "python": platform.python_version(),
        "config": {
            "coupling_scale": cfg.coupling_scale,
            "dt": cfg.dt,
            "steps": cfg.steps,
            "keep_frac": cfg.keep_frac,
            "theta0_perturb": cfg.theta0_perturb,
            "seed": cfg.seed,
            "noise_sigma": cfg.noise_sigma,
            "lambda_reg": cfg.lambda_reg,
            "penalty": cfg.penalty,
            "topology_rel_threshold": cfg.topology_rel_threshold,
        },
        "metrics": {
            "noiseless": noiseless.to_dict(),
            "noisy": noisy.to_dict(),
        },
        "gates": [g.to_dict() for g in all_gates],
        "verdict": verdict,
        "failed_gates": failed,
        "localized_refinement_targets": sorted({g.localises_to for g in all_gates if not g.passed}),
    }
    payload = json.dumps(ledger, sort_keys=True).encode("utf-8")
    ledger["ledger_sha256"] = hashlib.sha256(payload).hexdigest()
    return ledger


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on PASS, 1 on NEGATIVE (fail-closed)."""
    parser = argparse.ArgumentParser(description="CALIB-GRID-001 runner")
    parser.add_argument("--system", choices=sorted(_SYSTEMS), default="wscc9")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--r1",
        action="store_true",
        help="run the R1 refinement lineage (swing-aware estimator path)",
    )
    args = parser.parse_args(argv)

    system = _SYSTEMS[args.system]()
    cfg = SimConfig()
    ledger = build_r1_ledger(system, cfg) if args.r1 else build_ledger(system, cfg)

    text = json.dumps(ledger, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if ledger["verdict"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
