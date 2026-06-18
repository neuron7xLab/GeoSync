#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Deterministic bootstrap physics scoring oracle.

This is not a marketing score. It is a fail-closed repository evidence scanner.
It computes a static baseline from files, claim carriers, test carriers, and explicit
blockers. Runtime physics tests still have to be executed separately by pytest.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TARGET_INTERVAL = [88.0, 92.0]


@dataclass(frozen=True)
class Metric:
    id: str
    weight: float
    score: float
    metric: str
    measurement_fn: str
    threshold: str
    failure_condition: str
    status: str
    evidence: tuple[str, ...]

    @property
    def weighted_score(self) -> float:
        return round(self.weight * self.score, 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "weight": self.weight,
            "score": self.score,
            "weighted_score": self.weighted_score,
            "metric": self.metric,
            "measurement_fn": self.measurement_fn,
            "threshold": self.threshold,
            "failure_condition": self.failure_condition,
            "status": self.status,
            "evidence": list(self.evidence),
        }


def exists(path: str) -> bool:
    return (ROOT / path).exists()


def text(path: str) -> str:
    p = ROOT / path
    if not p.exists() or not p.is_file():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


def contains(path: str, *needles: str) -> bool:
    body = text(path)
    return all(needle in body for needle in needles)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def git_branch() -> str:
    try:
        return subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def build_metrics() -> list[Metric]:
    geosync_core = exists("docs/physics/geosync_core.md")
    t1_code = exists("core/kuramoto/kuramoto_ricci_engine.py")
    t1_tests = exists("tests/unit/physics/test_T1_kuramoto_ricci_boundary.py")
    claim_ledger = exists("docs/audit/claim_ledger.md")
    file_inventory = exists("docs/audit/file_inventory.md")

    has_rhs = contains(
        "core/kuramoto/kuramoto_ricci_engine.py",
        "kuramoto_ricci_rhs",
        "phase_transition_boundary",
        "order_parameter",
    )
    has_nulls = contains(
        "tests/unit/physics/test_T1_kuramoto_ricci_boundary.py",
        "test_negative_control_zero_coupling_phases_drift_apart",
        "test_negative_control_omega_nonzero_potential_can_increase",
    )
    has_boundary_tests = contains(
        "tests/unit/physics/test_T1_kuramoto_ricci_boundary.py",
        "test_phase_transition_boundary_complete_graph",
        "test_INV_KR1_subcritical_phi_negative_yields_low_R",
        "test_INV_KR1_supercritical_phi_positive_yields_high_R",
    )

    metrics: list[Metric] = [
        Metric(
            id="S_math_object",
            weight=0.12,
            score=70.0 if geosync_core and t1_code and has_rhs else 0.0,
            metric="canonical mathematical object has equation/code carriers",
            measurement_fn="exists(docs/physics/geosync_core.md) && contains(T1 code symbols)",
            threshold=">= 90 for final rank",
            failure_condition="no single canonical object or missing RHS/boundary/order parameter code",
            status="PARTIAL_PASS" if geosync_core and t1_code and has_rhs else "FAIL",
            evidence=("docs/physics/geosync_core.md", "core/kuramoto/kuramoto_ricci_engine.py"),
        ),
        Metric(
            id="S_dimensional_consistency",
            weight=0.10,
            score=35.0 if geosync_core else 0.0,
            metric="units declared and dimension tests present",
            measurement_fn="units text present; tests/physics/test_dimensional_homogeneity.py required for >60",
            threshold=">= 85 for final rank",
            failure_condition="range guards are mistaken for dimensional analysis",
            status="BLOCKED_NO_DIMENSIONAL_TEST",
            evidence=("docs/physics/geosync_core.md",),
        ),
        Metric(
            id="S_numerical_stability",
            weight=0.12,
            score=55.0 if t1_code and t1_tests else 0.0,
            metric="trajectory implementation plus convergence/stability tests",
            measurement_fn="T1 trajectory/test presence; dedicated convergence tests required for >70",
            threshold=">= 85 for final rank",
            failure_condition="solver fabricates physics or dt refinement is not measured",
            status="PARTIAL_PASS_NO_CONVERGENCE_GATE",
            evidence=("core/kuramoto/kuramoto_ricci_engine.py", "tests/unit/physics/test_T1_kuramoto_ricci_boundary.py"),
        ),
        Metric(
            id="S_invariant_preservation",
            weight=0.12,
            score=70.0 if t1_tests and has_boundary_tests else 0.0,
            metric="R bounds, boundary behavior, homogeneous potential, determinism",
            measurement_fn="static test carrier detection; pytest execution required for final",
            threshold=">= 90 for final rank",
            failure_condition="R outside [0,1], invalid Phi/Kc, non-finite theta, silent repair",
            status="IMPLEMENTED_NOT_RERUN",
            evidence=("tests/unit/physics/test_T1_kuramoto_ricci_boundary.py",),
        ),
        Metric(
            id="S_falsifiability",
            weight=0.14,
            score=65.0 if has_nulls and claim_ledger else 0.0,
            metric="nulls, H0/H1, negative controls, failure messages",
            measurement_fn="static negative-control/null carrier detection",
            threshold=">= 90 for final rank",
            failure_condition="positive signal survives K=0/shuffle/randomized nulls",
            status="PARTIAL_PASS_DEDICATED_NULL_MATRIX_MISSING",
            evidence=("docs/audit/claim_ledger.md", "tests/unit/physics/test_T1_kuramoto_ricci_boundary.py"),
        ),
        Metric(
            id="S_baseline_models",
            weight=0.10,
            score=45.0 if has_nulls else 0.0,
            metric="explicit null model factory coverage",
            measurement_fn="presence of K=0 and non-vacuity negative controls; dedicated factory missing",
            threshold=">= 85 for final rank",
            failure_condition="null model suite does not cover shuffled phases, ER matched graph, randomized omega",
            status="PARTIAL_PASS_FACTORY_MISSING",
            evidence=("tests/unit/physics/test_T1_kuramoto_ricci_boundary.py",),
        ),
        Metric(
            id="S_UQ",
            weight=0.08,
            score=0.0,
            metric="uncertainty source and UQ smoke artifact",
            measurement_fn="docs/physics/uq_plan.md and artifacts/physics_validation/uq_summary.json",
            threshold=">= 70 for final rank",
            failure_condition="confidence interval exists without prior/IC/noise/tolerance model",
            status="FAIL_NO_STOCHASTIC_SOURCE",
            evidence=(),
        ),
        Metric(
            id="S_reproducibility",
            weight=0.10,
            score=25.0 if exists("pyproject.toml") else 0.0,
            metric="manifest, command, seed, environment, generated artifacts",
            measurement_fn="reproducibility/manifest.json plus generated score artifact required for >70",
            threshold=">= 85 for final rank",
            failure_condition="artifact cannot be rerun from commit and command",
            status="BLOCKED_NO_MANIFEST",
            evidence=("pyproject.toml",),
        ),
        Metric(
            id="S_interface_contracts",
            weight=0.10,
            score=20.0,
            metric="shape/unit/range/sampling contracts across subsystem boundaries",
            measurement_fn="tests/physics/test_interface_contracts.py required",
            threshold=">= 80 for final rank",
            failure_condition="consumer accepts hidden or untyped conversion",
            status="FAIL_NO_INTERFACE_TESTS",
            evidence=(),
        ),
        Metric(
            id="S_traceability",
            weight=0.12,
            score=60.0 if claim_ledger and file_inventory else 0.0,
            metric="claim -> equation -> code -> test -> failure condition mapping",
            measurement_fn="claim ledger and inventory presence; equation traceability required for >80",
            threshold=">= 85 for final rank",
            failure_condition="claim lacks a carrier or source/code/test disagree",
            status="PARTIAL_PASS_TRACEABILITY_INDEX_MISSING",
            evidence=("docs/audit/claim_ledger.md", "docs/audit/file_inventory.md"),
        ),
    ]
    return metrics


def build_score() -> dict[str, Any]:
    metrics = build_metrics()
    weights_sum = round(sum(m.weight for m in metrics), 6)
    s_total = round(sum(m.weighted_score for m in metrics), 2)
    blockers = [
        {
            "id": "BLOCK-LOCAL-001",
            "severity": "HIGH",
            "reason": "full local repo scan and pytest run not observed by this connector pass",
            "required_action": "run python tools/physics_score.py --write and pytest tests/unit/physics/test_T1_kuramoto_ricci_boundary.py",
        },
        {
            "id": "BLOCK-DIM-001",
            "severity": "HIGH",
            "reason": "no dedicated dimensional homogeneity test exists",
            "required_action": "create tests/physics/test_dimensional_homogeneity.py and docs/physics/units_table.md",
        },
        {
            "id": "BLOCK-CONV-001",
            "severity": "HIGH",
            "reason": "no dt/dt2/dt4 convergence gate exists",
            "required_action": "create tests/physics/test_convergence.py with tolerance and expected order",
        },
        {
            "id": "BLOCK-UQ-001",
            "severity": "HIGH",
            "reason": "no stochastic source or UQ plan exists",
            "required_action": "define prior/IC/noise/tolerance uncertainty before Monte Carlo",
        },
        {
            "id": "BLOCK-IFACE-001",
            "severity": "MEDIUM",
            "reason": "vertical interface contracts are not tested",
            "required_action": "create docs/physics/interface_contracts.md and tests/physics/test_interface_contracts.py",
        },
        {
            "id": "BLOCK-BNSYN-MFN-001",
            "severity": "HIGH",
            "reason": "BN-Syn and MFN+ source trees are not present in this GeoSync validation branch",
            "required_action": "attach immutable repository refs before validating those subsystems",
        },
    ]
    verdict = "FAIL_BELOW_TARGET_BLOCKED_FOR_VALIDATION" if s_total < TARGET_INTERVAL[0] else "PASS_TARGET_INTERVAL"
    return {
        "schema_version": "2026.06.18",
        "status": "BOOTSTRAP_STATIC_BASELINE",
        "score_type": "static_evidence_oracle_not_pytest_result",
        "commit": git_commit(),
        "branch": git_branch(),
        "target_interval": TARGET_INTERVAL,
        "S_total": s_total,
        "weights_sum": weights_sum,
        "metrics": [m.as_dict() for m in metrics],
        "blockers": blockers,
        "verdict": verdict,
    }


def write_outputs(score: dict[str, Any]) -> None:
    out_dir = ROOT / "artifacts" / "physics_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_score.json").write_text(json.dumps(score, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# GeoSync Physics Validation Verdict",
        "",
        "Status: GENERATED_BY_tools/physics_score.py",
        f"Score type: `{score['score_type']}`",
        f"S_total: `{score['S_total']}`",
        f"Target interval: `{score['target_interval'][0]}-{score['target_interval'][1]}`",
        f"Verdict: `{score['verdict']}`",
        "",
        "## Blockers",
        "",
    ]
    for blocker in score["blockers"]:
        lines.append(f"- `{blocker['id']}` / {blocker['severity']}: {blocker['reason']} Required action: {blocker['required_action']}")
    lines.extend([
        "",
        "## Metric table",
        "",
        "| metric | weight | score | weighted | status |",
        "|---|---:|---:|---:|---|",
    ])
    for metric in score["metrics"]:
        lines.append(
            f"| `{metric['id']}` | {metric['weight']} | {metric['score']} | {metric['weighted_score']} | {metric['status']} |"
        )
    lines.extend([
        "",
        "## Final decision",
        "",
        "Physical rank claim is forbidden until the target interval is reached by a computed score and all HIGH blockers are closed.",
    ])
    (ROOT / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute GeoSync bootstrap physics validation score.")
    parser.add_argument("--write", action="store_true", help="write baseline_score.json and VERDICT.md")
    args = parser.parse_args()
    score = build_score()
    if args.write:
        write_outputs(score)
    print(json.dumps(score, indent=2, sort_keys=True))
    return 0 if score["weights_sum"] == 1.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
