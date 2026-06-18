#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Deterministic bootstrap physics scoring oracle.

This is not a rank generator. It is a fail-closed repository evidence scanner.
It scores only visible carriers: files, tests, artifacts, and blockers. Runtime
pytest or CI evidence is still required before any physical-rank claim.
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
    candidate = ROOT / path
    if not candidate.exists() or not candidate.is_file():
        return ""
    return candidate.read_text(encoding="utf-8", errors="replace")


def contains(path: str, *needles: str) -> bool:
    body = text(path)
    return all(needle in body for needle in needles)


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def build_metrics() -> list[Metric]:
    geosync_core = exists("docs/physics/geosync_core.md")
    t1_code = exists("core/kuramoto/kuramoto_ricci_engine.py")
    t1_tests = exists("tests/unit/physics/test_T1_kuramoto_ricci_boundary.py")
    claim_ledger = exists("docs/audit/claim_ledger.md")
    file_inventory = exists("docs/audit/file_inventory.md")
    units_table = exists("docs/physics/units_table.md")
    dimensional_test = exists("tests/physics/test_dimensional_homogeneity.py")
    convergence_test = exists("tests/physics/test_convergence.py")
    invariant_test = exists("tests/physics/test_invariants.py")
    null_test = exists("tests/physics/test_geosync_nulls.py")
    uq_plan = exists("docs/physics/uq_plan.md")
    uq_test = exists("tests/physics/test_uq_smoke.py")
    uq_artifact = exists("artifacts/physics_validation/uq_summary.json")
    replication_test = exists("tests/physics/test_replication.py")
    replication_artifact = exists("artifacts/physics_validation/replication_summary.json")
    reference_solver = exists("reproducibility/reference_solvers/geosync_kuramoto_numpy.py")
    ricci_test = exists("tests/physics/test_ricci_bridge.py")
    ricci_artifact = exists("artifacts/physics_validation/ricci_bridge_summary.json")
    interface_doc = exists("docs/physics/interface_contracts.md")
    interface_test = exists("tests/physics/test_interface_contracts.py")
    manifest = exists("reproducibility/manifest.json")
    traceability = exists("docs/physics/equation_traceability.md")

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

    return [
        Metric(
            "S_math_object",
            0.12,
            75.0 if geosync_core and t1_code and has_rhs and traceability else 0.0,
            "canonical mathematical object has equation/code carriers",
            "exists(docs/physics/geosync_core.md) and contains(T1 code symbols)",
            ">= 90 for final rank",
            "no single canonical object or missing RHS/boundary/order parameter code",
            "PARTIAL_PASS_TRACEABLE" if geosync_core and t1_code and has_rhs else "FAIL",
            ("docs/physics/geosync_core.md", "core/kuramoto/kuramoto_ricci_engine.py"),
        ),
        Metric(
            "S_dimensional_consistency",
            0.10,
            70.0 if units_table and dimensional_test else 0.0,
            "units declared and dimension tests present",
            "docs/physics/units_table.md and tests/physics/test_dimensional_homogeneity.py",
            ">= 85 for final rank",
            "range guards are mistaken for dimensional analysis",
            "PARTIAL_PASS_NO_TYPED_UNITS" if units_table and dimensional_test else "FAIL",
            ("docs/physics/units_table.md", "tests/physics/test_dimensional_homogeneity.py"),
        ),
        Metric(
            "S_numerical_stability",
            0.12,
            82.0 if t1_code and t1_tests and convergence_test and replication_test else 0.0,
            "trajectory implementation plus convergence and replication gates",
            "T1 trajectory, dt refinement, and independent NumPy reference test",
            ">= 85 for final rank",
            "solver fabricates physics or dt refinement is not measured",
            "PARTIAL_PASS_RUNTIME_REQUIRED" if convergence_test and replication_test else "FAIL",
            ("tests/physics/test_convergence.py", "tests/physics/test_replication.py"),
        ),
        Metric(
            "S_invariant_preservation",
            0.12,
            80.0 if t1_tests and has_boundary_tests and invariant_test else 0.0,
            "R bounds, boundary behavior, homogeneous potential, determinism",
            "static test carrier detection; pytest execution required for final",
            ">= 90 for final rank",
            "R outside [0,1], invalid Phi/Kc, non-finite theta, silent repair",
            "PARTIAL_PASS_RUNTIME_REQUIRED",
            (
                "tests/unit/physics/test_T1_kuramoto_ricci_boundary.py",
                "tests/physics/test_invariants.py",
            ),
        ),
        Metric(
            "S_falsifiability",
            0.14,
            82.0 if has_nulls and null_test and claim_ledger and ricci_test else 0.0,
            "nulls, H0/H1, negative controls, Ricci bridge smoke tests",
            "T1 negative controls plus dedicated null and Ricci bridge matrix",
            ">= 90 for final rank",
            "positive signal survives K=0/shuffle/randomized nulls",
            "PARTIAL_PASS_RUNTIME_REQUIRED" if null_test and ricci_test else "FAIL",
            ("tests/physics/test_geosync_nulls.py", "tests/physics/test_ricci_bridge.py"),
        ),
        Metric(
            "S_baseline_models",
            0.10,
            82.0 if null_test and ricci_artifact else 0.0,
            "explicit null model factory coverage",
            "K=0, phase shuffle, randomized omega, matched ER, Ricci bridge smoke",
            ">= 85 for final rank",
            "null model suite does not cover shuffled phases, ER graph, omega randomization",
            "PARTIAL_PASS_RUNTIME_REQUIRED" if null_test else "FAIL",
            (
                "tests/physics/test_geosync_nulls.py",
                "artifacts/physics_validation/ricci_bridge_summary.json",
            ),
        ),
        Metric(
            "S_UQ",
            0.08,
            72.0 if uq_plan and uq_test and uq_artifact else 0.0,
            "uncertainty source, UQ smoke test, and artifact",
            "docs/physics/uq_plan.md, tests/physics/test_uq_smoke.py, uq_summary.json",
            ">= 70 for final rank",
            "confidence interval exists without prior/IC/noise/tolerance model",
            "PARTIAL_PASS_RUNTIME_REQUIRED" if uq_test and uq_artifact else "SOURCE_DECLARED_ONLY",
            ("docs/physics/uq_plan.md", "tests/physics/test_uq_smoke.py"),
        ),
        Metric(
            "S_reproducibility",
            0.08,
            78.0 if manifest and reference_solver and replication_artifact else 0.0,
            "manifest, command, seed, reference solver, generated artifacts",
            "reproducibility manifest plus independent reference solver artifact",
            ">= 85 for final rank",
            "artifact cannot be rerun from commit and command",
            "PARTIAL_PASS_CI_LOGS_REQUIRED" if manifest and reference_solver else "FAIL",
            (
                "reproducibility/manifest.json",
                "reproducibility/reference_solvers/geosync_kuramoto_numpy.py",
            ),
        ),
        Metric(
            "S_interface_contracts",
            0.08,
            70.0 if interface_doc and interface_test else 0.0,
            "shape/unit/range/sampling contracts across subsystem boundaries",
            "docs/physics/interface_contracts.md and tests/physics/test_interface_contracts.py",
            ">= 80 for final rank",
            "consumer accepts hidden or untyped conversion",
            "PARTIAL_PASS_RUNTIME_REQUIRED" if interface_doc and interface_test else "FAIL",
            ("docs/physics/interface_contracts.md", "tests/physics/test_interface_contracts.py"),
        ),
        Metric(
            "S_traceability",
            0.06,
            85.0 if claim_ledger and file_inventory and traceability else 0.0,
            "claim -> equation -> code -> test -> failure condition mapping",
            "claim ledger, inventory, equation traceability index",
            ">= 85 for final rank",
            "claim lacks a carrier or source/code/test disagree",
            "PARTIAL_PASS_RUNTIME_REPORT_PENDING" if traceability else "FAIL",
            ("docs/audit/claim_ledger.md", "docs/physics/equation_traceability.md"),
        ),
    ]


def build_score() -> dict[str, Any]:
    metrics = build_metrics()
    weights_sum = round(sum(metric.weight for metric in metrics), 6)
    s_total = round(sum(metric.weighted_score for metric in metrics), 2)
    blockers = [
        {
            "id": "BLOCK-CI-001",
            "severity": "HIGH",
            "reason": "new physics tests and artifacts require CI or local pytest proof",
            "required_action": "run the physics validation commands and persist logs",
        },
        {
            "id": "BLOCK-RICCI-001",
            "severity": "HIGH",
            "reason": "Ricci bridge has smoke tests but no full market graph provenance audit",
            "required_action": "validate graph construction, curvature estimator, and matched nulls",
        },
        {
            "id": "BLOCK-BNSYN-MFN-001",
            "severity": "HIGH",
            "reason": "BN-Syn and MFN+ source trees are not present in this GeoSync branch",
            "required_action": "attach immutable source refs before validating those subsystems",
        },
    ]
    verdict = "FAIL_BELOW_TARGET_BLOCKED_FOR_VALIDATION"
    if s_total >= TARGET_INTERVAL[0] and not blockers:
        verdict = "PASS_TARGET_INTERVAL"
    return {
        "schema_version": "2026.06.18",
        "status": "BOOTSTRAP_STATIC_PLUS_SMOKE_ARTIFACTS",
        "score_type": "static_evidence_oracle_not_runtime_proof",
        "commit": git_value("rev-parse", "HEAD"),
        "branch": git_value("branch", "--show-current"),
        "target_interval": TARGET_INTERVAL,
        "S_total": s_total,
        "weights_sum": weights_sum,
        "metrics": [metric.as_dict() for metric in metrics],
        "blockers": blockers,
        "verdict": verdict,
    }


def write_outputs(score: dict[str, Any]) -> None:
    out_dir = ROOT / "artifacts" / "physics_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_score.json").write_text(
        json.dumps(score, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

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
        lines.append(
            f"- `{blocker['id']}` / {blocker['severity']}: "
            f"{blocker['reason']} Required action: {blocker['required_action']}"
        )
    lines.extend(
        [
            "",
            "## Metric table",
            "",
            "| metric | weight | score | weighted | status |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for metric in score["metrics"]:
        lines.append(
            f"| `{metric['id']}` | {metric['weight']} | {metric['score']} | "
            f"{metric['weighted_score']} | {metric['status']} |"
        )
    lines.extend(
        [
            "",
            "## Final decision",
            "",
            "Physical rank claim is forbidden until the target interval is reached "
            "by a computed score and all HIGH blockers are closed.",
        ]
    )
    (ROOT / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute GeoSync bootstrap physics validation score."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write baseline_score.json and VERDICT.md",
    )
    args = parser.parse_args()
    score = build_score()
    if args.write:
        write_outputs(score)
    print(json.dumps(score, indent=2, sort_keys=True))
    return 0 if score["weights_sum"] == 1.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
