#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate the persisted NFI physics CI evidence summary.

The evidence summary is not a release certificate. It records which CI gates
actually completed for the NFI physics lane and keeps the physics score
fail-closed while full PR-gate evidence is incomplete.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / "artifacts" / "physics_validation" / "ci_evidence_summary.json"

REQUIRED_SUCCESS_WORKFLOWS = {
    "Physics Invariants",
    "Physics Kernel Gate",
    "Physics Reliability Gate",
    "Commit Acceptor Gate",
    "Repo Integrity Gate",
}


def load_evidence(path: Path = EVIDENCE_PATH) -> dict[str, Any]:
    """Load the CI evidence artifact as a JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def successful_workflows(evidence: dict[str, Any]) -> set[str]:
    """Return the names of workflows recorded as successful."""
    workflows = evidence.get("workflow_runs", [])
    return {
        str(item.get("name", ""))
        for item in workflows
        if item.get("status") == "completed" and item.get("conclusion") == "success"
    }


def validate_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Compute a fail-closed validation result for the CI evidence artifact."""
    successes = successful_workflows(evidence)
    missing = sorted(REQUIRED_SUCCESS_WORKFLOWS - successes)
    release = evidence.get("release_gate", {})
    pr_gate_success = bool(release.get("pr_gate_success"))
    all_required_physics_gates_green = not missing
    status = "PHYSICS_CI_PARTIAL_PASS"
    blockers: list[str] = []
    if missing:
        status = "PHYSICS_CI_INCOMPLETE"
        blockers.append("missing_required_physics_workflows")
    if not pr_gate_success:
        blockers.append("full_pr_gate_not_green")
    return {
        "status": status,
        "required_success_workflows": sorted(REQUIRED_SUCCESS_WORKFLOWS),
        "successful_required_workflows": sorted(REQUIRED_SUCCESS_WORKFLOWS & successes),
        "missing_required_workflows": missing,
        "all_required_physics_gates_green": all_required_physics_gates_green,
        "full_pr_gate_green": pr_gate_success,
        "blockers": blockers,
    }


def main() -> int:
    evidence = load_evidence()
    result = validate_evidence(evidence)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_required_physics_gates_green"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
