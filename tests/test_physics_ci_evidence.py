# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "physics_ci_evidence.py"


def load_module():
    spec = importlib.util.spec_from_file_location("physics_ci_evidence", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_physics_ci_evidence_records_required_green_gates() -> None:
    module = load_module()
    evidence = module.load_evidence()
    result = module.validate_evidence(evidence)
    assert result["all_required_physics_gates_green"] is True
    assert result["missing_required_workflows"] == []
    assert "Physics Invariants" in result["successful_required_workflows"]
    assert "Physics Kernel Gate" in result["successful_required_workflows"]
    assert "Physics Reliability Gate" in result["successful_required_workflows"]


def test_physics_ci_evidence_does_not_promote_release_readiness() -> None:
    module = load_module()
    evidence = module.load_evidence()
    result = module.validate_evidence(evidence)
    assert result["full_pr_gate_green"] is False
    assert "full_pr_gate_not_green" in result["blockers"]
    assert evidence["score_implication"]["may_claim_final_physics_validation"] is False
    assert evidence["score_implication"]["may_claim_target_interval"] is False
