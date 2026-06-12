# SPDX-License-Identifier: MIT
"""Contract tests for the pre-release convergence task graph."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / "artifacts" / "governance" / "pre_release_convergence_task_graph.json"
SCHEMA = ROOT / "schemas" / "pre_release_convergence_task_graph.schema.json"


def test_pre_release_convergence_task_graph_validates_against_schema() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    payload = json.loads(GRAPH.read_text(encoding="utf-8"))

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)


def test_pre_release_convergence_task_graph_blocks_status_inflation() -> None:
    payload = json.loads(GRAPH.read_text(encoding="utf-8"))

    assert payload["definition_lock"]["claim_tier_change"] == "none"
    assert payload["evidence_status"]["score"] is None
    assert payload["evidence_status"]["confidence"] <= 0.65
    assert payload["evidence_status"]["status"] == "CANDIDATE_NOT_VALIDATED"
    assert payload["final_gate"]["status"] == "BLOCKED_BY_EVIDENCE"
    assert "production ready" in payload["definition_lock"]["forbidden_promotion"]
