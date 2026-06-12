# SPDX-License-Identifier: MIT
"""Contract tests for the MFN seven-role evidence cycle artifact."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
CYCLE = ROOT / "artifacts" / "mfn_evidence" / "MULTI_AGENT_CYCLE.json"
SCHEMA = ROOT / "schemas" / "mfn_multi_agent_cycle.schema.json"
BUNDLE = ROOT / "artifacts" / "mfn_evidence" / "EVIDENCE_BUNDLE.json"


def test_mfn_multi_agent_cycle_validates_against_schema() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    payload = json.loads(CYCLE.read_text(encoding="utf-8"))

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)


def test_mfn_multi_agent_cycle_preserves_blocked_evidence_boundary() -> None:
    payload = json.loads(CYCLE.read_text(encoding="utf-8"))
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))

    assert payload["final_gate"]["status"] == "BLOCKED"
    assert payload["evidence_status"]["score"] is None
    assert payload["evidence_status"]["confidence"] <= 0.9
    assert payload["agent_outputs"]["falsifier"]["status"] == "BROKEN"
    assert bundle["claim_tier_change"] == "none"
