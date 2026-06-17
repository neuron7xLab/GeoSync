# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Tests for deterministic claim-state quantization."""

from __future__ import annotations

import json
from pathlib import Path

from tools.governance.quantize_claim_state import ClaimState, quantize_claim_state, validate_declared_state


SCHEMA_PATH = Path("schemas/governance/claim_state_quantum_input.schema.json")


def _envelope() -> dict[str, object]:
    return {
        "claim_id": "CLAIM-001",
        "claim_text": "The checked governance claim is locally verifiable.",
        "source_refs": ["tools/governance/quantize_claim_state.py"],
        "test_refs": ["tests/governance/test_claim_state_quantum_input.py"],
        "commands": [
            {
                "cmd": "pytest -q tests/governance/test_claim_state_quantum_input.py",
                "output_ref": "local",
                "exit_code": 0,
            }
        ],
        "artifacts": ["claim_state_report.json"],
        "ci_proof": {
            "sha": "181f97160a407c8e4c7262d351c1f0d6658bbe87",
            "same_sha": False,
            "required_checks_green": False,
        },
        "failure_mode": "Missing links must prevent promotion.",
        "rollback": "Revert this governance contract commit.",
        "claim_boundary": {"allowed": ["governance quantization"], "blocked": ["domain validation"]},
        "negative_evidence": ["No CI proof is attached in this fixture."],
    }


def test_empty_envelope_is_untested() -> None:
    assert quantize_claim_state({}) is ClaimState.UNTESTED


def test_contradicted_envelope_is_false() -> None:
    envelope = _envelope()
    envelope["contradicted"] = True
    assert quantize_claim_state(envelope) is ClaimState.FALSE


def test_missing_links_are_partial() -> None:
    envelope = _envelope()
    envelope.pop("rollback")
    assert quantize_claim_state(envelope) is ClaimState.PARTIAL


def test_complete_without_ci_is_local_verified() -> None:
    assert quantize_claim_state(_envelope()) is ClaimState.LOCAL_VERIFIED


def test_same_sha_green_ci_is_ci_verified() -> None:
    envelope = _envelope()
    envelope["ci_proof"] = {
        "sha": "181f97160a407c8e4c7262d351c1f0d6658bbe87",
        "same_sha": True,
        "required_checks_green": True,
    }
    assert quantize_claim_state(envelope) is ClaimState.CI_VERIFIED


def test_scientific_vector_without_same_sha_ci_is_not_evidence_bearing() -> None:
    envelope = _envelope()
    envelope["scientific_evidence"] = {
        "real_data": True,
        "replay": True,
        "baseline": True,
        "falsifier": True,
        "semantic_validation": True,
    }
    assert quantize_claim_state(envelope) is ClaimState.LOCAL_VERIFIED


def test_real_data_replay_baseline_falsifier_semantic_validation_is_evidence_bearing() -> None:
    envelope = _envelope()
    envelope["ci_proof"] = {
        "sha": "181f97160a407c8e4c7262d351c1f0d6658bbe87",
        "same_sha": True,
        "required_checks_green": True,
    }
    envelope["scientific_evidence"] = {
        "real_data": True,
        "replay": True,
        "baseline": True,
        "falsifier": True,
        "semantic_validation": True,
    }
    assert quantize_claim_state(envelope) is ClaimState.EVIDENCE_BEARING


def test_declared_state_cannot_round_up() -> None:
    envelope = _envelope()
    envelope["declared_state"] = "CI_VERIFIED"
    result = validate_declared_state(envelope)
    assert result["state"] == "LOCAL_VERIFIED"
    assert result["upward_drift"] is True


def test_schema_is_fail_closed_for_claim_envelope_shape() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert schema["additionalProperties"] is False
    assert schema["properties"]["commands"]["minItems"] == 1
    assert schema["properties"]["ci_proof"]["additionalProperties"] is False
    assert schema["properties"]["scientific_evidence"]["additionalProperties"] is False
    assert schema["properties"]["claim_boundary"]["additionalProperties"] is False
