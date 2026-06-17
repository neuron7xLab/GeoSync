# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Tests for deterministic claim-state quantization."""

from __future__ import annotations

from tools.governance.quantize_claim_state import ClaimState, quantize_claim_state, validate_declared_state


def _envelope() -> dict[str, object]:
    return {
        "claim_id": "CLAIM-001",
        "claim_text": "The checked governance claim is locally verifiable.",
        "source_refs": ["tools/governance/quantize_claim_state.py"],
        "test_refs": ["tests/governance/test_claim_state_quantum_input.py"],
        "commands": [{"cmd": "pytest -q tests/governance/test_claim_state_quantum_input.py", "output_ref": "local"}],
        "artifacts": ["claim_state_report.json"],
        "ci_proof": {"same_sha": False, "required_checks_green": False},
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


def test_missing_link_is_partial() -> None:
    envelope = _envelope()
    envelope.pop("rollback")
    assert quantize_claim_state(envelope) is ClaimState.PARTIAL


def test_complete_without_ci_is_local_verified() -> None:
    assert quantize_claim_state(_envelope()) is ClaimState.LOCAL_VERIFIED


def test_same_sha_green_ci_is_ci_verified() -> None:
    envelope = _envelope()
    envelope["ci_proof"] = {"same_sha": True, "required_checks_green": True}
    assert quantize_claim_state(envelope) is ClaimState.CI_VERIFIED


def test_full_scientific_vector_is_evidence_bearing() -> None:
    envelope = _envelope()
    envelope["scientific_evidence"] = {
        "real_data": True,
        "replay": True,
        "baseline": True,
        "falsifier": True,
        "semantic_validation": True,
    }
    assert quantize_claim_state(envelope) is ClaimState.EVIDENCE_BEARING


def test_declared_state_cannot_round_upward() -> None:
    envelope = _envelope()
    envelope["declared_state"] = "CI_VERIFIED"
    result = validate_declared_state(envelope)
    assert result["state"] == "LOCAL_VERIFIED"
    assert result["upward_drift"] is True
