# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Deterministic claim-state quantizer.

This module turns a raw evidence envelope into exactly one discrete governance
state. It does not validate physics, markets, cognition, or predictive power.
It only prevents claim promotion when required evidence links are missing.
"""

from __future__ import annotations

import argparse
import json
from enum import Enum
from pathlib import Path
from typing import Any


class ClaimState(str, Enum):
    FALSE = "FALSE"
    UNTESTED = "UNTESTED"
    PARTIAL = "PARTIAL"
    LOCAL_VERIFIED = "LOCAL_VERIFIED"
    CI_VERIFIED = "CI_VERIFIED"
    EVIDENCE_BEARING = "EVIDENCE_BEARING"


REQUIRED_KEYS = {
    "claim_id",
    "claim_text",
    "source_refs",
    "test_refs",
    "commands",
    "artifacts",
    "ci_proof",
    "failure_mode",
    "rollback",
    "claim_boundary",
    "negative_evidence",
}


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def missing_required_links(envelope: dict[str, Any]) -> list[str]:
    """Return missing proof-chain links in stable order."""

    missing: list[str] = []
    for key in sorted(REQUIRED_KEYS):
        if key not in envelope or not _nonempty(envelope[key]):
            missing.append(key)
    return missing


def quantize_claim_state(envelope: dict[str, Any]) -> ClaimState:
    """Map a claim-evidence envelope to one allowed discrete state."""

    if envelope.get("contradicted") is True:
        return ClaimState.FALSE

    missing = missing_required_links(envelope)
    if len(missing) == len(REQUIRED_KEYS):
        return ClaimState.UNTESTED
    if missing:
        return ClaimState.PARTIAL

    ci = envelope.get("ci_proof", {})
    scientific = envelope.get("scientific_evidence", {})

    if (
        scientific.get("real_data") is True
        and scientific.get("replay") is True
        and scientific.get("baseline") is True
        and scientific.get("falsifier") is True
        and scientific.get("semantic_validation") is True
    ):
        return ClaimState.EVIDENCE_BEARING

    if ci.get("same_sha") is True and ci.get("required_checks_green") is True:
        return ClaimState.CI_VERIFIED

    return ClaimState.LOCAL_VERIFIED


def validate_declared_state(envelope: dict[str, Any]) -> dict[str, Any]:
    """Return quantization result and reject upward state drift."""

    actual = quantize_claim_state(envelope)
    declared = envelope.get("declared_state")
    allowed_order = list(ClaimState)
    upward_drift = False
    if declared:
        try:
            upward_drift = allowed_order.index(ClaimState(declared)) > allowed_order.index(actual)
        except ValueError:
            upward_drift = True
    return {
        "claim_id": envelope.get("claim_id", "UNKNOWN"),
        "state": actual.value,
        "declared_state": declared,
        "upward_drift": upward_drift,
        "missing_links": missing_required_links(envelope),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to claim evidence JSON envelope")
    args = parser.parse_args()
    envelope = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = validate_declared_state(envelope)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["upward_drift"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
