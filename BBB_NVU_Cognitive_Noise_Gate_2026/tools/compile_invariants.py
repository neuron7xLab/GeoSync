"""Compile human-readable invariants into hash-bound executable metadata."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
INVARIANTS = ROOT / "invariants.yaml"

TEST_BINDINGS = {
    "INV_DETERMINISTIC_RUN_HASH": ("R003", ["tests/test_invariants.py::test_inv_deterministic_hash_stability"]),
    "INV_FAIL_CLOSED_INVALID": ("R002", ["tests/test_invariants.py::test_inv_critical_invalid_zeroes_confidence_and_blocks_execution"]),
    "INV_DEGRADATION_EXPLICIT": ("R004", ["tests/test_deterministic_engine.py::test_missing_domain_is_explicit_degradation"]),
    "INV_HUMAN_REVIEW_HIGH_RISK": ("R005", ["tests/test_deterministic_engine.py::test_high_risk_and_invalid_states_require_human_review"]),
    "INV_NO_AUTONOMOUS_CRITICAL_ACTION": ("R002", ["tests/test_invariants.py::test_inv_critical_invalid_zeroes_confidence_and_blocks_execution"]),
}


def statement_hash(statement: str, enforcement: str) -> str:
    """Hash semantic invariant text and enforcement target."""
    payload = f"{statement}\n{enforcement}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compile_invariants() -> None:
    """Update invariant metadata with requirement/test bindings and statement hashes."""
    doc = yaml.safe_load(INVARIANTS.read_text(encoding="utf-8"))
    for invariant in doc["invariants"]:
        requirement_id, test_refs = TEST_BINDINGS[invariant["id"]]
        invariant["requirement_id"] = requirement_id
        invariant["test_refs"] = test_refs
        invariant["statement_hash"] = statement_hash(invariant["statement"], invariant["enforcement"])
    INVARIANTS.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")


if __name__ == "__main__":
    compile_invariants()
