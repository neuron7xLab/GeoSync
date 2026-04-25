# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Claim/evidence gate — schema and integrity tests.

Guards the canonical ``docs/CLAIMS.yaml`` registry against silent
drift:

* The registry parses cleanly under the v1 schema.
* Every gated (P0/P1) claim's evidence paths exist in the working tree.
* The validator script itself fail-closes on every documented violation
  (missing field, malformed id, unknown priority, empty description,
  empty evidence list, malformed date, missing path, schema-version
  mismatch, duplicate id).

Together these tests are the "self-falsification" proof that the
claim-evidence gate cannot accept a registry that lies about its
backing artefacts.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CLAIMS_PATH = ROOT / "docs" / "CLAIMS.yaml"
SCRIPT_PATH = ROOT / "scripts" / "ci" / "check_claims.py"


def _load_module() -> Any:
    """Import the script as a module without polluting sys.path."""
    spec = importlib.util.spec_from_file_location("check_claims", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_claims"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cc() -> Any:
    return _load_module()


# ---------------------------------------------------------------------------
# Live registry — must always pass on main
# ---------------------------------------------------------------------------


def test_live_registry_passes(cc: Any) -> None:
    """The on-main registry must satisfy every gate it declares — this
    is the inverse of an aspirational README: missing evidence on a
    P0/P1 claim breaks the gate, breaks CI, breaks main."""
    registry = cc._load_registry(CLAIMS_PATH)
    claims = cc._parse_claims(registry)
    for claim in claims:
        failures = cc._validate_evidence(claim, ROOT)
        assert failures == [], (
            f"claim {claim.id} ({claim.priority}) has missing evidence: "
            f"{[f.reason for f in failures]}"
        )


def test_live_registry_has_at_least_one_p0(cc: Any) -> None:
    """Sanity: a registry without a P0 claim is decorative, not a
    gate. We require at least one to keep the contract live."""
    registry = cc._load_registry(CLAIMS_PATH)
    claims = cc._parse_claims(registry)
    assert any(
        c.priority == "P0" for c in claims
    ), "no P0 claims registered — gate has nothing to enforce"


def test_live_registry_ids_are_unique(cc: Any) -> None:
    registry = cc._load_registry(CLAIMS_PATH)
    claims = cc._parse_claims(registry)
    ids = [c.id for c in claims]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Synthetic registries — every fail-closed branch
# ---------------------------------------------------------------------------


def _write_registry(tmp_path: Path, body: dict[str, Any]) -> Path:
    target = tmp_path / "CLAIMS.yaml"
    target.write_text(yaml.safe_dump(body), encoding="utf-8")
    return target


def _minimal_valid(extra_claims: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    base = [
        {
            "id": "test-claim",
            "priority": "P2",
            "description": "synthetic",
            "evidence_paths": ["docs/CLAIMS.yaml"],  # any tracked path
            "added_utc": "2026-04-25",
        },
    ]
    if extra_claims:
        base.extend(extra_claims)
    return {"schema_version": 1, "claims": base}


def test_rejects_wrong_schema_version(tmp_path: Path, cc: Any) -> None:
    body = _minimal_valid()
    body["schema_version"] = 99
    path = _write_registry(tmp_path, body)
    registry = cc._load_registry(path)
    with pytest.raises(ValueError, match="schema_version"):
        cc._parse_claims(registry)


def test_rejects_missing_claims_list(tmp_path: Path, cc: Any) -> None:
    body = {"schema_version": 1}
    path = _write_registry(tmp_path, body)
    registry = cc._load_registry(path)
    with pytest.raises(ValueError, match="claims"):
        cc._parse_claims(registry)


def test_rejects_missing_field(tmp_path: Path, cc: Any) -> None:
    body = _minimal_valid()
    body["claims"][0].pop("description")
    path = _write_registry(tmp_path, body)
    registry = cc._load_registry(path)
    with pytest.raises(ValueError, match="missing fields"):
        cc._parse_claims(registry)


def test_rejects_non_kebab_id(tmp_path: Path, cc: Any) -> None:
    body = _minimal_valid()
    body["claims"][0]["id"] = "Not-Kebab-Case"
    path = _write_registry(tmp_path, body)
    registry = cc._load_registry(path)
    with pytest.raises(ValueError, match="kebab-case"):
        cc._parse_claims(registry)


def test_rejects_unknown_priority(tmp_path: Path, cc: Any) -> None:
    body = _minimal_valid()
    body["claims"][0]["priority"] = "P9"
    path = _write_registry(tmp_path, body)
    registry = cc._load_registry(path)
    with pytest.raises(ValueError, match="priority"):
        cc._parse_claims(registry)


def test_rejects_empty_description(tmp_path: Path, cc: Any) -> None:
    body = _minimal_valid()
    body["claims"][0]["description"] = "   "
    path = _write_registry(tmp_path, body)
    registry = cc._load_registry(path)
    with pytest.raises(ValueError, match="description"):
        cc._parse_claims(registry)


def test_rejects_empty_evidence_list(tmp_path: Path, cc: Any) -> None:
    body = _minimal_valid()
    body["claims"][0]["evidence_paths"] = []
    path = _write_registry(tmp_path, body)
    registry = cc._load_registry(path)
    with pytest.raises(ValueError, match="evidence_paths"):
        cc._parse_claims(registry)


def test_rejects_non_string_evidence(tmp_path: Path, cc: Any) -> None:
    body = _minimal_valid()
    body["claims"][0]["evidence_paths"] = ["docs/CLAIMS.yaml", 123]
    path = _write_registry(tmp_path, body)
    registry = cc._load_registry(path)
    with pytest.raises(ValueError, match="evidence_paths"):
        cc._parse_claims(registry)


def test_rejects_malformed_date(tmp_path: Path, cc: Any) -> None:
    body = _minimal_valid()
    body["claims"][0]["added_utc"] = "yesterday"
    path = _write_registry(tmp_path, body)
    registry = cc._load_registry(path)
    with pytest.raises(ValueError, match="added_utc"):
        cc._parse_claims(registry)


def test_rejects_duplicate_id(tmp_path: Path, cc: Any) -> None:
    extra = {
        "id": "test-claim",
        "priority": "P2",
        "description": "duplicate",
        "evidence_paths": ["docs/CLAIMS.yaml"],
        "added_utc": "2026-04-25",
    }
    body = _minimal_valid(extra_claims=[extra])
    path = _write_registry(tmp_path, body)
    registry = cc._load_registry(path)
    with pytest.raises(ValueError, match="duplicate"):
        cc._parse_claims(registry)


def test_p2_claim_with_missing_evidence_does_not_fail_gate(
    tmp_path: Path,
    cc: Any,
) -> None:
    """P2 is informational — missing evidence should be reported by
    a future linter but must not break the build."""
    body = _minimal_valid()
    body["claims"][0]["evidence_paths"] = ["does/not/exist.txt"]
    path = _write_registry(tmp_path, body)
    registry = cc._load_registry(path)
    claims = cc._parse_claims(registry)
    failures = cc._validate_evidence(claims[0], ROOT)
    assert failures == []


def test_p1_claim_with_missing_evidence_fails_gate(
    tmp_path: Path,
    cc: Any,
) -> None:
    body = _minimal_valid()
    body["claims"][0]["priority"] = "P1"
    body["claims"][0]["evidence_paths"] = ["does/not/exist.txt"]
    path = _write_registry(tmp_path, body)
    registry = cc._load_registry(path)
    claims = cc._parse_claims(registry)
    failures = cc._validate_evidence(claims[0], ROOT)
    assert len(failures) == 1
    assert "does/not/exist.txt" in failures[0].reason


def test_p0_claim_with_missing_evidence_fails_gate(
    tmp_path: Path,
    cc: Any,
) -> None:
    body = _minimal_valid()
    body["claims"][0]["priority"] = "P0"
    body["claims"][0]["evidence_paths"] = [
        "docs/CLAIMS.yaml",
        "totally/missing/path.py",
    ]
    path = _write_registry(tmp_path, body)
    registry = cc._load_registry(path)
    claims = cc._parse_claims(registry)
    failures = cc._validate_evidence(claims[0], ROOT)
    assert len(failures) == 1
    assert "totally/missing/path.py" in failures[0].reason


# ---------------------------------------------------------------------------
# CLI exit codes
# ---------------------------------------------------------------------------


def test_main_exits_zero_on_passing_registry(cc: Any) -> None:
    """End-to-end: the live registry on this branch returns 0."""
    rc = cc.main([])
    assert rc == 0
