from __future__ import annotations

import json
from pathlib import Path

import pytest
import shutil

from tools.claims.validate_claims_have_evidence import validate


import shutil
import pytest


@pytest.fixture(autouse=True)
def _clean_test_baseline_dirs() -> None:
    for name in (
        "baseline_run_test_mismatch",
        "baseline_run_test_valid",
        "baseline_run_test_failed",
    ):
        d = Path("reports/test_audit") / name
        if d.exists():
            shutil.rmtree(d)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _base_claim(**kwargs):
    data = {
        "claim_id": "C1",
        "claim_text": "sample claim",
        "domain": "testing",
        "claim_status": "proven",
        "command": "pytest -q",
        "artifact_path": "",
        "verdict": "accepted",
        "release_decision": "allow",
        "last_verified_sha": "abc",
        "timestamp_utc": "2026-05-22T00:00:00Z",
    }
    data.update(kwargs)
    return data


def test_valid_proven_claim_with_real_artifact_passes(tmp_path: Path) -> None:
    art = tmp_path / "ok.txt"
    art.write_text("ok")
    p = tmp_path / "evidence.json"
    _write_json(p, {"schema_version": "1.1", "claims": [_base_claim(artifact_path=str(art))]})
    assert validate(p) == []


def test_proven_claim_without_artifact_path_fails(tmp_path: Path) -> None:
    p = tmp_path / "evidence.json"
    _write_json(p, {"schema_version": "1.1", "claims": [_base_claim(artifact_path="")]})
    assert validate(p)


def test_proven_claim_with_missing_artifact_file_fails(tmp_path: Path) -> None:
    p = tmp_path / "evidence.json"
    _write_json(
        p,
        {
            "schema_version": "1.1",
            "claims": [_base_claim(artifact_path=str(tmp_path / "missing.txt"))],
        },
    )
    assert validate(p)


def test_proven_claim_without_sha_fails(tmp_path: Path) -> None:
    art = tmp_path / "ok.txt"
    art.write_text("ok")
    p = tmp_path / "evidence.json"
    _write_json(
        p,
        {
            "schema_version": "1.1",
            "claims": [_base_claim(artifact_path=str(art), last_verified_sha="")],
        },
    )
    assert validate(p)


def test_proven_claim_without_timestamp_fails(tmp_path: Path) -> None:
    art = tmp_path / "ok.txt"
    art.write_text("ok")
    p = tmp_path / "evidence.json"
    _write_json(
        p,
        {
            "schema_version": "1.1",
            "claims": [_base_claim(artifact_path=str(art), timestamp_utc="")],
        },
    )
    assert validate(p)


def test_goal_claim_presented_as_achieved_fails(tmp_path: Path) -> None:
    p = tmp_path / "evidence.json"
    _write_json(
        p,
        {
            "schema_version": "1.1",
            "claims": [_base_claim(claim_status="goal", release_decision="accepted_as_proven")],
        },
    )
    assert validate(p)


def test_partial_claim_presented_as_proven_fails(tmp_path: Path) -> None:
    p = tmp_path / "evidence.json"
    _write_json(
        p,
        {
            "schema_version": "1.1",
            "claims": [_base_claim(claim_status="partial", release_decision="allow")],
        },
    )
    assert validate(p)


def test_quarantine_claim_used_as_release_evidence_fails(tmp_path: Path) -> None:
    p = tmp_path / "evidence.json"
    _write_json(
        p,
        {
            "schema_version": "1.1",
            "claims": [_base_claim(claim_status="quarantine", release_decision="allow")],
        },
    )
    assert validate(p)


def test_remove_claim_used_as_accepted_evidence_fails(tmp_path: Path) -> None:
    p = tmp_path / "evidence.json"
    _write_json(
        p,
        {
            "schema_version": "1.1",
            "claims": [_base_claim(claim_status="remove", release_decision="allow")],
        },
    )
    assert validate(p)


def test_malformed_evidence_index_fails(tmp_path: Path) -> None:
    p = tmp_path / "evidence.json"
    _write_json(p, {"schema_version": "1.1", "claims": {}})
    assert validate(p)


def test_artifact_manifest_mismatch_fails(tmp_path: Path) -> None:
    run = Path("reports/test_audit/baseline_run_test_mismatch")
    run.mkdir(parents=True, exist_ok=True)
    (run / "a.txt").write_text("x")
    (run / "artifact_manifest.sha256").write_text("deadbeef  a.txt\n")
    p = tmp_path / "evidence.json"
    art = tmp_path / "ok.txt"
    art.write_text("ok")
    _write_json(p, {"schema_version": "1.1", "claims": [_base_claim(artifact_path=str(art))]})
    assert validate(p)


def test_artifact_manifest_valid_passes(tmp_path: Path) -> None:
    mismatch = Path("reports/test_audit/baseline_run_test_mismatch")
    if mismatch.exists():
        for x in mismatch.glob("*"):
            x.unlink()
        mismatch.rmdir()
    run = Path("reports/test_audit/baseline_run_test_valid")
    run.mkdir(parents=True, exist_ok=True)
    (run / "a.txt").write_text("x")
    import hashlib

    h = hashlib.sha256((run / "a.txt").read_bytes()).hexdigest()
    (run / "artifact_manifest.sha256").write_text(f"{h}  a.txt\n")
    p = tmp_path / "evidence.json"
    art = tmp_path / "ok.txt"
    art.write_text("ok")
    _write_json(p, {"schema_version": "1.1", "claims": [_base_claim(artifact_path=str(art))]})
    assert validate(p) == []


def test_failed_baseline_only_allowed_for_failure_detection(tmp_path: Path) -> None:
    run = Path("reports/test_audit/baseline_run_test_failed")
    run.mkdir(parents=True, exist_ok=True)
    (run / "exit_code.txt").write_text("1\n")
    art = run / "junit.xml"
    art.write_text("<testsuite/>")
    p = tmp_path / "evidence.json"
    _write_json(
        p,
        {
            "schema_version": "1.1",
            "claims": [_base_claim(artifact_path=str(art), claim_text="test suite passes")],
        },
    )
    assert validate(p)
