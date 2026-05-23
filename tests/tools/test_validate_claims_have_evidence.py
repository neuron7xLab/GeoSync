from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.claims.validate_claims_have_evidence import validate


@pytest.fixture
def audit_root(tmp_path: Path) -> Path:
    root = tmp_path / "reports" / "test_audit"
    root.mkdir(parents=True)
    return root


def _write_evidence(path: Path, claims: list[dict]) -> Path:
    payload = {"schema_version": "1.1", "claims": claims}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _base_claim(**kwargs: object) -> dict[str, object]:
    data: dict[str, object] = {
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


def test_valid_proven_claim_with_real_artifact_passes(tmp_path: Path, audit_root: Path) -> None:
    artifact = tmp_path / "proof.txt"
    artifact.write_text("ok", encoding="utf-8")
    evidence = _write_evidence(
        tmp_path / "evidence.json", [_base_claim(artifact_path=str(artifact))]
    )
    assert validate(evidence, audit_root=audit_root, manifest_scope="referenced") == []


def test_proven_claim_without_artifact_path_fails(tmp_path: Path, audit_root: Path) -> None:
    evidence = _write_evidence(tmp_path / "evidence.json", [_base_claim(artifact_path="")])
    assert validate(evidence, audit_root=audit_root, manifest_scope="referenced")


def test_proven_claim_with_missing_artifact_file_fails(tmp_path: Path, audit_root: Path) -> None:
    evidence = _write_evidence(
        tmp_path / "evidence.json",
        [_base_claim(artifact_path=str(tmp_path / "missing.txt"))],
    )
    assert validate(evidence, audit_root=audit_root, manifest_scope="referenced")


def test_proven_claim_without_sha_fails(tmp_path: Path, audit_root: Path) -> None:
    artifact = tmp_path / "proof.txt"
    artifact.write_text("ok", encoding="utf-8")
    evidence = _write_evidence(
        tmp_path / "evidence.json",
        [_base_claim(artifact_path=str(artifact), last_verified_sha="")],
    )
    assert validate(evidence, audit_root=audit_root, manifest_scope="referenced")


def test_proven_claim_without_timestamp_fails(tmp_path: Path, audit_root: Path) -> None:
    artifact = tmp_path / "proof.txt"
    artifact.write_text("ok", encoding="utf-8")
    evidence = _write_evidence(
        tmp_path / "evidence.json",
        [_base_claim(artifact_path=str(artifact), timestamp_utc="")],
    )
    assert validate(evidence, audit_root=audit_root, manifest_scope="referenced")


def test_goal_claim_presented_as_achieved_fails(tmp_path: Path, audit_root: Path) -> None:
    evidence = _write_evidence(
        tmp_path / "evidence.json",
        [_base_claim(claim_status="goal", release_decision="accepted_as_proven")],
    )
    assert validate(evidence, audit_root=audit_root, manifest_scope="referenced")


def test_partial_claim_presented_as_proven_fails(tmp_path: Path, audit_root: Path) -> None:
    evidence = _write_evidence(
        tmp_path / "evidence.json",
        [_base_claim(claim_status="partial", release_decision="allow")],
    )
    assert validate(evidence, audit_root=audit_root, manifest_scope="referenced")


def test_quarantine_claim_used_as_release_evidence_fails(tmp_path: Path, audit_root: Path) -> None:
    evidence = _write_evidence(
        tmp_path / "evidence.json",
        [_base_claim(claim_status="quarantine", release_decision="allow")],
    )
    assert validate(evidence, audit_root=audit_root, manifest_scope="referenced")


def test_remove_claim_used_as_accepted_evidence_fails(tmp_path: Path, audit_root: Path) -> None:
    evidence = _write_evidence(
        tmp_path / "evidence.json",
        [_base_claim(claim_status="remove", release_decision="allow")],
    )
    assert validate(evidence, audit_root=audit_root, manifest_scope="referenced")


def test_malformed_evidence_index_fails(tmp_path: Path, audit_root: Path) -> None:
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps({"schema_version": "1.1", "claims": {}}), encoding="utf-8")
    assert validate(path, audit_root=audit_root, manifest_scope="referenced")


def test_artifact_manifest_valid_passes(tmp_path: Path, audit_root: Path) -> None:
    baseline = audit_root / "baseline_run_test_valid"
    baseline.mkdir(parents=True)
    artifact = baseline / "a.txt"
    artifact.write_text("x", encoding="utf-8")

    import hashlib

    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (baseline / "artifact_manifest.sha256").write_text(f"{digest}  a.txt\n", encoding="utf-8")

    evidence = _write_evidence(
        tmp_path / "evidence.json",
        [_base_claim(artifact_path="baseline_run_test_valid/a.txt")],
    )
    assert validate(evidence, audit_root=audit_root, manifest_scope="referenced") == []


def test_manifest_mismatch_fails_for_referenced_baseline(tmp_path: Path, audit_root: Path) -> None:
    baseline = audit_root / "baseline_run_test_mismatch"
    baseline.mkdir(parents=True)
    artifact = baseline / "pytest.log"
    artifact.write_text("ok", encoding="utf-8")
    (baseline / "artifact_manifest.sha256").write_text("deadbeef  coverage.xml\n", encoding="utf-8")

    evidence = _write_evidence(
        tmp_path / "evidence.json",
        [_base_claim(artifact_path="baseline_run_test_mismatch/pytest.log")],
    )
    errors = validate(evidence, audit_root=audit_root, manifest_scope="referenced")
    assert any("manifest missing file" in e or "manifest hash mismatch" in e for e in errors)


def test_failed_baseline_only_allowed_for_failure_detection(
    tmp_path: Path, audit_root: Path
) -> None:
    baseline = audit_root / "baseline_run_test_failed"
    baseline.mkdir(parents=True)
    (baseline / "exit_code.txt").write_text("1\n", encoding="utf-8")
    artifact = baseline / "junit.xml"
    artifact.write_text("<testsuite/>", encoding="utf-8")

    evidence = _write_evidence(
        tmp_path / "evidence.json",
        [_base_claim(artifact_path=str(artifact), claim_text="test suite passes")],
    )
    assert validate(evidence, audit_root=audit_root, manifest_scope="referenced")


def test_manifest_scope_all_reports_unreferenced_manifest_errors(
    tmp_path: Path, audit_root: Path
) -> None:
    artifact = tmp_path / "proof.txt"
    artifact.write_text("ok", encoding="utf-8")
    evidence = _write_evidence(
        tmp_path / "evidence.json", [_base_claim(artifact_path=str(artifact))]
    )

    unreferenced = audit_root / "baseline_run_test_unreferenced"
    unreferenced.mkdir(parents=True)
    (unreferenced / "artifact_manifest.sha256").write_text(
        "deadbeef  pytest.log\n", encoding="utf-8"
    )

    errors = validate(evidence, audit_root=audit_root, manifest_scope="all")
    assert any("manifest missing file" in e for e in errors)


def test_manifest_scope_none_skips_manifest_validation(tmp_path: Path, audit_root: Path) -> None:
    artifact = tmp_path / "proof.txt"
    artifact.write_text("ok", encoding="utf-8")
    evidence = _write_evidence(
        tmp_path / "evidence.json", [_base_claim(artifact_path=str(artifact))]
    )

    bad = audit_root / "baseline_run_test_none"
    bad.mkdir(parents=True)
    (bad / "artifact_manifest.sha256").write_text("deadbeef  missing.log\n", encoding="utf-8")

    assert validate(evidence, audit_root=audit_root, manifest_scope="none") == []


def test_relative_artifact_path_resolves_against_audit_root(
    tmp_path: Path, audit_root: Path
) -> None:
    baseline = audit_root / "baseline_run_test_rel"
    baseline.mkdir(parents=True)
    artifact = baseline / "pytest.log"
    artifact.write_text("ok", encoding="utf-8")

    evidence = _write_evidence(
        tmp_path / "evidence.json", [_base_claim(artifact_path="baseline_run_test_rel/pytest.log")]
    )
    assert validate(evidence, audit_root=audit_root, manifest_scope="referenced") == []


def test_relative_artifact_path_marks_referenced_baseline_for_manifest_validation(
    tmp_path: Path, audit_root: Path
) -> None:
    baseline = audit_root / "baseline_run_test_rel_bad"
    baseline.mkdir(parents=True)
    artifact = baseline / "pytest.log"
    artifact.write_text("ok", encoding="utf-8")
    (baseline / "artifact_manifest.sha256").write_text("deadbeef  coverage.xml\n", encoding="utf-8")

    evidence = _write_evidence(
        tmp_path / "evidence.json",
        [_base_claim(artifact_path="baseline_run_test_rel_bad/pytest.log")],
    )
    errors = validate(evidence, audit_root=audit_root, manifest_scope="referenced")
    assert any("manifest missing file" in e for e in errors)


def test_default_manifest_scope_validates_all_manifests(tmp_path: Path, audit_root: Path) -> None:
    artifact = tmp_path / "proof.txt"
    artifact.write_text("ok", encoding="utf-8")
    evidence = _write_evidence(
        tmp_path / "evidence.json", [_base_claim(artifact_path=str(artifact))]
    )

    bad = audit_root / "baseline_run_test_global"
    bad.mkdir(parents=True)
    (bad / "artifact_manifest.sha256").write_text("deadbeef  missing.log\n", encoding="utf-8")

    errors = validate(evidence, audit_root=audit_root)
    assert any("manifest missing file" in e for e in errors)


def test_manifest_scope_referenced_ignores_unreferenced_bad_manifest(
    tmp_path: Path, audit_root: Path
) -> None:
    artifact = tmp_path / "proof.txt"
    artifact.write_text("ok", encoding="utf-8")
    evidence = _write_evidence(
        tmp_path / "evidence.json", [_base_claim(artifact_path=str(artifact))]
    )

    bad = audit_root / "baseline_run_test_unref"
    bad.mkdir(parents=True)
    (bad / "artifact_manifest.sha256").write_text("deadbeef  missing.log\n", encoding="utf-8")

    assert validate(evidence, audit_root=audit_root, manifest_scope="referenced") == []


def test_relative_artifact_path_resolves_when_cwd_changes(
    tmp_path: Path, audit_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = audit_root / "baseline_run_test_cwd"
    baseline.mkdir(parents=True)
    artifact = baseline / "pytest.log"
    artifact.write_text("ok", encoding="utf-8")

    evidence = _write_evidence(
        tmp_path / "evidence.json", [_base_claim(artifact_path="baseline_run_test_cwd/pytest.log")]
    )

    other = tmp_path / "othercwd"
    other.mkdir()
    monkeypatch.chdir(other)

    assert validate(evidence, audit_root=audit_root, manifest_scope="referenced") == []