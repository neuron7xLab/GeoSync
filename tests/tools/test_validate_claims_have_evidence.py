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
    payload = {"claims": claims}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_valid_proven_claim_with_real_artifact_passes(tmp_path: Path, audit_root: Path) -> None:
    artifact = tmp_path / "proof.txt"
    artifact.write_text("ok", encoding="utf-8")
    evidence = _write_evidence(tmp_path / "evidence.json", [{"claim_id": "C1", "status": "proven", "artifact_path": str(artifact)}])
    assert validate(evidence, audit_root=audit_root) == []


def test_artifact_manifest_valid_passes(tmp_path: Path, audit_root: Path) -> None:
    baseline = audit_root / "baseline_run_test_valid"
    baseline.mkdir(parents=True)
    artifact = baseline / "pytest.log"
    artifact.write_text("ok", encoding="utf-8")
    (baseline / "artifact_manifest.sha256").write_text("deadbeef pytest.log\n", encoding="utf-8")
    evidence = _write_evidence(tmp_path / "evidence.json", [{"claim_id": "C1", "status": "proven", "artifact_path": str(artifact)}])
    assert validate(evidence, audit_root=audit_root) == []


def test_manifest_mismatch_fails_for_referenced_baseline(tmp_path: Path, audit_root: Path) -> None:
    baseline = audit_root / "baseline_run_test_mismatch"
    baseline.mkdir(parents=True)
    artifact = baseline / "pytest.log"
    artifact.write_text("ok", encoding="utf-8")
    (baseline / "artifact_manifest.sha256").write_text("deadbeef coverage.xml\n", encoding="utf-8")
    evidence = _write_evidence(tmp_path / "evidence.json", [{"claim_id": "C1", "status": "proven", "artifact_path": str(artifact)}])
    errors = validate(evidence, audit_root=audit_root)
    assert any("Manifest references missing artifact" in error for error in errors)


def test_failed_baseline_fails_proven_claim(tmp_path: Path, audit_root: Path) -> None:
    baseline = audit_root / "baseline_run_test_failed"
    baseline.mkdir(parents=True)
    (baseline / "exit_code.txt").write_text("1\n", encoding="utf-8")
    artifact = baseline / "pytest.log"
    artifact.write_text("ok", encoding="utf-8")
    evidence = _write_evidence(tmp_path / "evidence.json", [{"claim_id": "C1", "status": "proven", "artifact_path": str(artifact)}])
    errors = validate(evidence, audit_root=audit_root)
    assert any("artifact belongs to failed baseline" in error for error in errors)


def test_validate_custom_evidence_path_does_not_scan_repo_global_manifests(tmp_path: Path) -> None:
    audit_root = tmp_path / "reports" / "test_audit"
    audit_root.mkdir(parents=True)
    artifact = tmp_path / "proof.txt"
    artifact.write_text("ok", encoding="utf-8")
    evidence = _write_evidence(tmp_path / "evidence.json", [{"claim_id": "C1", "status": "proven", "artifact_path": str(artifact)}])
    assert validate(evidence, audit_root=audit_root) == []


def test_manifest_scope_all_reports_unreferenced_manifest_errors(tmp_path: Path, audit_root: Path) -> None:
    artifact = tmp_path / "proof.txt"
    artifact.write_text("ok", encoding="utf-8")
    evidence = _write_evidence(tmp_path / "evidence.json", [{"claim_id": "C1", "status": "proven", "artifact_path": str(artifact)}])

    unreferenced = audit_root / "baseline_run_test_unreferenced"
    unreferenced.mkdir(parents=True)
    (unreferenced / "artifact_manifest.sha256").write_text("deadbeef pytest.log\n", encoding="utf-8")

    errors = validate(evidence, audit_root=audit_root, manifest_scope="all")
    assert any("Manifest references missing artifact" in error for error in errors)


def test_manifest_scope_none_skips_manifest_validation(tmp_path: Path, audit_root: Path) -> None:
    artifact = tmp_path / "proof.txt"
    artifact.write_text("ok", encoding="utf-8")
    evidence = _write_evidence(tmp_path / "evidence.json", [{"claim_id": "C1", "status": "proven", "artifact_path": str(artifact)}])

    bad = audit_root / "baseline_run_test_none"
    bad.mkdir(parents=True)
    (bad / "artifact_manifest.sha256").write_text("deadbeef missing.log\n", encoding="utf-8")

    assert validate(evidence, audit_root=audit_root, manifest_scope="none") == []


def test_manifest_scope_referenced_ignores_unreferenced_bad_manifest(tmp_path: Path, audit_root: Path) -> None:
    artifact = tmp_path / "proof.txt"
    artifact.write_text("ok", encoding="utf-8")
    evidence = _write_evidence(tmp_path / "evidence.json", [{"claim_id": "C1", "status": "proven", "artifact_path": str(artifact)}])

    bad = audit_root / "baseline_run_test_unreferenced_bad"
    bad.mkdir(parents=True)
    (bad / "artifact_manifest.sha256").write_text("deadbeef missing.log\n", encoding="utf-8")

    assert validate(evidence, audit_root=audit_root, manifest_scope="referenced") == []


def test_manifest_scope_referenced_fails_referenced_bad_manifest(tmp_path: Path, audit_root: Path) -> None:
    baseline = audit_root / "baseline_run_test_bad"
    baseline.mkdir(parents=True)
    artifact = baseline / "pytest.log"
    artifact.write_text("ok", encoding="utf-8")
    (baseline / "artifact_manifest.sha256").write_text("deadbeef coverage.xml\n", encoding="utf-8")

    evidence = _write_evidence(
        tmp_path / "evidence.json",
        [{"claim_id": "C1", "status": "proven", "artifact_path": str(artifact)}],
    )
    errors = validate(evidence, audit_root=audit_root, manifest_scope="referenced")
    assert any("Manifest references missing artifact" in error for error in errors)
