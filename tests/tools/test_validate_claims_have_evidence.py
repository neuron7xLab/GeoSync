from __future__ import annotations

import json
from pathlib import Path

from tools.claims.validate_claims_have_evidence import validate

METRICS_TEMPLATE = """
| id | domain | claim | measurement_command | evidence_path | status | notes |
|----|--------|-------|---------------------|---------------|--------|-------|
{rows}
"""


def _write(path: Path, txt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt, encoding="utf-8")


def _mk_claim(claim_id: str, status: str, notes: str = "") -> str:
    return f"| {claim_id} | coverage | sample | cmd | artifact | {status} | {notes} |"


def test_proven_claim_without_artifact_fails(tmp_path: Path) -> None:
    metrics = tmp_path / "m.md"
    _write(metrics, METRICS_TEMPLATE.format(rows=_mk_claim("C1", "proven")))
    evidence = tmp_path / "e.json"
    evidence.write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "claim_id": "C1",
                        "claim_status": "proven",
                        "command": "x",
                        "artifact_path": "missing.txt",
                        "last_verified_sha": "s",
                        "timestamp_utc": "t",
                    }
                ]
            }
        )
    )
    assert validate(metrics, evidence)


def test_goal_claim_as_achievement_fails(tmp_path: Path) -> None:
    metrics = tmp_path / "m.md"
    _write(metrics, METRICS_TEMPLATE.format(rows=_mk_claim("C2", "goal", "verified achieved")))
    evidence = tmp_path / "e.json"
    evidence.write_text(
        json.dumps({"claims": [{"claim_id": "C2", "claim_status": "goal", "verdict": "accepted"}]})
    )
    assert validate(metrics, evidence)


def test_partial_claim_as_proven_fails(tmp_path: Path) -> None:
    metrics = tmp_path / "m.md"
    _write(metrics, METRICS_TEMPLATE.format(rows=_mk_claim("C3", "partial", "treated as proven")))
    evidence = tmp_path / "e.json"
    evidence.write_text(
        json.dumps(
            {"claims": [{"claim_id": "C3", "claim_status": "partial", "verdict": "accepted"}]}
        )
    )
    assert validate(metrics, evidence)


def test_valid_evidence_passes(tmp_path: Path) -> None:
    artifact = tmp_path / "a.txt"
    artifact.write_text("ok")
    metrics = tmp_path / "m.md"
    _write(metrics, METRICS_TEMPLATE.format(rows=_mk_claim("C4", "proven")))
    evidence = tmp_path / "e.json"
    evidence.write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "claim_id": "C4",
                        "claim_status": "proven",
                        "command": "x",
                        "artifact_path": str(artifact),
                        "last_verified_sha": "sha",
                        "timestamp_utc": "2026-05-22T00:00:00Z",
                    }
                ]
            }
        )
    )
    assert validate(metrics, evidence) == []


def test_missing_timestamp_for_proven_fails(tmp_path: Path) -> None:
    artifact = tmp_path / "a.txt"
    artifact.write_text("ok")
    metrics = tmp_path / "m.md"
    _write(metrics, METRICS_TEMPLATE.format(rows=_mk_claim("C5", "proven")))
    evidence = tmp_path / "e.json"
    evidence.write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "claim_id": "C5",
                        "claim_status": "proven",
                        "command": "x",
                        "artifact_path": str(artifact),
                        "last_verified_sha": "sha",
                        "timestamp_utc": "",
                    }
                ]
            }
        )
    )
    assert validate(metrics, evidence)


def test_missing_sha_for_proven_fails(tmp_path: Path) -> None:
    artifact = tmp_path / "a.txt"
    artifact.write_text("ok")
    metrics = tmp_path / "m.md"
    _write(metrics, METRICS_TEMPLATE.format(rows=_mk_claim("C6", "proven")))
    evidence = tmp_path / "e.json"
    evidence.write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "claim_id": "C6",
                        "claim_status": "proven",
                        "command": "x",
                        "artifact_path": str(artifact),
                        "last_verified_sha": "",
                        "timestamp_utc": "2026-05-22T00:00:00Z",
                    }
                ]
            }
        )
    )
    assert validate(metrics, evidence)
