from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.cns_program.verify_reports_contract import verify_reports

ROOT = Path(__file__).resolve().parents[2]


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def load_json_object(text: str) -> dict[str, object]:
    payload: object = json.loads(text)
    if not isinstance(payload, dict):
        raise AssertionError("expected JSON object")
    return {str(key): value for key, value in payload.items()}


def test_release_gate_chain_passes() -> None:
    for args in [
        ["tools/cns_program/deploy_release_gate.py"],
        ["tools/cns_program/build_release_manifest.py"],
        ["tools/cns_program/verify_release_manifest.py"],
        ["tools/cns_program/verify_reports_contract.py"],
        ["tools/cns_program/quality_gate.py"],
    ]:
        completed = run(args)
        assert completed.returncode == 0, completed.stderr or completed.stdout

    payload = load_json_object(
        (ROOT / "results/cns_quality_gate.json").read_text(encoding="utf-8")
    )
    assert payload["passed"] is True
    assert payload["quality_score"] == 1.0


def test_manifest_verifier_fails_on_digest_mismatch(tmp_path: Path) -> None:
    assert run(["tools/cns_program/deploy_release_gate.py"]).returncode == 0
    assert run(["tools/cns_program/build_release_manifest.py"]).returncode == 0
    manifest_path = ROOT / "results/cns_release_manifest.json"
    data = load_json_object(manifest_path.read_text(encoding="utf-8"))
    sha_map = data["sha256"]
    assert isinstance(sha_map, dict)
    first = next(iter(sha_map))
    sha_map[first] = "0" * 64
    bad_manifest = tmp_path / "bad_manifest.json"
    bad_manifest.write_text(json.dumps(data), encoding="utf-8")
    completed = run(
        ["tools/cns_program/verify_release_manifest.py", "--manifest", str(bad_manifest)]
    )
    assert completed.returncode != 0
    payload = load_json_object(completed.stdout)
    assert payload["valid"] is False
    errors = payload["errors"]
    assert isinstance(errors, list)
    assert any(
        isinstance(error, str) and error.startswith("artifact_hash_mismatch:")
        for error in errors
    )


def test_reports_contract_fails_on_missing_report() -> None:
    missing_report = Path("reports/__missing__.md")
    payload = verify_reports((missing_report,))
    assert payload["valid"] is False
    assert payload["errors"] == ["report_missing:reports/__missing__.md"]
