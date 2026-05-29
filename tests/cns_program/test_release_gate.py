from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.cns_program.verify_reports_contract import verify_reports

ROOT = Path(__file__).resolve().parents[2]


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_release_gate_chain_green() -> None:
    for args in [
        ["tools/cns_program/deploy_release_gate.py"],
        ["tools/cns_program/build_release_manifest.py"],
        ["tools/cns_program/verify_release_manifest.py"],
        ["tools/cns_program/verify_reports_contract.py"],
        ["tools/cns_program/quality_gate.py"],
    ]:
        completed = run_command(args)
        assert completed.returncode == 0, completed.stderr or completed.stdout

    payload = json.loads(
        (ROOT / "results/cns_quality_gate.json").read_text(encoding="utf-8")
    )
    assert payload["passed"] is True
    assert payload["quality_score"] == 1.0


def test_manifest_verifier_negative_case(tmp_path: Path) -> None:
    assert run_command(["tools/cns_program/deploy_release_gate.py"]).returncode == 0
    assert run_command(["tools/cns_program/build_release_manifest.py"]).returncode == 0
    manifest_path = ROOT / "results/cns_release_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = next(iter(data["sha256"]))
    data["sha256"][first] = "0" * 64
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(data), encoding="utf-8")
    completed = run_command(
        ["tools/cns_program/verify_release_manifest.py", "--manifest", str(candidate)]
    )
    payload = json.loads(completed.stdout)
    assert completed.returncode != 0
    assert payload["valid"] is False
    assert payload["errors"]


def test_reports_contract_negative_case() -> None:
    payload = verify_reports((Path("reports/__absent__.md"),))
    assert payload["valid"] is False
    assert payload["errors"]
