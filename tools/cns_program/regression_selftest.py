#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def assert_ok(args: list[str]) -> None:
    completed = run(args)
    if completed.returncode != 0:
        raise AssertionError(completed.stderr or completed.stdout)


def test_release_chain() -> None:
    for args in [
        ["tools/cns_program/deploy_release_gate.py"],
        ["tools/cns_program/build_release_manifest.py"],
        ["tools/cns_program/verify_release_manifest.py"],
        ["tools/cns_program/verify_reports_contract.py"],
        ["tools/cns_program/quality_gate.py"],
    ]:
        assert_ok(args)

    payload = json.loads((ROOT / "results/cns_quality_gate.json").read_text(encoding="utf-8"))
    if payload.get("passed") is not True:
        raise AssertionError("quality gate did not pass")
    if payload.get("quality_score") != 1.0:
        raise AssertionError("quality score drifted")


def test_manifest_negative_case() -> None:
    assert_ok(["tools/cns_program/deploy_release_gate.py"])
    assert_ok(["tools/cns_program/build_release_manifest.py"])
    manifest_path = ROOT / "results/cns_release_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = next(iter(data["sha256"]))
    data["sha256"][first] = "0" * 64

    with tempfile.TemporaryDirectory() as tmpdir:
        candidate = Path(tmpdir) / "candidate.json"
        candidate.write_text(json.dumps(data), encoding="utf-8")
        completed = run(["tools/cns_program/verify_release_manifest.py", "--manifest", str(candidate)])
    if completed.returncode == 0:
        raise AssertionError("manifest verifier accepted changed artifact")
    payload = json.loads(completed.stdout)
    if payload.get("valid") is not False:
        raise AssertionError("negative manifest case did not report invalid")
    if not payload.get("errors"):
        raise AssertionError("negative manifest case did not report errors")


def test_reports_negative_case() -> None:
    from tools.cns_program.verify_reports_contract import verify_reports

    payload = verify_reports((Path("reports/__absent__.md"),))
    if payload.get("valid") is not False:
        raise AssertionError("missing report case did not report invalid")
    if not payload.get("errors"):
        raise AssertionError("missing report case did not report errors")


def main() -> int:
    test_release_chain()
    test_manifest_negative_case()
    test_reports_negative_case()
    print(json.dumps({"cns_regression_selftest": "passed"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
