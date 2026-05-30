#!/usr/bin/env python3
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def regression_chain_passes() -> None:
    for args in [
        ["tools/cns_program/deploy_release_gate.py"],
        ["tools/cns_program/build_release_manifest.py"],
        ["tools/cns_program/verify_release_manifest.py"],
        ["tools/cns_program/verify_reports_contract.py"],
        ["tools/cns_program/quality_gate.py"],
    ]:
        completed = run(args)
        require(completed.returncode == 0, completed.stderr or completed.stdout)

    payload = json.loads(
        (ROOT / "results/cns_quality_gate.json").read_text(encoding="utf-8")
    )
    require(payload["passed"] is True, "quality gate did not pass")
    require(payload["quality_score"] == 1.0, "quality score is not 1.0")


def regression_missing_report_fails() -> None:
    payload = verify_reports((Path("reports/__missing__.md"),))
    require(payload["valid"] is False, "missing report unexpectedly passed")
    require(
        payload["errors"] == ["report_missing:reports/__missing__.md"],
        "missing report error mismatch",
    )


def main() -> int:
    regression_chain_passes()
    regression_missing_report_fails()
    print(json.dumps({"cns_regression_gate": "PASS"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
