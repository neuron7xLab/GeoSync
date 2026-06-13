# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Invariant tests for PR preflight reports."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import stat
from typing import Any

from tools.ci import pr_preflight

CRITICAL_FAILURE_STATUSES = {"FAIL", "BLOCKED", "TIMEOUT"}
SUMMARY_BY_STATUS = {
    "PASS": "passed",
    "FAIL": "failed",
    "SKIPPED_OPTIONAL": "skipped_optional",
    "BLOCKED": "blocked",
    "TIMEOUT": "timeouts",
}


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _repo_with_stubs(tmp_path: Path, monkeypatch: Any, *, fail_ruff: bool = True) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "requirements.txt").write_text("# runtime\n", encoding="utf-8")
    (root / "requirements-dev.txt").write_text("# dev\n", encoding="utf-8")
    (root / "constraints").mkdir()
    (root / "constraints" / "security.txt").write_text("# constraints\n", encoding="utf-8")
    (root / "tests").mkdir()
    for dirname in ("core", "backtest", "execution", "src", "application"):
        (root / dirname).mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("pip", "black", "mypy", "pytest", "detect-secrets"):
        _write_executable(bin_dir / tool, f"#!/bin/sh\necho {tool}\nexit 0\n")
    ruff_exit = 7 if fail_ruff else 0
    _write_executable(bin_dir / "ruff", f"#!/bin/sh\necho ruff\nexit {ruff_exit}\n")
    monkeypatch.setenv("PATH", str(bin_dir))
    return root, tmp_path / "report"


def _assert_invariants(report: dict[str, Any], report_dir: Path) -> None:
    checks = report["checks"]
    failure_count = sum(1 for check in checks if check["critical"] and check["status"] in CRITICAL_FAILURE_STATUSES)
    assert report["failure_count"] == failure_count
    if report["failure_count"] > 0:
        assert report["status"] in {"FAIL", "BLOCKED"}
    else:
        assert report["status"] == "PASS"
    assert all(not (check["critical"] and check["status"] == "SKIPPED_OPTIONAL") for check in checks)
    for check in checks:
        if check["status"] != "PASS":
            assert check["failure_reason"]
        assert check["stdout_log"]
        assert check["stderr_log"]
        assert Path(check["stdout_log"]).is_relative_to(report_dir)
        assert Path(check["stderr_log"]).is_relative_to(report_dir)
        assert isinstance(check["command"], list)
    ids = [check["id"] for check in checks]
    assert len(ids) == len(set(ids))
    counts = Counter(check["status"] for check in checks)
    for status, summary_key in SUMMARY_BY_STATUS.items():
        assert report["summary"][summary_key] == counts[status]


def test_report_invariants_hold_for_generated_reports(tmp_path: Path, monkeypatch: Any) -> None:
    root, report_dir = _repo_with_stubs(tmp_path, monkeypatch, fail_ruff=True)
    code, report = pr_preflight.run_preflight(root=root, report_dir=report_dir, skip_install=False)

    assert code == 1
    _assert_invariants(report, report_dir.resolve())


def test_unknown_status_collapses_to_blocked() -> None:
    normalized = pr_preflight._normalize_statuses(
        [
            {
                "id": "synthetic",
                "name": "Synthetic",
                "critical": True,
                "status": "UNKNOWN",
                "command": [],
                "exit_code": None,
                "duration_seconds": 0.0,
                "stdout_log": "out.log",
                "stderr_log": "err.log",
                "failure_reason": "",
            }
        ]
    )

    assert normalized[0]["status"] == "BLOCKED"
    assert "unknown status" in str(normalized[0]["failure_reason"])
    assert pr_preflight.summarize(normalized)["blocked"] == 1
    assert pr_preflight._final_status(normalized) == "BLOCKED"


def test_runner_does_not_use_shell_true() -> None:
    source = Path(pr_preflight.__file__).read_text(encoding="utf-8")
    assert "shell=True" not in source
