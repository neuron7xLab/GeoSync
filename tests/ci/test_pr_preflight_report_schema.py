# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Structural schema tests for PR preflight reports."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import stat
from typing import Any

from tools.ci.pr_preflight import run_preflight

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "schemas" / "pr_preflight_report.schema.json"
CHECK_STATUSES = {"PASS", "FAIL", "SKIPPED_OPTIONAL", "BLOCKED", "TIMEOUT"}
FINAL_STATUSES = {"PASS", "FAIL", "BLOCKED"}
SUMMARY_KEYS = {"passed", "failed", "skipped_optional", "blocked", "timeouts"}


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _stub_repo(tmp_path: Path, monkeypatch: object, *, fail_ruff: bool = False) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    root.mkdir(parents=True)
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
    _write_executable(bin_dir / "ruff", "#!/bin/sh\necho ruff fail\nexit 7\n" if fail_ruff else "#!/bin/sh\necho ruff\nexit 0\n")
    monkeypatch.setenv("PATH", str(bin_dir))
    return root, tmp_path / "report"


def _assert_report_matches_schema(report: dict[str, Any]) -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    allowed_top = set(schema["properties"])
    assert set(report) == allowed_top
    assert set(schema["required"]) == allowed_top
    assert report["schema_version"] == 1
    assert report["status"] in FINAL_STATUSES
    assert isinstance(report["root"], str) and report["root"]
    assert isinstance(report["first_file_to_open"], str) and report["first_file_to_open"].endswith("preflight_report.json")
    assert isinstance(report["first_failing_log"], str)
    assert isinstance(report["failure_count"], int) and report["failure_count"] >= 0
    assert isinstance(report["duration_seconds"], int | float) and report["duration_seconds"] >= 0
    assert isinstance(report["next_action"], str)
    summary = report["summary"]
    assert isinstance(summary, dict)
    assert set(summary) == SUMMARY_KEYS
    assert all(isinstance(value, int) and value >= 0 for value in summary.values())
    checks = report["checks"]
    assert isinstance(checks, list) and checks
    check_allowed = set(schema["properties"]["checks"]["items"]["properties"])
    for check in checks:
        assert set(check) == check_allowed
        assert check["status"] in CHECK_STATUSES
        assert isinstance(check["command"], list)
        assert isinstance(check["stdout_log"], str) and check["stdout_log"]
        assert isinstance(check["stderr_log"], str) and check["stderr_log"]


def _assert_schema_rejects(report: dict[str, Any]) -> None:
    try:
        _assert_report_matches_schema(report)
    except AssertionError:
        return
    raise AssertionError("schema validator accepted invalid report")


def test_report_schema_file_is_structural_contract() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_version"] == {"const": 1}
    assert schema["properties"]["status"]["enum"] == ["PASS", "FAIL", "BLOCKED"]
    check_statuses = schema["properties"]["checks"]["items"]["properties"]["status"]["enum"]
    assert set(check_statuses) == CHECK_STATUSES
    assert schema["properties"]["checks"]["items"]["additionalProperties"] is False
    assert set(schema["properties"]["summary"]["required"]) == SUMMARY_KEYS


def test_schema_accepts_generated_pass_fail_and_blocked_reports(tmp_path: Path, monkeypatch: object) -> None:
    pass_root, pass_report_dir = _stub_repo(tmp_path / "pass", monkeypatch, fail_ruff=False)
    pass_code, pass_report = run_preflight(root=pass_root, report_dir=pass_report_dir, skip_install=False)
    assert pass_code == 0
    assert pass_report["status"] == "PASS"
    _assert_report_matches_schema(pass_report)

    fail_root, fail_report_dir = _stub_repo(tmp_path / "fail", monkeypatch, fail_ruff=True)
    fail_code, fail_report = run_preflight(root=fail_root, report_dir=fail_report_dir, skip_install=False)
    assert fail_code == 1
    assert fail_report["status"] == "FAIL"
    _assert_report_matches_schema(fail_report)

    blocked_code, blocked_report = run_preflight(
        root=tmp_path / "missing-root",
        report_dir=tmp_path / "blocked-report",
        skip_install=False,
    )
    assert blocked_code == 1
    assert blocked_report["status"] == "BLOCKED"
    _assert_report_matches_schema(blocked_report)


def test_schema_rejects_unknown_fields_missing_required_and_wrong_version(tmp_path: Path, monkeypatch: object) -> None:
    root, report_dir = _stub_repo(tmp_path, monkeypatch, fail_ruff=True)
    _, report = run_preflight(root=root, report_dir=report_dir, skip_install=False)

    unknown_field = deepcopy(report)
    unknown_field["unexpected"] = True
    _assert_schema_rejects(unknown_field)

    missing_required = deepcopy(report)
    del missing_required["first_file_to_open"]
    _assert_schema_rejects(missing_required)

    wrong_version = deepcopy(report)
    wrong_version["schema_version"] = 2
    _assert_schema_rejects(wrong_version)
