#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Structured local PR preflight runner.

This module keeps the user-facing shell command stable while moving the actual
preflight policy into deterministic Python code that preserves evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PASS = "PASS"
FAIL = "FAIL"
SKIPPED_OPTIONAL = "SKIPPED_OPTIONAL"
BLOCKED = "BLOCKED"
TIMEOUT = "TIMEOUT"

FINAL_PASS = "PASS"
FINAL_FAIL = "FAIL"
FINAL_BLOCKED = "BLOCKED"

ALLOWED_CHECK_STATUSES = {PASS, FAIL, SKIPPED_OPTIONAL, BLOCKED, TIMEOUT}
ALLOWED_FINAL_STATUSES = {FINAL_PASS, FINAL_FAIL, FINAL_BLOCKED}


@dataclass(frozen=True)
class CheckSpec:
    id: str
    name: str
    command: list[str]
    critical: bool = True
    optional_if_missing: bool = False
    cwd: str = "."
    timeout_seconds: int = 300
    success_exit_codes: list[int] = field(default_factory=lambda: [0])
    stdout_log: str = ""
    stderr_log: str = ""
    skip_reason: str = ""

    def __post_init__(self) -> None:
        if not self.stdout_log:
            object.__setattr__(self, "stdout_log", f"logs/{self.id}.stdout.log")
        if not self.stderr_log:
            object.__setattr__(self, "stderr_log", f"logs/{self.id}.stderr.log")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _python_executable() -> str:
    return os.environ.get("PR_PREFLIGHT_PYTHON", sys.executable)


def _requirements_install_command(root: Path, python_executable: str) -> list[str]:
    requirements = [
        name for name in ("requirements.txt", "requirements-dev.txt") if (root / name).exists()
    ]
    if requirements:
        command = [python_executable, "-m", "pip", "install"]
        if (root / "constraints" / "security.txt").exists():
            command.extend(["-c", "constraints/security.txt"])
        for requirement in requirements:
            command.extend(["-r", requirement])
        return command
    if (root / "pyproject.toml").exists():
        return [python_executable, "-m", "pip", "install", "-e", ".[dev]"]
    return []


def build_check_registry(root: Path, skip_install: bool = False) -> list[CheckSpec]:
    python_executable = _python_executable()
    if skip_install:
        install_checks = [
            CheckSpec(
                id="pip_bootstrap",
                name="Pip bootstrap",
                command=[],
                critical=False,
                skip_reason="skipped by --skip-install",
            ),
            CheckSpec(
                id="project_dependencies",
                name="Project dependencies",
                command=[],
                critical=False,
                skip_reason="skipped by --skip-install",
            ),
        ]
    else:
        dependency_command = _requirements_install_command(root, python_executable)
        install_checks = [
            CheckSpec(
                id="pip_bootstrap",
                name="Pip bootstrap",
                command=[
                    python_executable,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "pip",
                    "setuptools",
                    "wheel",
                    "-q",
                ],
                critical=True,
                timeout_seconds=300,
            ),
            CheckSpec(
                id="project_dependencies",
                name="Project dependencies",
                command=dependency_command,
                critical=bool(dependency_command),
                skip_reason="no requirements or pyproject dependency manifest found",
                timeout_seconds=900,
            ),
        ]

    return [
        *install_checks,
        CheckSpec(
            id="ruff",
            name="Ruff lint",
            command=["ruff", "check", "."],
            critical=True,
            timeout_seconds=300,
        ),
        CheckSpec(
            id="black",
            name="Black format check",
            command=["black", "--check", ".", "--quiet"],
            critical=True,
            timeout_seconds=300,
        ),
        CheckSpec(
            id="mypy",
            name="Mypy type check",
            command=["mypy", "--no-error-summary", "."],
            critical=True,
            timeout_seconds=300,
        ),
        CheckSpec(
            id="detect_secrets",
            name="Detect secrets scan",
            command=[
                "detect-secrets",
                "scan",
                "core",
                "backtest",
                "execution",
                "src",
                "application",
                "--baseline",
                ".secrets.baseline",
            ],
            critical=True,
            optional_if_missing=True,
            timeout_seconds=300,
        ),
        CheckSpec(
            id="quick_tests",
            name="Quick pytest suite",
            command=[
                "pytest",
                "tests/",
                "-m",
                "not slow and not flaky and not integration",
                "--maxfail=3",
                "-q",
                "--tb=no",
            ],
            critical=True,
            timeout_seconds=600,
        ),
        CheckSpec(
            id="coverage_artifact",
            name="Coverage artifact observation",
            command=[],
            critical=False,
            timeout_seconds=1,
            skip_reason="coverage.xml is absent",
        ),
    ]


def _tool_available(command: list[str]) -> tuple[bool, str]:
    if not command:
        return True, ""
    executable = command[0]
    if os.sep in executable or (os.altsep and os.altsep in executable):
        return Path(executable).exists(), executable
    resolved = shutil.which(executable)
    return resolved is not None, resolved or executable


def _log_path(report_dir: Path, relative_log_path: str) -> Path:
    return report_dir / relative_log_path


def _base_result(
    spec: CheckSpec,
    root: Path,
    stdout_log: Path,
    stderr_log: Path,
    critical: bool,
) -> dict[str, Any]:
    return {
        "id": spec.id,
        "name": spec.name,
        "critical": critical,
        "status": BLOCKED,
        "command": list(spec.command),
        "exit_code": None,
        "duration_seconds": 0.0,
        "stdout_log": _display_path(stdout_log, root),
        "stderr_log": _display_path(stderr_log, root),
        "failure_reason": "",
        "tool_available": False,
        "cwd": spec.cwd,
        "timeout_seconds": spec.timeout_seconds,
        "success_exit_codes": list(spec.success_exit_codes),
        "optional_if_missing": spec.optional_if_missing,
    }


def _empty_check_result(
    spec: CheckSpec,
    root: Path,
    stdout_log: Path,
    stderr_log: Path,
    status: str,
    critical: bool,
    failure_reason: str,
    tool_available: bool,
    duration_seconds: float = 0.0,
) -> dict[str, Any]:
    _write_text(stdout_log, "")
    _write_text(stderr_log, "")
    result = _base_result(spec, root, stdout_log, stderr_log, critical)
    result.update(
        {
            "status": status,
            "duration_seconds": round(duration_seconds, 6),
            "failure_reason": failure_reason,
            "tool_available": tool_available,
        }
    )
    return result


def run_check(spec: CheckSpec, root: Path, report_dir: Path) -> dict[str, Any]:
    stdout_log = _log_path(report_dir, spec.stdout_log)
    stderr_log = _log_path(report_dir, spec.stderr_log)
    started = time.monotonic()

    if spec.id == "coverage_artifact":
        coverage_path = root / "coverage.xml"
        stdout = f"coverage_xml={coverage_path.exists()}\n"
        _write_text(stdout_log, stdout)
        _write_text(stderr_log, "")
        status = PASS if coverage_path.exists() else SKIPPED_OPTIONAL
        reason = "" if coverage_path.exists() else spec.skip_reason
        result = _base_result(spec, root, stdout_log, stderr_log, critical=False)
        result.update(
            {
                "status": status,
                "exit_code": 0 if coverage_path.exists() else None,
                "duration_seconds": round(time.monotonic() - started, 6),
                "failure_reason": reason,
                "tool_available": True,
            }
        )
        return result

    if not spec.command:
        return _empty_check_result(
            spec=spec,
            root=root,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            status=SKIPPED_OPTIONAL,
            critical=False,
            failure_reason=spec.skip_reason or "check disabled by configuration",
            tool_available=True,
            duration_seconds=time.monotonic() - started,
        )

    critical = spec.critical
    cwd = (root / spec.cwd).resolve()
    if not cwd.exists():
        return _empty_check_result(
            spec=spec,
            root=root,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            status=BLOCKED,
            critical=critical,
            failure_reason=f"cwd does not exist: {cwd}",
            tool_available=True,
            duration_seconds=time.monotonic() - started,
        )

    available, executable = _tool_available(spec.command)
    if not available and spec.optional_if_missing:
        return _empty_check_result(
            spec=spec,
            root=root,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            status=SKIPPED_OPTIONAL,
            critical=False,
            failure_reason=f"optional executable missing: {executable}",
            tool_available=False,
            duration_seconds=time.monotonic() - started,
        )
    if not available:
        return _empty_check_result(
            spec=spec,
            root=root,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            status=BLOCKED,
            critical=critical,
            failure_reason=f"required executable missing: {executable}",
            tool_available=False,
            duration_seconds=time.monotonic() - started,
        )

    try:
        completed = subprocess.run(
            spec.command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=spec.timeout_seconds,
            check=False,
        )
        stdout = _coerce_text(completed.stdout)
        stderr = _coerce_text(completed.stderr)
        _write_text(stdout_log, stdout)
        _write_text(stderr_log, stderr)
        exit_code = completed.returncode
        status = PASS if exit_code in spec.success_exit_codes else FAIL
        reason = ""
        if status == FAIL:
            reason = f"exit code {exit_code} not in success codes {spec.success_exit_codes}"
        result = _base_result(spec, root, stdout_log, stderr_log, critical)
        result.update(
            {
                "status": status,
                "exit_code": exit_code,
                "duration_seconds": round(time.monotonic() - started, 6),
                "failure_reason": reason,
                "tool_available": True,
            }
        )
        return result
    except subprocess.TimeoutExpired as exc:
        _write_text(stdout_log, _coerce_text(exc.stdout))
        _write_text(stderr_log, _coerce_text(exc.stderr))
        result = _base_result(spec, root, stdout_log, stderr_log, critical)
        result.update(
            {
                "status": TIMEOUT,
                "duration_seconds": round(time.monotonic() - started, 6),
                "failure_reason": f"timed out after {spec.timeout_seconds} seconds",
                "tool_available": True,
            }
        )
        return result
    except OSError as exc:
        _write_text(stdout_log, "")
        _write_text(stderr_log, str(exc))
        result = _base_result(spec, root, stdout_log, stderr_log, critical)
        result.update(
            {
                "status": BLOCKED,
                "duration_seconds": round(time.monotonic() - started, 6),
                "failure_reason": f"could not execute command: {exc}",
                "tool_available": True,
            }
        )
        return result


def summarize(checks: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "passed": sum(1 for check in checks if check["status"] == PASS),
        "failed": sum(1 for check in checks if check["status"] == FAIL),
        "skipped_optional": sum(1 for check in checks if check["status"] == SKIPPED_OPTIONAL),
        "blocked": sum(1 for check in checks if check["status"] == BLOCKED),
        "timeouts": sum(1 for check in checks if check["status"] == TIMEOUT),
    }


def final_status(checks: list[dict[str, Any]]) -> str:
    critical_checks = [check for check in checks if check["critical"]]
    if any(check["status"] == BLOCKED for check in critical_checks):
        return FINAL_BLOCKED
    if any(check["status"] != PASS for check in critical_checks):
        return FINAL_FAIL
    return FINAL_PASS


def first_failure(checks: list[dict[str, Any]]) -> str:
    for check in checks:
        if check["critical"] and check["status"] != PASS:
            reason = check["failure_reason"] or check["status"]
            return f"Fix {check['id']}: {reason}"
    return "No critical failures."


def build_report(
    root: Path,
    report_dir: Path,
    checks: list[dict[str, Any]],
    started_at: str,
) -> dict[str, Any]:
    finished_at = _utc_now()
    summary = summarize(checks)
    status = final_status(checks)
    failure_count = sum(1 for check in checks if check["critical"] and check["status"] != PASS)
    report_path = report_dir / "preflight_report.json"
    return {
        "schema_version": 1,
        "status": status,
        "root": str(root.resolve()),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": 0.0,
        "checks": checks,
        "summary": summary,
        "failure_count": failure_count,
        "next_action": first_failure(checks),
        "first_file_to_open": _display_path(report_path, root),
    }


def run_preflight(root: Path, report_dir: Path, skip_install: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    started_at = _utc_now()
    root = root.resolve()
    report_dir = report_dir.resolve()
    checks = [
        run_check(spec, root, report_dir) for spec in build_check_registry(root, skip_install)
    ]
    report = build_report(root, report_dir, checks, started_at)
    report["duration_seconds"] = round(time.monotonic() - started, 6)
    _write_text(report_dir / "preflight_report.json", json.dumps(report, indent=2, sort_keys=True))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the structured GeoSync PR preflight gate.")
    parser.add_argument(
        "--root", default=".", help="Repository root. Defaults to current directory."
    )
    parser.add_argument(
        "--report-dir",
        default="artifacts/pr_preflight",
        help="Directory for JSON report and per-check logs.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip pip bootstrap and dependency installation checks.",
    )
    parser.add_argument("--json", action="store_true", help="Print full report JSON to stdout.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    report_dir_arg = Path(args.report_dir)
    report_dir = report_dir_arg if report_dir_arg.is_absolute() else root / report_dir_arg
    report = run_preflight(root=root, report_dir=report_dir, skip_install=args.skip_install)
    report_json = json.dumps(report, indent=2, sort_keys=True)
    if args.json:
        print(report_json)
    else:
        print(f"PR_PREFLIGHT_STATUS={report['status']}")
        print(f"FIRST_FILE_TO_OPEN={report['first_file_to_open']}")
        print(f"FAILURE_COUNT={report['failure_count']}")
    return 0 if report["status"] == FINAL_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
