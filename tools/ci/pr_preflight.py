# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Structured local PR preflight runner.

The runner is intentionally small and standard-library only: it executes a
fixed registry of local preflight checks, preserves per-check stdout/stderr,
and writes a machine-readable report even when critical checks fail.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

CHECK_STATUSES = {"PASS", "FAIL", "SKIPPED_OPTIONAL", "BLOCKED", "TIMEOUT"}
FINAL_STATUSES = {"PASS", "FAIL", "BLOCKED"}
CRITICAL_FAILURE_STATUSES = {"FAIL", "BLOCKED", "TIMEOUT"}


@dataclass(frozen=True)
class Check:
    id: str
    name: str
    command: tuple[str, ...]
    critical: bool
    optional_if_missing: bool = False
    cwd: str = "."
    timeout_seconds: int = 300
    success_exit_codes: tuple[int, ...] = (0,)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _log_paths(report_dir: Path, check_id: str) -> tuple[Path, Path]:
    logs = report_dir / "logs"
    return logs / f"{check_id}.stdout.log", logs / f"{check_id}.stderr.log"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _has_requirements(root: Path) -> bool:
    return (root / "requirements.txt").is_file() and (root / "requirements-dev.txt").is_file()


def build_checks(root: Path, *, skip_install: bool, timeout_seconds: int = 300) -> list[Check]:
    checks: list[Check] = []
    if skip_install:
        checks.extend(
            [
                Check(
                    id="pip_bootstrap",
                    name="Pip bootstrap",
                    command=("pip", "install", "--upgrade", "pip", "setuptools", "wheel", "-q"),
                    critical=False,
                    optional_if_missing=True,
                ),
                Check(
                    id="project_dependencies",
                    name="Project dependencies",
                    command=(
                        "pip",
                        "install",
                        "-c",
                        "constraints/security.txt",
                        "-r",
                        "requirements.txt",
                        "-r",
                        "requirements-dev.txt",
                        "-q",
                    ),
                    critical=False,
                    optional_if_missing=True,
                ),
            ]
        )
    else:
        checks.append(
            Check(
                id="pip_bootstrap",
                name="Pip bootstrap",
                command=("pip", "install", "--upgrade", "pip", "setuptools", "wheel", "-q"),
                critical=True,
            )
        )
        checks.append(
            Check(
                id="project_dependencies",
                name="Project dependencies",
                command=(
                    "pip",
                    "install",
                    "-c",
                    "constraints/security.txt",
                    "-r",
                    "requirements.txt",
                    "-r",
                    "requirements-dev.txt",
                    "-q",
                ),
                critical=_has_requirements(root),
                optional_if_missing=not _has_requirements(root),
            )
        )

    checks.extend(
        [
            Check(id="ruff", name="Ruff", command=("ruff", "check", "."), critical=True),
            Check(id="black", name="Black", command=("black", "--check", ".", "--quiet"), critical=True),
            Check(id="mypy", name="Mypy", command=("mypy", "--no-error-summary"), critical=True),
            Check(
                id="detect_secrets",
                name="Detect secrets",
                command=(
                    "detect-secrets",
                    "scan",
                    "core",
                    "backtest",
                    "execution",
                    "src",
                    "application",
                    "--baseline",
                    ".secrets.baseline",
                ),
                critical=shutil.which("detect-secrets") is not None,
                optional_if_missing=True,
            ),
            Check(
                id="quick_tests",
                name="Quick tests",
                command=(
                    "pytest",
                    "tests/",
                    "-m",
                    "not slow and not flaky and not integration",
                    "--maxfail=3",
                    "-q",
                    "--tb=no",
                ),
                critical=True,
            ),
            Check(
                id="coverage_artifact",
                name="Coverage artifact",
                command=(),
                critical=False,
                optional_if_missing=True,
            ),
        ]
    )
    if timeout_seconds != 300:
        checks = [replace(check, timeout_seconds=timeout_seconds) for check in checks]
    return checks


def _base_record(check: Check, report_dir: Path) -> dict[str, object]:
    stdout_log, stderr_log = _log_paths(report_dir, check.id)
    return {
        "id": check.id,
        "name": check.name,
        "critical": check.critical,
        "status": "BLOCKED",
        "command": list(check.command),
        "exit_code": None,
        "duration_seconds": 0.0,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "failure_reason": "",
    }


def run_check(check: Check, *, root: Path, report_dir: Path) -> dict[str, object]:
    started = time.monotonic()
    record = _base_record(check, report_dir)
    stdout_log = Path(str(record["stdout_log"]))
    stderr_log = Path(str(record["stderr_log"]))

    if check.id == "coverage_artifact":
        _write_text(stdout_log, "")
        _write_text(stderr_log, "")
        record["status"] = "PASS" if (root / "coverage.xml").is_file() else "SKIPPED_OPTIONAL"
        if record["status"] == "SKIPPED_OPTIONAL":
            record["failure_reason"] = "coverage.xml not present; optional observation only"
        record["duration_seconds"] = round(time.monotonic() - started, 6)
        return record

    if check.id in {"pip_bootstrap", "project_dependencies"} and not check.critical:
        _write_text(stdout_log, "")
        _write_text(stderr_log, "")
        record["status"] = "SKIPPED_OPTIONAL"
        record["failure_reason"] = "install checks skipped by --skip-install" if check.id == "pip_bootstrap" else "project dependency install skipped or requirements files absent"
        record["duration_seconds"] = round(time.monotonic() - started, 6)
        return record

    executable = check.command[0] if check.command else ""
    if executable and shutil.which(executable) is None:
        _write_text(stdout_log, "")
        _write_text(stderr_log, f"executable not found: {executable}\n")
        record["status"] = "SKIPPED_OPTIONAL" if check.optional_if_missing and not check.critical else "BLOCKED"
        record["failure_reason"] = f"executable not found: {executable}"
        record["duration_seconds"] = round(time.monotonic() - started, 6)
        return record

    try:
        proc = subprocess.run(
            list(check.command),
            cwd=root / check.cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=check.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        _write_text(stdout_log, exc.stdout or "")
        _write_text(stderr_log, exc.stderr or f"timeout after {check.timeout_seconds}s\n")
        record["status"] = "TIMEOUT"
        record["failure_reason"] = f"timeout after {check.timeout_seconds}s"
        record["duration_seconds"] = round(time.monotonic() - started, 6)
        return record
    except OSError as exc:
        _write_text(stdout_log, "")
        _write_text(stderr_log, f"{exc}\n")
        record["status"] = "BLOCKED"
        record["failure_reason"] = str(exc)
        record["duration_seconds"] = round(time.monotonic() - started, 6)
        return record

    _write_text(stdout_log, proc.stdout)
    _write_text(stderr_log, proc.stderr)
    record["exit_code"] = proc.returncode
    if proc.returncode in check.success_exit_codes:
        record["status"] = "PASS"
    else:
        record["status"] = "FAIL"
        record["failure_reason"] = f"exit code {proc.returncode} not in {list(check.success_exit_codes)}"
    record["duration_seconds"] = round(time.monotonic() - started, 6)
    return record


def summarize(checks: Sequence[dict[str, object]]) -> dict[str, int]:
    return {
        "passed": sum(1 for item in checks if item["status"] == "PASS"),
        "failed": sum(1 for item in checks if item["status"] == "FAIL"),
        "skipped_optional": sum(1 for item in checks if item["status"] == "SKIPPED_OPTIONAL"),
        "blocked": sum(1 for item in checks if item["status"] == "BLOCKED"),
        "timeouts": sum(1 for item in checks if item["status"] == "TIMEOUT"),
    }


def _normalize_statuses(checks: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for item in checks:
        record = dict(item)
        status = record.get("status")
        if status not in CHECK_STATUSES:
            record["status"] = "BLOCKED"
            record["failure_reason"] = f"unknown status collapsed to BLOCKED: {status!r}"
        normalized.append(record)
    return normalized


def _final_status(checks: Sequence[dict[str, object]]) -> str:
    critical_bad = [
        item
        for item in checks
        if item["critical"] and item["status"] in CRITICAL_FAILURE_STATUSES
    ]
    if any(item["status"] == "BLOCKED" for item in critical_bad):
        return "BLOCKED"
    if critical_bad:
        return "FAIL"
    return "PASS"


def _invalid_root_record(root: Path, report_dir: Path) -> dict[str, object]:
    stdout_log, stderr_log = _log_paths(report_dir, "invalid_root")
    _write_text(stdout_log, "")
    _write_text(stderr_log, f"invalid root: {root}\n")
    return {
        "id": "invalid_root",
        "name": "Repository root",
        "critical": True,
        "status": "BLOCKED",
        "command": [],
        "exit_code": None,
        "duration_seconds": 0.0,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "failure_reason": f"invalid root: {root}",
    }


def _first_failing_log(records: Sequence[dict[str, object]]) -> str:
    for record in records:
        if record["critical"] and record["status"] in CRITICAL_FAILURE_STATUSES:
            return str(record["stderr_log"])
    for record in records:
        if record["status"] != "PASS":
            return str(record["stderr_log"])
    return ""


def _write_review_readme(report_dir: Path, report: dict[str, object]) -> None:
    report_path = Path(str(report["first_file_to_open"]))
    first_failing_log = str(report.get("first_failing_log", ""))
    lines = [
        "# PR preflight review bundle",
        "",
        f"Status: {report['status']}",
        f"Failure count: {report['failure_count']}",
        f"Report: {report_path}",
    ]
    if first_failing_log:
        lines.append(f"First failing stderr log: {first_failing_log}")
    else:
        lines.append("First failing stderr log: none")
    lines.extend(
        [
            "",
            "Open `preflight_report.json` first, then follow the per-check `stdout_log` and `stderr_log` fields for raw command output.",
            "This bundle is generated by `tools/ci/pr_preflight.py` and is local evidence, not release proof.",
            "",
        ]
    )
    (report_dir / "README_FOR_REVIEW.md").write_text("\n".join(lines), encoding="utf-8")


def run_preflight(*, root: Path, report_dir: Path, skip_install: bool, timeout_seconds: int = 300) -> tuple[int, dict[str, object]]:
    root = root.resolve()
    report_dir = report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "logs").mkdir(parents=True, exist_ok=True)

    started_at = _utc_now()
    started = time.monotonic()
    if not root.is_dir():
        records = [_invalid_root_record(root, report_dir)]
    else:
        records = _normalize_statuses(
            [
                run_check(check, root=root, report_dir=report_dir)
                for check in build_checks(root, skip_install=skip_install, timeout_seconds=timeout_seconds)
            ]
        )
    finished_at = _utc_now()
    summary = summarize(records)
    status = _final_status(records)
    if status not in FINAL_STATUSES:
        status = "BLOCKED"
    failure_count = sum(
        1
        for item in records
        if item["critical"] and item["status"] in CRITICAL_FAILURE_STATUSES
    )
    report_path = report_dir / "preflight_report.json"
    first_failing_log = _first_failing_log(records)
    next_action = f"Open the first failing check stderr log: {first_failing_log}" if failure_count else "No critical preflight failures recorded."
    report: dict[str, object] = {
        "schema_version": 1,
        "status": status,
        "root": str(root),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": round(time.monotonic() - started, 6),
        "checks": records,
        "summary": summary,
        "failure_count": failure_count,
        "next_action": next_action,
        "first_file_to_open": str(report_path),
        "first_failing_log": first_failing_log,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_review_readme(report_dir, report)
    return (0 if status == "PASS" else 1), report


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the structured GeoSync PR preflight gate.")
    parser.add_argument("--root", default=".", help="Repository root to run checks from.")
    parser.add_argument("--report-dir", default="artifacts/pr_preflight", help="Directory for report JSON and logs.")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip bootstrap and project dependency installation checks.")
    parser.add_argument("--json", action="store_true", help="Print the final report JSON to stdout.")
    parser.add_argument("--timeout-seconds", type=int, default=300, help="Per-check timeout in seconds.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        code, report = run_preflight(
            root=Path(args.root),
            report_dir=Path(args.report_dir),
            skip_install=args.skip_install,
            timeout_seconds=args.timeout_seconds,
        )
    except OSError as exc:
        print("PR_PREFLIGHT_STATUS=BLOCKED", file=sys.stderr)
        print(f"PR_PREFLIGHT_BLOCKER=failed to write preflight evidence: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"PR_PREFLIGHT_STATUS={report['status']}")
        print(f"PR_PREFLIGHT_FAILURE_COUNT={report['failure_count']}")
        print(f"FIRST_FILE_TO_OPEN={report['first_file_to_open']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
