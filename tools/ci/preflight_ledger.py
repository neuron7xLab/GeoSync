#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Append a machine-readable inference ledger entry for PR preflight runs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_REPORT_KEYS = {
    "schema_version",
    "status",
    "root",
    "started_at",
    "finished_at",
    "duration_seconds",
    "checks",
    "summary",
    "failure_count",
    "next_action",
    "first_file_to_open",
}
REQUIRED_LEDGER_KEYS = {
    "schema_version",
    "event_type",
    "recorded_at",
    "run_id",
    "report_path",
    "report_schema_version",
    "status",
    "failure_count",
    "critical_failure_ids",
    "blocked_check_ids",
    "timeout_check_ids",
    "skipped_optional_ids",
    "next_action",
    "first_file_to_open",
}
ALLOWED_FINAL_STATUSES = {"PASS", "FAIL", "BLOCKED"}
ALLOWED_CHECK_STATUSES = {"PASS", "FAIL", "SKIPPED_OPTIONAL", "BLOCKED", "TIMEOUT"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_report(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(isinstance(report, dict), "preflight report must be a JSON object")
    missing = sorted(REQUIRED_REPORT_KEYS - set(report))
    _require(not missing, f"preflight report missing keys: {missing}")
    _require(report["status"] in ALLOWED_FINAL_STATUSES, "invalid preflight final status")
    _require(isinstance(report["checks"], list), "preflight checks must be a list")
    return report


def _ids_with_status(checks: list[dict[str, Any]], status: str) -> list[str]:
    return [str(check.get("id", "")) for check in checks if check.get("status") == status]


def build_ledger_entry(report: dict[str, Any], report_path: Path) -> dict[str, Any]:
    checks = report["checks"]
    for index, check in enumerate(checks):
        _require(isinstance(check, dict), f"check {index} must be an object")
        _require(check.get("status") in ALLOWED_CHECK_STATUSES, f"invalid check status at {index}")

    critical_failure_ids = [
        str(check.get("id", ""))
        for check in checks
        if check.get("critical") is True and check.get("status") != "PASS"
    ]
    run_id = f"{report['started_at']}|{report['finished_at']}|{report['status']}"
    entry = {
        "schema_version": 1,
        "event_type": "pr_preflight_run",
        "recorded_at": _utc_now(),
        "run_id": run_id,
        "report_path": report_path.as_posix(),
        "report_schema_version": report["schema_version"],
        "status": report["status"],
        "failure_count": report["failure_count"],
        "critical_failure_ids": critical_failure_ids,
        "blocked_check_ids": _ids_with_status(checks, "BLOCKED"),
        "timeout_check_ids": _ids_with_status(checks, "TIMEOUT"),
        "skipped_optional_ids": _ids_with_status(checks, "SKIPPED_OPTIONAL"),
        "next_action": report["next_action"],
        "first_file_to_open": report["first_file_to_open"],
    }
    validate_ledger_entry(entry)
    return entry


def validate_ledger_entry(entry: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_LEDGER_KEYS - set(entry))
    _require(not missing, f"ledger entry missing keys: {missing}")
    _require(entry["schema_version"] == 1, "ledger schema_version must be 1")
    _require(entry["event_type"] == "pr_preflight_run", "invalid ledger event_type")
    _require(entry["status"] in ALLOWED_FINAL_STATUSES, "invalid ledger status")
    _require(isinstance(entry["failure_count"], int), "failure_count must be int")
    for key in (
        "critical_failure_ids",
        "blocked_check_ids",
        "timeout_check_ids",
        "skipped_optional_ids",
    ):
        _require(isinstance(entry[key], list), f"{key} must be list")
        _require(all(isinstance(value, str) for value in entry[key]), f"{key} values must be str")
    _require(entry["first_file_to_open"].endswith("preflight_report.json"), "bad first_file_to_open")


def append_ledger(report_path: Path, ledger_path: Path) -> dict[str, Any]:
    report = load_report(report_path)
    entry = build_ledger_entry(report, report_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append PR preflight inference ledger entry.")
    parser.add_argument("--report", required=True, help="Path to preflight_report.json")
    parser.add_argument("--ledger", help="Ledger JSONL path. Defaults next to report.")
    parser.add_argument("--json", action="store_true", help="Print appended ledger entry as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report_path = Path(args.report).resolve()
    ledger_path = Path(args.ledger).resolve() if args.ledger else report_path.parent / "inference_ledger.jsonl"
    try:
        entry = append_ledger(report_path, ledger_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"PR_PREFLIGHT_LEDGER_ERROR={exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(entry, indent=2, sort_keys=True))
    else:
        print(f"PR_PREFLIGHT_LEDGER={ledger_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
