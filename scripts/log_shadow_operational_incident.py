from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHADOW_DIR = REPO / "results" / "cross_asset_kuramoto" / "shadow_validation"
INCIDENTS = SHADOW_DIR / "operational_incidents.csv"
FIELDNAMES = (
    "timestamp",
    "scheduled_time",
    "actual_run_time",
    "bar_date",
    "reason",
    "recovery_status",
    "scheduled_date",
    "detected_at",
    "gap_size_bars",
)
TIMING_FIELDNAMES = (
    "timestamp",
    "scheduled_time",
    "actual_run_time",
    "bar_date",
    "reason",
    "recovery_status",
)
LEGACY_FIELDNAMES = (
    "incident_ts",
    "incident_type",
    "severity",
    "affected_run_date",
    "description",
    "resolved_yes_no",
    "resolution_ts",
    "changed_artifacts_yes_no",
)
CANONICAL_V1_FIELDNAMES = ("timestamp", "bar_date", "reason", "recovery_status")


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _date_from_time(value: str) -> str:
    if value and len(value) >= 10:
        return value[:10]
    return ""


def _canonical_row(
    *,
    timestamp: str,
    scheduled_time: str,
    actual_run_time: str,
    bar_date: str,
    reason: str,
    recovery_status: str,
    scheduled_date: str = "",
    detected_at: str = "",
    gap_size_bars: str = "",
) -> dict[str, str]:
    scheduled = scheduled_date or bar_date or _date_from_time(scheduled_time)
    detected = detected_at or timestamp
    return {
        "timestamp": timestamp,
        "scheduled_time": scheduled_time,
        "actual_run_time": actual_run_time,
        "bar_date": bar_date,
        "reason": reason,
        "recovery_status": recovery_status,
        "scheduled_date": scheduled,
        "detected_at": detected,
        "gap_size_bars": gap_size_bars,
    }


def _normalise_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames == list(FIELDNAMES):
            return [{k: row.get(k, "") for k in FIELDNAMES} for row in reader]
        if reader.fieldnames == list(TIMING_FIELDNAMES):
            return [
                _canonical_row(
                    timestamp=row.get("timestamp", ""),
                    scheduled_time=row.get("scheduled_time", ""),
                    actual_run_time=row.get("actual_run_time", ""),
                    bar_date=row.get("bar_date", ""),
                    reason=row.get("reason", ""),
                    recovery_status=row.get("recovery_status", ""),
                )
                for row in reader
            ]
        if reader.fieldnames == list(CANONICAL_V1_FIELDNAMES):
            rows: list[dict[str, str]] = []
            for row in reader:
                ts = row.get("timestamp", "")
                rows.append(
                    _canonical_row(
                        timestamp=ts,
                        scheduled_time=ts,
                        actual_run_time=ts,
                        bar_date=row.get("bar_date", ""),
                        reason=row.get("reason", ""),
                        recovery_status=row.get("recovery_status", ""),
                    )
                )
            return rows
        if reader.fieldnames == list(LEGACY_FIELDNAMES):
            rows = []
            for row in reader:
                ts = row.get("incident_ts", "")
                reason = row.get("incident_type", "legacy_incident")
                desc = row.get("description", "")
                if desc:
                    reason = f"{reason}: {desc}"
                resolved = row.get("resolved_yes_no", "")
                changed = row.get("changed_artifacts_yes_no", "")
                recovery_status = f"legacy_resolved={resolved}; changed_artifacts={changed}".strip()
                rows.append(
                    _canonical_row(
                        timestamp=ts,
                        scheduled_time=ts,
                        actual_run_time=ts,
                        bar_date=row.get("affected_run_date", ""),
                        reason=reason,
                        recovery_status=recovery_status,
                    )
                )
            return rows
        raise ValueError(f"Unsupported operational_incidents.csv schema: {reader.fieldnames}")


def append_incident(
    *,
    reason: str,
    recovery_status: str,
    timestamp: str | None = None,
    scheduled_time: str | None = None,
    actual_run_time: str | None = None,
    bar_date: str | None = None,
    scheduled_date: str | None = None,
    detected_at: str | None = None,
    gap_size_bars: str | int | None = None,
    path: Path = INCIDENTS,
) -> None:
    incident_ts = timestamp or now_utc()
    actual = actual_run_time or incident_ts
    scheduled = scheduled_time or actual
    bar = bar_date or scheduled_date or today_utc()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _normalise_existing_rows(path)
    rows.append(
        _canonical_row(
            timestamp=incident_ts,
            scheduled_time=scheduled,
            actual_run_time=actual,
            bar_date=bar,
            reason=reason,
            recovery_status=recovery_status,
            scheduled_date=scheduled_date or bar,
            detected_at=detected_at or incident_ts,
            gap_size_bars="" if gap_size_bars is None else str(gap_size_bars),
        )
    )
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Append a GeoSync shadow operational incident")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--scheduled-time", default=None)
    parser.add_argument("--actual-run-time", default=None)
    parser.add_argument("--bar-date", default=None)
    parser.add_argument("--scheduled-date", default=None)
    parser.add_argument("--detected-at", default=None)
    parser.add_argument("--gap-size-bars", default=None)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--recovery-status", required=True)
    parser.add_argument("--incidents-path", type=Path, default=INCIDENTS)
    args = parser.parse_args(argv)

    append_incident(
        timestamp=args.timestamp,
        scheduled_time=args.scheduled_time,
        actual_run_time=args.actual_run_time,
        bar_date=args.bar_date,
        scheduled_date=args.scheduled_date,
        detected_at=args.detected_at,
        gap_size_bars=args.gap_size_bars,
        reason=args.reason,
        recovery_status=args.recovery_status,
        path=args.incidents_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
