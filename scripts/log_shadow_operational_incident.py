"""Canonical append-only operational incident logger for GeoSync shadow runs.

The systemd wrapper uses this tiny script because shell traps must be able to
record failed/skipped scheduled runs even when the Python shadow pipeline dies
before its own fail-closed ledger path is reached. No network. No interactive
input. One row per incident.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHADOW_DIR = REPO / "results" / "cross_asset_kuramoto" / "shadow_validation"
INCIDENTS = SHADOW_DIR / "operational_incidents.csv"
FIELDNAMES = ("timestamp", "bar_date", "reason", "recovery_status")
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


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _normalise_existing_rows(path: Path) -> list[dict[str, str]]:
    """Read current ledger and map legacy rows into the canonical schema."""
    if not path.exists() or path.stat().st_size == 0:
        return []

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames == list(FIELDNAMES):
            return [{k: row.get(k, "") for k in FIELDNAMES} for row in reader]
        if reader.fieldnames == list(LEGACY_FIELDNAMES):
            rows: list[dict[str, str]] = []
            for row in reader:
                reason = row.get("incident_type", "legacy_incident")
                desc = row.get("description", "")
                if desc:
                    reason = f"{reason}: {desc}"
                resolved = row.get("resolved_yes_no", "")
                changed = row.get("changed_artifacts_yes_no", "")
                recovery_status = f"legacy_resolved={resolved}; changed_artifacts={changed}".strip()
                rows.append(
                    {
                        "timestamp": row.get("incident_ts", ""),
                        "bar_date": row.get("affected_run_date", ""),
                        "reason": reason,
                        "recovery_status": recovery_status,
                    }
                )
            return rows
        raise ValueError(f"Unsupported operational_incidents.csv schema: {reader.fieldnames}")


def append_incident(
    *,
    reason: str,
    recovery_status: str,
    timestamp: str | None = None,
    bar_date: str | None = None,
    path: Path = INCIDENTS,
) -> None:
    """Append a canonical operational incident row.

    Existing legacy rows are migrated in-place on first write so the ledger has
    one stable contract: timestamp, bar_date, reason, recovery_status.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _normalise_existing_rows(path)
    rows.append(
        {
            "timestamp": timestamp or now_utc(),
            "bar_date": bar_date or today_utc(),
            "reason": reason,
            "recovery_status": recovery_status,
        }
    )
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Append a GeoSync shadow operational incident")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--bar-date", default=None)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--recovery-status", required=True)
    parser.add_argument("--incidents-path", type=Path, default=INCIDENTS)
    args = parser.parse_args(argv)

    append_incident(
        timestamp=args.timestamp,
        bar_date=args.bar_date,
        reason=args.reason,
        recovery_status=args.recovery_status,
        path=args.incidents_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
