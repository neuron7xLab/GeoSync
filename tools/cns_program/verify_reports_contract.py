#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allow `python tools/cns_program/<script>.py` from repo root
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.cns_program.cns_contract import (  # noqa: E402
    REPORTS_CONTRACT_RESULT,
    REQUIRED_REPORTS,
)


def verify_reports(
    required_reports: tuple[Path, ...] = REQUIRED_REPORTS,
) -> dict[str, Any]:
    errors: list[str] = []
    checked = 0
    for path in required_reports:
        if not path.exists():
            errors.append(f"report_missing:{path}")
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            errors.append(f"report_empty:{path}")
            continue
        if not text.startswith("#"):
            errors.append(f"report_heading_missing:{path}")
            continue
        checked += 1
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "valid": not errors,
        "errors": errors,
        "checked_reports": checked,
        "total_reports": len(required_reports),
    }


def main() -> int:
    payload = verify_reports()
    REPORTS_CONTRACT_RESULT.parent.mkdir(parents=True, exist_ok=True)
    REPORTS_CONTRACT_RESULT.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return 0 if bool(payload["valid"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
