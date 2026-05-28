#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import UTC, datetime

from tools.cns_program.cns_contract import REPORTS_CONTRACT_RESULT, REQUIRED_REPORTS


def main() -> int:
    errors: list[str] = []
    checked = 0
    for path in REQUIRED_REPORTS:
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
    payload = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "valid": not errors,
        "errors": errors,
        "checked_reports": checked,
        "total_reports": len(REQUIRED_REPORTS),
    }
    REPORTS_CONTRACT_RESULT.parent.mkdir(parents=True, exist_ok=True)
    REPORTS_CONTRACT_RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
