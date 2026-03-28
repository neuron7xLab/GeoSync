#!/usr/bin/env python3
import json
import sys
from pathlib import Path

if len(sys.argv) < 2:
    raise SystemExit(
        "Usage: pip_audit_high_gate.py <pip-audit-json> [<pip-audit-json> ...]"
    )

high_or_critical = 0
for report_path in sys.argv[1:]:
    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    for dep in data.get("dependencies", []):
        for vuln in dep.get("vulns", []):
            severity = str(vuln.get("severity", "")).upper()
            if severity in {"HIGH", "CRITICAL"}:
                high_or_critical += 1

if high_or_critical:
    raise SystemExit(
        f"pip-audit found {high_or_critical} HIGH/CRITICAL vulnerabilities"
    )

print("pip-audit HIGH/CRITICAL gate passed.")
