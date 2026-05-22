from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage-xml", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    coverage_xml = Path(args.coverage_xml)
    output = Path(args.output)

    payload = {
        "schema_version": "1.0",
        "coverage_xml": str(coverage_xml),
        "line_rate": 0.0,
        "branch_rate": 0.0,
        "modules": {"core": None, "backtest": None, "execution": None},
        "verdict": "blocked",
        "notes": "",
    }

    if not coverage_xml.exists():
        payload["notes"] = "coverage.xml missing; module-level coverage unavailable"
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return 0

    try:
        root = ET.parse(coverage_xml).getroot()
    except ET.ParseError:
        payload["notes"] = "coverage.xml unreadable/empty; module-level coverage unavailable"
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return 0
    payload["line_rate"] = float(root.attrib.get("line-rate", 0.0))
    payload["branch_rate"] = float(root.attrib.get("branch-rate", 0.0))
    payload["notes"] = (
        "Module-level coverage cannot be reliably derived from current XML mapping; "
        "modules set to null by policy."
    )
    payload["verdict"] = "downgraded" if payload["line_rate"] > 0 else "blocked"

    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
