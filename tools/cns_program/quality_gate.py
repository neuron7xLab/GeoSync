#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tools.cns_program.cns_contract import (
    DEPLOY_GATE_RESULT,
    MANIFEST_VERIFICATION_RESULT,
    QUALITY_GATE_RESULT,
    REPORTS_CONTRACT_RESULT,
)


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def main() -> int:
    deploy = load(DEPLOY_GATE_RESULT)
    manifest = load(MANIFEST_VERIFICATION_RESULT)
    reports = load(REPORTS_CONTRACT_RESULT)

    protocol = deploy.get("protocol", {})
    if not isinstance(protocol, dict):
        protocol = {}

    protocol_valid = bool(protocol.get("valid"))
    manifest_valid = bool(manifest.get("valid"))
    reports_valid = bool(reports.get("valid"))
    deterministic_artifacts = (
        manifest.get("checked_artifacts") == manifest.get("total_artifacts")
        and int(manifest.get("total_artifacts", 0)) > 0
    )
    metrics = [
        {"name": "protocol_valid", "value": 1.0 if protocol_valid else 0.0, "weight": 0.35},
        {"name": "manifest_valid", "value": 1.0 if manifest_valid else 0.0, "weight": 0.30},
        {"name": "reports_valid", "value": 1.0 if reports_valid else 0.0, "weight": 0.20},
        {
            "name": "deterministic_artifacts",
            "value": 1.0 if deterministic_artifacts else 0.0,
            "weight": 0.15,
        },
    ]
    score = sum(float(item["value"]) * float(item["weight"]) for item in metrics)
    passed = score >= 0.80 and all(item["value"] == 1.0 for item in metrics)
    payload = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "quality_score": score,
        "threshold": 0.80,
        "passed": passed,
        "metrics": metrics,
    }
    QUALITY_GATE_RESULT.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
