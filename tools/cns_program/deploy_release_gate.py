#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.cns_program.cns_contract import CONFIG_PATH, DEPLOY_GATE_RESULT, REQUIRED_STATES
from tools.cns_program.simple_yaml import load_yaml_file


def validate_protocol() -> dict:
    config = load_yaml_file(CONFIG_PATH)
    states = config.get("states", [])
    missing = [state for state in REQUIRED_STATES if state not in states]
    weights = config.get("metrics", {}).get("quality_weights", {})
    weight_sum = sum(float(weights.get(key, 0.0)) for key in weights)
    threshold = float(config.get("thresholds", {}).get("quality_score_min", 0.0))
    valid = not missing and abs(weight_sum - 1.0) < 1e-9 and 0.0 < threshold <= 1.0
    return {
        "valid": valid,
        "missing_states": missing,
        "weight_sum": weight_sum,
        "quality_score_min": threshold,
    }


def main() -> int:
    DEPLOY_GATE_RESULT.parent.mkdir(parents=True, exist_ok=True)
    protocol = validate_protocol()
    payload = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "deploy_ready": bool(protocol["valid"]),
        "protocol": protocol,
    }
    DEPLOY_GATE_RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["deploy_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
