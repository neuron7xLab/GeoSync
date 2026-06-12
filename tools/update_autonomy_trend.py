#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "autonomy_trend.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "entries": []}
    ledger = _read_object(path)
    if ledger.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("ledger schema_version mismatch")
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        raise ValueError("ledger entries must be a list")
    return ledger


def _entry(cycle_path: Path, cycle: dict[str, Any], sequence: int) -> dict[str, Any]:
    decision = cycle.get("decision")
    phases = cycle.get("phases")
    if not isinstance(decision, dict):
        raise ValueError("cycle decision must be an object")
    if not isinstance(phases, list):
        raise ValueError("cycle phases must be a list")
    vector = [str(item.get("status", "UNKNOWN")) for item in phases if isinstance(item, dict)]
    return {
        "sequence": sequence,
        "cycle_schema_version": str(cycle.get("schema_version", "")),
        "cycle_sha256": _sha256(cycle_path),
        "decision_state": str(decision.get("state", "UNKNOWN")),
        "status_vector": vector,
        "phase_count": len(phases),
    }


def build_ledger(cycle_path: Path, ledger_path: Path, limit: int) -> dict[str, Any]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    cycle = _read_object(cycle_path)
    ledger = _read_ledger(ledger_path)
    entries = list(ledger["entries"])
    entries.append(_entry(cycle_path, cycle, len(entries) + 1))
    entries = entries[-limit:]
    return {"schema_version": SCHEMA_VERSION, "entries": entries}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", default="artifacts/validation/autonomy_cycle.json")
    parser.add_argument("--ledger", default="artifacts/validation/autonomy_trend.json")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)
    cycle_path = Path(args.cycle)
    ledger_path = Path(args.ledger)
    payload = build_ledger(cycle_path, ledger_path, args.limit)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
