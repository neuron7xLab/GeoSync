from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.update_autonomy_trend import build_ledger


def _cycle(path: Path, state: str = "READY_FOR_NEXT_CYCLE") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "autonomy_cycle.v1",
                "phases": [
                    {"phase": "observe", "status": "OK"},
                    {"phase": "validate", "status": "OK"},
                    {"phase": "receipt", "status": "OK"},
                ],
                "decision": {"state": state, "status_vector": ["OK", "OK", "OK"]},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_build_ledger_records_cycle_digest(tmp_path: Path) -> None:
    cycle = tmp_path / "cycle.json"
    ledger_path = tmp_path / "trend.json"
    _cycle(cycle)
    ledger = build_ledger(cycle, ledger_path, 20)
    assert ledger["schema_version"] == "autonomy_trend.v1"
    entry = ledger["entries"][0]
    assert entry["sequence"] == 1
    assert entry["cycle_schema_version"] == "autonomy_cycle.v1"
    assert len(entry["cycle_sha256"]) == 64
    assert entry["decision_state"] == "READY_FOR_NEXT_CYCLE"
    assert entry["status_vector"] == ["OK", "OK", "OK"]
    assert entry["phase_count"] == 3


def test_build_ledger_respects_limit(tmp_path: Path) -> None:
    cycle = tmp_path / "cycle.json"
    ledger_path = tmp_path / "trend.json"
    _cycle(cycle, "OBSERVE_WITH_DEBT")
    ledger_path.write_text(
        json.dumps(
            {
                "schema_version": "autonomy_trend.v1",
                "entries": [
                    {
                        "sequence": 1,
                        "cycle_schema_version": "autonomy_cycle.v1",
                        "cycle_sha256": "0" * 64,
                        "decision_state": "READY_FOR_NEXT_CYCLE",
                        "status_vector": ["OK"],
                        "phase_count": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    ledger = build_ledger(cycle, ledger_path, 1)
    assert len(ledger["entries"]) == 1
    assert ledger["entries"][0]["decision_state"] == "OBSERVE_WITH_DEBT"


def test_build_ledger_rejects_non_positive_limit(tmp_path: Path) -> None:
    cycle = tmp_path / "cycle.json"
    _cycle(cycle)
    with pytest.raises(ValueError, match="limit"):
        build_ledger(cycle, tmp_path / "trend.json", 0)
