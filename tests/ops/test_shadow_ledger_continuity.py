"""T10 · shadow ledger continuity is fail-closed.

The daily shadow rail validates a 90-bar business-day window. A systemd
catch-up can re-trigger execution after offline time, but it cannot prove
that the paper ledger contains every expected bar. The evaluator must
therefore reject internal business-day gaps before it updates the live
scoreboard.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = REPO / "scripts" / "evaluate_cross_asset_kuramoto_shadow.py"


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None, f"spec_from_file_location returned None for {path}"
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _write_equity_ledger(path: Path, dates: list[str]) -> None:
    rows = []
    equity = 1.0
    btc_equity = 1.0
    for day_n, date in enumerate(dates, start=1):
        net_ret = 0.001
        equity *= 1.0 + net_ret
        btc_equity *= 1.0005
        rows.append(
            {
                "date": date,
                "day_n": day_n,
                "net_ret": net_ret,
                "equity": equity,
                "turnover": 0.1,
                "cost": 0.00001,
                "btc_equity": btc_equity,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_complete_business_day_ledger_loads_without_incident(tmp_path: Path) -> None:
    mod = _load_module(EVAL_SCRIPT, "shadow_eval_continuity_ok")
    ledger = tmp_path / "equity.csv"
    incidents = tmp_path / "operational_incidents.csv"
    _write_equity_ledger(
        ledger,
        ["2026-06-15", "2026-06-16", "2026-06-17", "2026-06-18"],
    )

    live = mod._load_live_ledger(
        ledger,
        incidents_path=incidents,
        detected_at="2026-06-18T22:02:00Z",
    )

    assert len(live) == 4
    assert not incidents.exists(), "clean ledger must not emit operational noise"


def test_missing_business_day_logs_missing_bar_and_fails_closed(tmp_path: Path) -> None:
    mod = _load_module(EVAL_SCRIPT, "shadow_eval_continuity_gap")
    ledger = tmp_path / "equity.csv"
    incidents = tmp_path / "operational_incidents.csv"
    _write_equity_ledger(
        ledger,
        # Wednesday 2026-06-17 is intentionally absent.
        ["2026-06-15", "2026-06-16", "2026-06-18"],
    )

    with pytest.raises(mod.LedgerContinuityError) as exc_info:
        mod._load_live_ledger(
            ledger,
            incidents_path=incidents,
            detected_at="2026-06-18T22:02:00Z",
        )

    assert "missing_bar" in str(exc_info.value)
    assert "scheduled_date=2026-06-17" in str(exc_info.value)
    assert incidents.is_file(), "fail-closed path must leave an incident row"

    rows = pd.read_csv(incidents)
    assert list(rows.columns) == list(mod._incident_logger_module().FIELDNAMES)
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["reason"] == "missing_bar"
    assert row["scheduled_date"] == "2026-06-17"
    assert row["detected_at"] == "2026-06-18T22:02:00Z"
    assert int(row["gap_size_bars"]) == 1
    assert row["bar_date"] == "2026-06-17"
    assert row["scheduled_time"] == "2026-06-17T22:00:00Z"
    assert row["actual_run_time"] == "2026-06-18T22:02:00Z"
    assert "fail_closed_continuity_broken" in row["recovery_status"]
    assert "missing_dates=2026-06-17" in row["recovery_status"]
