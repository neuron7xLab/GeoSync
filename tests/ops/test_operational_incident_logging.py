"""T8 · operational_incidents.csv is append-only with canonical timing schema."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SHADOW = REPO / "results" / "cross_asset_kuramoto" / "shadow_validation"
INCIDENTS = SHADOW / "operational_incidents.csv"
LOGGER_SCRIPT = REPO / "scripts" / "log_shadow_operational_incident.py"


def _load_logger() -> ModuleType:
    spec = importlib.util.spec_from_file_location("shadow_incident_logger", LOGGER_SCRIPT)
    assert spec is not None, f"spec_from_file_location returned None for {LOGGER_SCRIPT}"
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_incident_schema_matches_contract() -> None:
    mod = _load_logger()
    expected = list(mod.FIELDNAMES)
    if not INCIDENTS.exists():
        pytest.skip("no incidents logged yet")
    df = pd.read_csv(INCIDENTS)
    assert list(df.columns) == expected


def test_incident_append_only(tmp_path: Path) -> None:
    mod = _load_logger()
    fake = tmp_path / "operational_incidents.csv"
    mod.append_incident(
        timestamp="2026-04-22T00:00:01Z",
        scheduled_time="2026-04-22T22:00:00Z",
        actual_run_time="2026-04-22T22:04:15Z",
        bar_date="2026-04-22",
        reason="unit_test_first_event",
        recovery_status="logged",
        path=fake,
    )
    a = pd.read_csv(fake)
    assert len(a) == 1
    mod.append_incident(
        timestamp="2026-04-23T00:00:02Z",
        scheduled_time="2026-04-23T22:00:00Z",
        actual_run_time="2026-04-24T07:11:00Z",
        bar_date="2026-04-23",
        reason="unit_test_second_event",
        recovery_status="logged",
        path=fake,
    )
    b = pd.read_csv(fake)
    assert len(b) == 2
    assert list(b["reason"]) == ["unit_test_first_event", "unit_test_second_event"]


def test_incident_records_scheduled_and_actual_run_time_separately(tmp_path: Path) -> None:
    mod = _load_logger()
    fake = tmp_path / "operational_incidents.csv"
    mod.append_incident(
        timestamp="2026-06-20T07:00:10Z",
        scheduled_time="2026-06-19T22:00:00Z",
        actual_run_time="2026-06-20T07:00:00Z",
        bar_date="2026-06-19",
        reason="offline_catch_up_gap",
        recovery_status="catch_up_logged_before_state_refresh",
        path=fake,
    )
    df = pd.read_csv(fake)
    row = df.iloc[0]
    assert row["timestamp"] == "2026-06-20T07:00:10Z"
    assert row["scheduled_time"] == "2026-06-19T22:00:00Z"
    assert row["actual_run_time"] == "2026-06-20T07:00:00Z"
    assert row["bar_date"] == "2026-06-19"
    assert row["scheduled_time"] != row["actual_run_time"]
    assert row["reason"] == "offline_catch_up_gap"


def test_legacy_incident_ledger_is_migrated_to_canonical_schema(tmp_path: Path) -> None:
    mod = _load_logger()
    fake = tmp_path / "operational_incidents.csv"
    fake.write_text(
        "incident_ts,incident_type,severity,affected_run_date,description,"
        "resolved_yes_no,resolution_ts,changed_artifacts_yes_no\n"
        "2026-04-21T22:47:24Z,manual_tick,LOW,2026-04-21,manual refresh,yes,"
        "2026-04-21T22:47:24Z,yes\n",
        encoding="utf-8",
    )
    mod.append_incident(
        timestamp="2026-06-20T07:00:10Z",
        scheduled_time="2026-06-19T22:00:00Z",
        actual_run_time="2026-06-20T07:00:00Z",
        bar_date="2026-06-19",
        reason="offline_catch_up_gap",
        recovery_status="catch_up_logged_before_state_refresh",
        path=fake,
    )
    df = pd.read_csv(fake)
    assert list(df.columns) == list(mod.FIELDNAMES)
    assert len(df) == 2
    assert df.iloc[0]["timestamp"] == "2026-04-21T22:47:24Z"
    assert df.iloc[0]["scheduled_time"] == "2026-04-21T22:47:24Z"
    assert df.iloc[0]["actual_run_time"] == "2026-04-21T22:47:24Z"
    assert df.iloc[0]["reason"].startswith("manual_tick:")
    assert df.iloc[1]["bar_date"] == "2026-06-19"
    assert df.iloc[1]["scheduled_time"] == "2026-06-19T22:00:00Z"
    assert df.iloc[1]["actual_run_time"] == "2026-06-20T07:00:00Z"
    assert df.iloc[1]["reason"] == "offline_catch_up_gap"
    assert df.iloc[1]["recovery_status"] == "catch_up_logged_before_state_refresh"


def test_canonical_v1_ledger_is_migrated_to_timing_schema(tmp_path: Path) -> None:
    mod = _load_logger()
    fake = tmp_path / "operational_incidents.csv"
    fake.write_text(
        "timestamp,bar_date,reason,recovery_status\n"
        "2026-06-19T22:05:00Z,2026-06-19,systemd_run_failed,logged\n",
        encoding="utf-8",
    )
    mod.append_incident(
        timestamp="2026-06-20T22:01:00Z",
        scheduled_time="2026-06-20T22:00:00Z",
        actual_run_time="2026-06-20T22:00:05Z",
        bar_date="2026-06-20",
        reason="paper_trader_skipped_missing",
        recovery_status="skipped_logged_continue_with_shadow_evaluation",
        path=fake,
    )
    df = pd.read_csv(fake)
    assert list(df.columns) == list(mod.FIELDNAMES)
    assert len(df) == 2
    assert df.iloc[0]["scheduled_time"] == "2026-06-19T22:05:00Z"
    assert df.iloc[0]["actual_run_time"] == "2026-06-19T22:05:00Z"
    assert df.iloc[1]["scheduled_time"] == "2026-06-20T22:00:00Z"


def test_renderer_runs_offline_and_without_input() -> None:
    rc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "render_cross_asset_kuramoto_shadow_report.py")],
        cwd=str(REPO),
        capture_output=True,
        timeout=30,
    )
    assert rc.returncode == 0
    text = (SHADOW / "SHADOW_SUMMARY.md").read_text(encoding="utf-8")
    assert "OBS-1" in text
    assert "DP5" in text or "forward-fill" in text
    assert "DP3" in text or "snapshot" in text
