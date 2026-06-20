"""T8 · operational_incidents.csv is append-only with canonical schema."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SHADOW = REPO / "results" / "cross_asset_kuramoto" / "shadow_validation"
INCIDENTS = SHADOW / "operational_incidents.csv"
LOGGER_SCRIPT = REPO / "scripts" / "log_shadow_operational_incident.py"


def _load_logger():
    spec = importlib.util.spec_from_file_location("shadow_incident_logger", LOGGER_SCRIPT)
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
    """Appending an incident must never shrink the existing ledger."""
    mod = _load_logger()
    fake = tmp_path / "operational_incidents.csv"
    mod.append_incident(
        timestamp="2026-04-22T00:00:00Z",
        bar_date="2026-04-22",
        reason="unit_test_first_event",
        recovery_status="logged",
        path=fake,
    )
    a = pd.read_csv(fake)
    assert len(a) == 1
    mod.append_incident(
        timestamp="2026-04-22T00:00:02Z",
        bar_date="2026-04-22",
        reason="unit_test_second_event",
        recovery_status="logged",
        path=fake,
    )
    b = pd.read_csv(fake)
    assert len(b) == 2
    assert list(b["reason"]) == ["unit_test_first_event", "unit_test_second_event"]


def test_legacy_incident_ledger_is_migrated_to_canonical_schema(tmp_path: Path) -> None:
    """First canonical append rewrites legacy rows without losing history."""
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
        timestamp="2026-06-20T22:00:00Z",
        bar_date="2026-06-20",
        reason="offline_catch_up_gap last_eval_utc=2026-06-13T22:00:00Z gap_hours=168",
        recovery_status="catch_up_logged_before_state_refresh",
        path=fake,
    )

    df = pd.read_csv(fake)
    assert list(df.columns) == list(mod.FIELDNAMES)
    assert len(df) == 2
    assert df.iloc[0]["timestamp"] == "2026-04-21T22:47:24Z"
    assert df.iloc[0]["reason"].startswith("manual_tick:")
    assert df.iloc[1]["bar_date"] == "2026-06-20"
    assert "offline_catch_up_gap" in df.iloc[1]["reason"]
    assert df.iloc[1]["recovery_status"] == "catch_up_logged_before_state_refresh"


def test_renderer_runs_offline_and_without_input() -> None:
    """T9 · renderer produces SHADOW_SUMMARY.md and surfaces caveats."""
    import subprocess

    rc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "render_cross_asset_kuramoto_shadow_report.py")],
        cwd=str(REPO),
        capture_output=True,
        timeout=30,
    )
    assert rc.returncode == 0
    summary = SHADOW / "SHADOW_SUMMARY.md"
    text = summary.read_text()
    # T10: known caveats must be surfaced
    assert "OBS-1" in text
    assert "DP5" in text or "forward-fill" in text
    assert "DP3" in text or "snapshot" in text
    # No optimisation language
    for bad in ("optimize", "tune", "grid", "sweep", "edge", "alpha"):
        assert bad.lower() not in text.lower(), f"forbidden term {bad!r} in summary"
