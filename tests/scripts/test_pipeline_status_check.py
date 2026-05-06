# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""FIX-1 contract: pipeline_status_check writes a fresh daily file.

Falsification gate:
    after refresh(today), daily/<today>/pipeline_status.csv exists AND
    a second consecutive call overwrites it (mtime advances).
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from geosync.pipeline_status_check import (
    PIPELINE_STATUS_COLUMNS,
    STALENESS_LIMIT_BD,
    refresh,
)


@pytest.fixture()
def fake_paper_equity(tmp_path: Path) -> Path:
    """Build a synthetic paper-state ledger with a known last bar."""
    target = tmp_path / "equity.csv"
    last_bar = date(2026, 5, 5)
    rows = []
    for i in range(20):
        rows.append(
            {
                "day_n": i + 1,
                "date": (last_bar - timedelta(days=19 - i)).isoformat(),
                "regime": "high_sync",
                "R": 0.4,
                "net_ret": 0.0,
                "log_ret": 0.0,
                "abs_pos_change_sum": 0.0,
                "cost_charged": 0.0,
                "equity": 1.0,
                "btc_equity": 1.0,
                "BTC": 0.0,
                "ETH": 0.0,
                "SPY": 0.0,
                "TLT": 0.0,
                "GLD": 0.0,
            }
        )
    pd.DataFrame(rows).to_csv(target, index=False, lineterminator="\n")
    return target


def _read_status_row(path: Path) -> dict[str, object]:
    df = pd.read_csv(path)
    assert tuple(df.columns) == PIPELINE_STATUS_COLUMNS, (
        f"FIX-1 VIOLATED: pipeline_status.csv columns mismatch. "
        f"Expected {PIPELINE_STATUS_COLUMNS}, got {tuple(df.columns)}."
    )
    raw = df.iloc[0].to_dict()
    return {str(k): v for k, v in raw.items()}


def test_fresh_paper_state_yields_op_unsafe_false(tmp_path: Path, fake_paper_equity: Path) -> None:
    """Last bar 1 BD before run_date → not stale → op_unsafe=False."""
    daily_root = tmp_path / "daily"
    target = refresh(
        date(2026, 5, 6),
        daily_root=daily_root,
        equity_path=fake_paper_equity,
    )
    assert target.is_file(), f"FIX-1 VIOLATED: refresh did not produce {target}."
    row = _read_status_row(target)
    assert not bool(row["operationally_unsafe"]), (
        f"FIX-1 VIOLATED: fresh paper-state (last bar 2026-05-05, "
        f"run 2026-05-06, ~1 BD age) flagged op_unsafe. Row: {row}."
    )
    assert str(row["stale_assets_over_5bdays"]) in ("", "nan"), (
        f"FIX-1 VIOLATED: stale set should be empty for fresh state. "
        f"Got: {row['stale_assets_over_5bdays']!r}."
    )


def test_stale_paper_state_yields_op_unsafe_true(tmp_path: Path, fake_paper_equity: Path) -> None:
    """Last bar > 5 BD before run_date → stale → op_unsafe=True AND stale set populated."""
    daily_root = tmp_path / "daily"
    target = refresh(
        date(2026, 5, 30),  # 18 BD after fake last bar 2026-05-05
        daily_root=daily_root,
        equity_path=fake_paper_equity,
    )
    row = _read_status_row(target)
    assert bool(row["operationally_unsafe"]), (
        f"FIX-1 VIOLATED: stale paper-state "
        f"(last bar 2026-05-05, run 2026-05-30, ~18 BD age, "
        f"limit {STALENESS_LIMIT_BD}) NOT flagged op_unsafe. Row: {row}."
    )
    stale_str = str(row["stale_assets_over_5bdays"])
    assert stale_str not in ("", "nan"), (
        f"FIX-1 VIOLATED: op_unsafe=True but stale_assets_over_5bdays "
        f"is empty — flag must agree with the populated stale set. "
        f"Got: {stale_str!r}. Row: {row}."
    )
    assert (
        "BTC" in stale_str or "PORTFOLIO" in stale_str
    ), f"FIX-1 VIOLATED: stale set missing canonical asset/portfolio label. Got: {stale_str!r}."


def test_consecutive_refresh_advances_mtime(tmp_path: Path, fake_paper_equity: Path) -> None:
    """Two refresh() calls produce monotonic mtimes."""
    daily_root = tmp_path / "daily"
    target_1 = refresh(
        date(2026, 5, 6),
        daily_root=daily_root,
        equity_path=fake_paper_equity,
    )
    mtime_1 = target_1.stat().st_mtime
    time.sleep(1.1)
    target_2 = refresh(
        date(2026, 5, 6),
        daily_root=daily_root,
        equity_path=fake_paper_equity,
    )
    assert (
        target_1 == target_2
    ), f"Same run_date should target the same path; got {target_1} vs {target_2}."
    mtime_2 = target_2.stat().st_mtime
    assert mtime_2 > mtime_1, (
        f"FIX-1 VIOLATED: mtime did not advance across consecutive "
        f"refresh() calls. mtime_1={mtime_1}, mtime_2={mtime_2}. "
        f"Self-refresh is supposed to overwrite; the file is now stuck."
    )


def test_missing_paper_state_raises(tmp_path: Path) -> None:
    """Absent paper-state ledger → FileNotFoundError, NOT silent pass."""
    with pytest.raises(FileNotFoundError):
        refresh(
            date(2026, 5, 6),
            daily_root=tmp_path / "daily",
            equity_path=tmp_path / "nonexistent.csv",
        )
