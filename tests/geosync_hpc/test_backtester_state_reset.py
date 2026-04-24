# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Regression tests for mutable-state isolation in BacktesterCAL."""

from __future__ import annotations

import errno
import pytest

from geosync_hpc.data import read_ticks_csv
from geosync_hpc.synthetic import generate_demo_ticks


def _cfg() -> dict:
    return {
        "seed": 7,
        "features": {"lookbacks": [5, 20], "fracdiff_d": 0.4, "ofi_window": 20},
        "regime": {"bins": [0.0, 0.5, 1.5, 5.0]},
        "quantile": {"low_q": 0.2, "high_q": 0.8},
        "conformal": {
            "alpha": 0.1,
            "decay": 0.005,
            "window": 400,
            "online_window": 200,
            "online_update": True,
        },
        "policy": {
            "max_pos": 1.0,
            "kelly_shrink": 0.2,
            "risk_gamma": 10.0,
            "cvar_alpha": 0.95,
            "cvar_window": 200,
        },
        "execution": {
            "fee_bps": 1.0,
            "impact_coeff": 0.8,
            "impact_model": "square_root",
            "queue_fill_p": 0.85,
        },
        "risk": {
            "intraday_dd_limit": 0.2,
            "loss_streak_cooldown": 6,
            "vola_spike_mult": 2.5,
            "exposure_cap": 1.0,
        },
        "target": {"horizon": 10},
    }


def test_backtester_repeated_runs_are_deterministic(tmp_path) -> None:
    pytest.importorskip("sklearn")
    from geosync_hpc.backtest import BacktesterCAL

    csv_path = generate_demo_ticks(tmp_path / "ticks.csv", n=1800, seed=11)
    df = read_ticks_csv(csv_path)
    feat_cols = ["ret1", "ret5", "ret20", "vol10", "vol50", "spread"]

    fit_end = 700
    cal_end = 1200

    bt = BacktesterCAL(_cfg())
    bt.fit_quantiles(df[feat_cols].iloc[:fit_end], df["y"].iloc[:fit_end])
    bt.calibrate_conformal(df[feat_cols].iloc[fit_end:cal_end], df["y"].iloc[fit_end:cal_end])

    eval_df = df.iloc[cal_end:]
    first = bt.run(eval_df, feat_cols=feat_cols, y_col="y")
    second = bt.run(eval_df, feat_cols=feat_cols, y_col="y")
    third = bt.run(eval_df, feat_cols=feat_cols, y_col="y")

    assert len(first) > 0
    assert first["eq"].to_list() == second["eq"].to_list()
    assert second["eq"].to_list() == third["eq"].to_list()


def test_trade_step_invariant_rejects_non_finite_values() -> None:
    from geosync_hpc.state import TradeStep
    from geosync_hpc.validation import ValidationService

    with pytest.raises(ValueError, match="Non-finite"):
        ValidationService.trade_step(
            TradeStep(
                mid=100.0,
                spread_frac=0.001,
                costs=float("nan"),
                target=0.2,
                cur_pos=0.0,
                fill_price=100.1,
                pnl=0.0,
            ),
            exposure_cap=1.0,
            max_position_jump_mult=2.0,
        )


def test_trade_step_invariant_rejects_negative_costs() -> None:
    from geosync_hpc.state import TradeStep
    from geosync_hpc.validation import ValidationService

    with pytest.raises(ValueError, match="Negative costs"):
        ValidationService.trade_step(
            TradeStep(
                mid=100.0,
                spread_frac=0.001,
                costs=-0.1,
                target=0.2,
                cur_pos=0.0,
                fill_price=100.05,
                pnl=0.0,
            ),
            exposure_cap=1.0,
            max_position_jump_mult=2.0,
        )


def test_calibrate_conformal_rejects_non_finite_inputs() -> None:
    pytest.importorskip("sklearn")
    import pandas as pd

    from geosync_hpc.backtest import BacktesterCAL

    bt = BacktesterCAL(_cfg())
    x = pd.DataFrame(
        {
            "ret1": [0.1],
            "ret5": [0.1],
            "ret20": [0.1],
            "vol10": [0.1],
            "vol50": [0.1],
            "spread": [0.1],
        }
    )
    y = pd.Series([float("nan")])
    with pytest.raises(ValueError, match="Non-finite"):
        bt.calibrate_conformal(x, y)


def test_session_full_determinism_and_safety(tmp_path) -> None:
    pytest.importorskip("sklearn")
    from geosync_hpc.backtest import BacktestSession

    csv_path = generate_demo_ticks(tmp_path / "ticks_full.csv", n=1500, seed=21)
    df = read_ticks_csv(csv_path)
    feat_cols = ["ret1", "ret5", "ret20", "vol10", "vol50", "spread"]

    bt = BacktestSession(_cfg())
    bt.fit_quantiles(df[feat_cols].iloc[:600], df["y"].iloc[:600])
    bt.calibrate_conformal(df[feat_cols].iloc[600:1000], df["y"].iloc[600:1000])

    eval_df = df.iloc[1000:]
    r1 = bt.run(eval_df, feat_cols=feat_cols, y_col="y")
    r2 = bt.run(eval_df, feat_cols=feat_cols, y_col="y")
    r3 = bt.run(eval_df, feat_cols=feat_cols, y_col="y")
    assert r1["eq"].to_list() == r2["eq"].to_list() == r3["eq"].to_list()

    state = bt.get_state()
    bt.set_state(state)
    assert bt.get_state() == state


def test_state_roundtrip_restores_guard_session_started_flag() -> None:
    pytest.importorskip("sklearn")
    from geosync_hpc.backtest import BacktestSession

    bt = BacktestSession(_cfg())
    bt.guard.start_session(0.0)
    snapshot = bt.get_state()
    bt.guard.reset()

    bt.set_state(snapshot)
    assert bt.guard._session_started is True


def test_mid_run_resume_produces_bitwise_identical_equity(tmp_path) -> None:
    pytest.importorskip("sklearn")
    from geosync_hpc.backtest import BacktestSession

    csv_path = generate_demo_ticks(tmp_path / "ticks_resume.csv", n=1600, seed=31)
    df = read_ticks_csv(csv_path)
    feat_cols = ["ret1", "ret5", "ret20", "vol10", "vol50", "spread"]

    fit_end = 650
    cal_end = 1100
    eval_df = df.iloc[cal_end:]

    full = BacktestSession(_cfg())
    full.fit_quantiles(df[feat_cols].iloc[:fit_end], df["y"].iloc[:fit_end])
    full.calibrate_conformal(df[feat_cols].iloc[fit_end:cal_end], df["y"].iloc[fit_end:cal_end])
    full_res = full.run(eval_df, feat_cols=feat_cols, y_col="y")

    midpoint = max(1, (len(eval_df) - 1) // 2)
    staged = BacktestSession(_cfg())
    staged.fit_quantiles(df[feat_cols].iloc[:fit_end], df["y"].iloc[:fit_end])
    staged.calibrate_conformal(df[feat_cols].iloc[fit_end:cal_end], df["y"].iloc[fit_end:cal_end])
    first_half = staged.run(eval_df, feat_cols=feat_cols, y_col="y", start_idx=0, end_idx=midpoint)
    checkpoint = staged.get_state()

    resumed = BacktestSession(_cfg())
    resumed.fit_quantiles(df[feat_cols].iloc[:fit_end], df["y"].iloc[:fit_end])
    resumed.calibrate_conformal(df[feat_cols].iloc[fit_end:cal_end], df["y"].iloc[fit_end:cal_end])
    resumed.set_state(checkpoint)
    second_half = resumed.run(
        eval_df,
        feat_cols=feat_cols,
        y_col="y",
        start_idx=midpoint,
        reset_state=False,
    )

    eq_joined = first_half["eq"].to_list() + second_half["eq"].to_list()
    assert eq_joined == full_res["eq"].to_list()


def test_run_save_csv_surfaces_permission_error(tmp_path, monkeypatch) -> None:
    pytest.importorskip("sklearn")
    import pandas as pd

    from geosync_hpc.backtest import BacktestSession

    csv_path = generate_demo_ticks(tmp_path / "ticks_io_perm.csv", n=1400, seed=17)
    df = read_ticks_csv(csv_path)
    feat_cols = ["ret1", "ret5", "ret20", "vol10", "vol50", "spread"]
    bt = BacktestSession(_cfg())
    bt.fit_quantiles(df[feat_cols].iloc[:550], df["y"].iloc[:550])
    bt.calibrate_conformal(df[feat_cols].iloc[550:950], df["y"].iloc[550:950])
    eval_df = df.iloc[950:]

    def _raise_eacces(self, *_args, **_kwargs):
        raise PermissionError(errno.EACCES, "permission denied")

    monkeypatch.setattr(pd.DataFrame, "to_csv", _raise_eacces, raising=True)
    with pytest.raises(PermissionError):
        bt.run(eval_df, feat_cols=feat_cols, y_col="y", save_csv=str(tmp_path / "out.csv"))


def test_run_save_csv_surfaces_disk_full_error(tmp_path, monkeypatch) -> None:
    pytest.importorskip("sklearn")
    import pandas as pd

    from geosync_hpc.backtest import BacktestSession

    csv_path = generate_demo_ticks(tmp_path / "ticks_io_enospc.csv", n=1400, seed=19)
    df = read_ticks_csv(csv_path)
    feat_cols = ["ret1", "ret5", "ret20", "vol10", "vol50", "spread"]
    bt = BacktestSession(_cfg())
    bt.fit_quantiles(df[feat_cols].iloc[:550], df["y"].iloc[:550])
    bt.calibrate_conformal(df[feat_cols].iloc[550:950], df["y"].iloc[550:950])
    eval_df = df.iloc[950:]

    def _raise_enospc(self, *_args, **_kwargs):
        raise OSError(errno.ENOSPC, "no space left on device")

    monkeypatch.setattr(pd.DataFrame, "to_csv", _raise_enospc, raising=True)
    with pytest.raises(OSError, match="no space left on device"):
        bt.run(eval_df, feat_cols=feat_cols, y_col="y", save_csv=str(tmp_path / "out.csv"))
