# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Regression tests for mutable-state isolation in BacktesterCAL."""

from __future__ import annotations

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


def test_backtester_step_invariant_rejects_non_finite_values() -> None:
    pytest.importorskip("sklearn")
    from geosync_hpc.backtest import BacktesterCAL, TradeStep

    bt = BacktesterCAL(_cfg())
    with pytest.raises(ValueError, match="Non-finite runtime values"):
        bt._assert_step_invariants(
            TradeStep(
                mid=100.0,
                spread_frac=0.001,
                costs=float("nan"),
                target=0.2,
                cur_pos=0.0,
                fill_price=100.1,
                pnl=0.0,
            )
        )


def test_backtester_step_invariant_rejects_negative_costs() -> None:
    pytest.importorskip("sklearn")
    from geosync_hpc.backtest import BacktesterCAL, TradeStep

    bt = BacktesterCAL(_cfg())
    with pytest.raises(ValueError, match="Negative costs"):
        bt._assert_step_invariants(
            TradeStep(
                mid=100.0,
                spread_frac=0.001,
                costs=-0.1,
                target=0.2,
                cur_pos=0.0,
                fill_price=100.05,
                pnl=0.0,
            )
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
