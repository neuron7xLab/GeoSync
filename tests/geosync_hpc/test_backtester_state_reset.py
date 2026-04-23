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

    assert len(first) > 0
    assert first["eq"].to_list() == second["eq"].to_list()
