# DRO-ARA v7 · Walk-Forward Calibration Report

## Header

- data_source: `data/askar/SPDR_S_P_500_ETF_GMT_0_NO-DST.parquet`
- date_range: 2017-02-16 → 2026-02-20
- n_daily_bars: 2263
- n_folds: 69
- grid_size: 7 × 11 = 77
- train_window=252, test_window=63, step=21, min_history=504

## Current vs Optimal

| param | current | optimal | delta |
|-------|---------|---------|-------|
| H_threshold (H_CRITICAL) | 0.45 | 0.30 | +0.15 |
| rs_threshold (RS_LONG_THRESH) | 0.33 | 0.10 | +0.23 |

## Top-5 Pairs (by mean OOS Sharpe)

| rank | H | rs | mean_sharpe | std_sharpe | worst_fold_sharpe | worst_dd | mean_trades | passes_filters |
|------|----|----|-------------|------------|--------------------|----------|-------------|----------------|
| 1 | 0.30 | 0.10 | 0.000 | 0.000 | 0.000 | -0.000 | 0.0 | False |
| 2 | 0.30 | 0.15 | 0.000 | 0.000 | 0.000 | -0.000 | 0.0 | False |
| 3 | 0.30 | 0.20 | 0.000 | 0.000 | 0.000 | -0.000 | 0.0 | False |
| 4 | 0.30 | 0.25 | 0.000 | 0.000 | 0.000 | -0.000 | 0.0 | False |
| 5 | 0.30 | 0.30 | 0.000 | 0.000 | 0.000 | -0.000 | 0.0 | False |

## Robustness

- Top pair (H=0.30, rs=0.10): mean_sharpe=0.000, std_sharpe=0.000, worst_fold_sharpe=0.000, worst_dd=-0.000, mean_trades=0.0, gate_on_folds=0

## Crisis Window (2022)

| H | rs | crisis_mean_sharpe | crisis_worst_dd | crisis_folds |
|----|----|--------------------|-----------------|--------------|
| 0.30 | 0.10 | 0.000 | -0.000 | 12 |
| 0.30 | 0.15 | 0.000 | -0.000 | 12 |
| 0.30 | 0.20 | 0.000 | -0.000 | 12 |
| 0.30 | 0.25 | 0.000 | -0.000 | 12 |
| 0.30 | 0.30 | 0.000 | -0.000 | 12 |

## Recommendation

**NO_SIGNAL / REJECT** — across the entire H × rs grid, no (H_threshold, rs_threshold) pair produced any gate-on fold with a non-zero Sharpe. The upstream ADF stationarity filter (INV-DRO3) dominates on this asset: stationary train windows are rare, and among those, train-derived rs rarely exceeds the grid's minimum rs_threshold. The binding constraint is therefore **not** H_CRITICAL or RS_LONG_THRESH but the engine's upstream regime filter. Operator action: DO NOT modify thresholds based on this run; escalate as an engine-design question (stationarity convention: ADF on prices vs log-returns) rather than a parameter tune.

Notes: Hurst computed exclusively on train windows (no leakage). Signal = combo_v1 (AMMComboStrategy) position × DRO-ARA gate. Gate logic = INV-DRO4 under grid-swept thresholds. Backtest = vectorized_backtest(fee=0.0005, signal shift=1 anti-lookahead).
