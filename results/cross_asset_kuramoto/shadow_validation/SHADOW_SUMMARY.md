# Cross-Asset Kuramoto · Shadow Summary
Deterministic snapshot from the live scoreboard. Numbers and
gate decisions only; no marketing language.

## Current live bar count

- Live bars completed: **13**
- Spike paper-trader start date: 2026-04-11 (day-90 gate ≈ 2026-07-10)

## Operational health

- Predictive envelope built: **True**
- Operational incidents logged: **1**
- Operationally unsafe (latest): **True**
- Any invariant fail: **False**

## Live metrics

| metric | value |
|---|---:|
| cumulative_net_return | -0.0255 |
| annualized_return_live | -0.5011 |
| annualized_vol_live | 0.0965 |
| sharpe_live | -5.1957 |
| max_dd_live | 0.0354 |
| hit_rate_live | 0.1538 |
| turnover_ann_live | 2.6385 |
| cost_drag_bps_live | 26.3900 |

## Benchmark comparison

| metric | value |
|---|---:|
| benchmark_cum_return | 0.0722 |
| benchmark_sharpe_live | 4.2197 |
| relative_return_vs_benchmark | -0.0977 |

## Envelope position

- Quantile band (live vs historical OOS block-bootstrap): **p05_p25**
- Envelope source: demo-ready OOS integrated log returns
  (`results/cross_asset_kuramoto/demo/equity_curve.csv`); seed and
  block length locked in `DRIFT_NOTE.md`.

## Cost sensitivity (demo-baseline, recorded)

Baseline OOS cost drag at lock: 231 bps / year (23.1 % of gross,
see `COST_MODEL.md`). Live cost drag above is reported with the
paper-trader's own per-bar cost slot, cumulated over live bars.

## Known caveats carried from demo-ready stage

- OBS-1 `scipy.signal.hilbert` non-causal; preserved
  (`INTEGRATION_NOTES.md`).
- DP5 forward-fill(limit=3) material (ΔSharpe 0.22); preserved
  (`PIPELINE_AUDIT.md`).
- DP3 data snapshot age >5 bdays vs current clock; this is
  expected for the spike snapshot and **is** the source of the
  current `OPERATIONALLY_UNSAFE` label.
- Fold 3 (2022) Sharpe −1.15 historically; preserved.

## Current recommendation

- **status_label:** `OPERATIONALLY_UNSAFE`
- **gate_decision:** `ESCALATE_REVIEW`

Claude Code does not authorize capital deployment. The gate is
advisory and computed from `ACCEPTANCE_GATES.md`. At 90 live bars
the truth gate fires exactly one of `DEPLOYMENT_CANDIDATE_PENDING_OWNER`,
`CONTINUE_SHADOW`, or `NO_DEPLOY`.
