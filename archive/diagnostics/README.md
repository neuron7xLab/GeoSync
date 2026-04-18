# archive/diagnostics

Historical derivation scripts from PR #237 diagnostic stack.
Retained for evidence-replay; not part of the production spine.

## Why archived
Composition via `scripts/run_l2_killtest.py` and `scripts/run_l2_pnl.py`
flags covers the orchestration these scripts performed. They remain here
verbatim for forensic reproducibility against merged JSON evidence in
`results/L2_*.json`.

## Replay legacy output
Checkout the PR #237 merged head, run the script, diff against tracked
JSON. Example:

    git checkout <PR-237-SHA> -- scripts/
    python scripts/l2_regime_oos.py --data-dir data/binance_l2_perp
    diff <new_output> results/L2_REGIME_OOS.json

## Contents — orchestrator scripts (5)
- l2_killtest_recursive.py     — bisection tree + K=8 cyclic
- l2_regime_analysis.py        — per-block regime features
- l2_regime_conditional.py     — in-sample regime-conditional IC
- l2_regime_oos.py             — 50/50 train/test OOS (GATE 1 source)
- l2_regime_cross_session.py   — cross-session scaffold

## Contents — historical derivation scripts (5)
- l2_walk_forward.py                     — 56 rolling windows, regime ρ=+0.352
- l2_regime_walkforward_calibration.py   — rolling cal+eval, ≥2h-cal requirement
- l2_per_symbol_ic.py                    — per-symbol decomposition, 10/10 positive
- l2_horizon_sweep.py                    — characteristic τ=180s, inverts past 900s
- l2_rv_decile_analysis.py               — U-shape finding
