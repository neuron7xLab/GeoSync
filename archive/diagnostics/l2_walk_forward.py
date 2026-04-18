#!/usr/bin/env python3
"""Rolling walk-forward IC trajectory with regime features.

Slide a 40-minute window across the substrate with 5-minute step.
For each window: compute block-level IC + same regime features as
l2_regime_analysis.py (rv, corr, disp, trend, κ moments). Then:

    1. Report the IC trajectory shape + stability.
    2. Compute Spearman rank correlation of IC against every feature
       at the rolling-window resolution (many more points than the
       K=8 non-overlapping blocks).
    3. Bin rolling windows by the top-correlated feature, report IC
       per bin to identify a regime discriminator threshold.

Output: results/L2_WALK_FORWARD.json + printed summary.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from research.microstructure.killtest import (
    FeatureFrame,
    build_feature_frame,
    cross_sectional_ricci_signal,
    run_killtest,
    slice_features,
)
from research.microstructure.killtest import (
    _load_parquets as load_parquets,
)
from research.microstructure.l2_schema import DEFAULT_SYMBOLS

_WINDOW_SEC = 40 * 60
_STEP_SEC = 5 * 60


@dataclass
class WFRow:
    start: int
    end: int
    n: int
    ic_signal: float
    residual_ic: float
    residual_p: float
    perm_p: float
    rv_mean: float
    corr_mean: float
    disp_mean: float
    trend_signed: float
    trend_abs: float
    ricci_mean: float
    ricci_std: float


def _block_features(feat: FeatureFrame, start: int, end: int) -> dict[str, float]:
    sub = slice_features(feat, start, end)
    log_mid = np.log(sub.mid)
    ret = np.vstack([np.zeros((1, sub.n_symbols)), np.diff(log_mid, axis=0)])
    rv = pd.DataFrame(ret).rolling(window=60, min_periods=30).std().to_numpy()
    c = np.nan_to_num(np.corrcoef(ret.T), nan=0.0)
    mask = ~np.eye(sub.n_symbols, dtype=bool)
    return {
        "rv_mean": float(np.nanmean(rv)),
        "corr_mean": float(c[mask].mean()),
        "disp_mean": float(np.nanmean(ret.std(axis=1))),
        "trend_signed": float((log_mid[-1] - log_mid[0]).mean()),
        "trend_abs": float(abs((log_mid[-1] - log_mid[0]).mean())),
    }


def main() -> int:
    data_dir = Path("data/binance_l2_perp")
    frames = load_parquets(data_dir, DEFAULT_SYMBOLS)
    features = build_feature_frame(frames, DEFAULT_SYMBOLS)
    print(f"substrate: n_rows={features.n_rows}  n_symbols={features.n_symbols}")

    n = features.n_rows
    rows: list[WFRow] = []
    start = 0
    while start + _WINDOW_SEC <= n:
        end = start + _WINDOW_SEC
        sub = slice_features(features, start, end)
        v = run_killtest(sub)
        feats = _block_features(features, start, end)
        ricci = cross_sectional_ricci_signal(sub.ofi)
        rows.append(
            WFRow(
                start=start,
                end=end,
                n=int(sub.n_rows),
                ic_signal=float(v.ic_signal),
                residual_ic=float(v.residual_ic),
                residual_p=float(v.residual_ic_pvalue),
                perm_p=float(v.null_test_pvalues["permutation_shuffle"]),
                **feats,
                ricci_mean=float(np.nanmean(ricci)),
                ricci_std=float(np.nanstd(ricci)),
            )
        )
        start += _STEP_SEC

    print(f"rolling windows: {len(rows)}")
    print(
        f"IC summary: mean={np.mean([r.ic_signal for r in rows]):+.4f}  "
        f"median={np.median([r.ic_signal for r in rows]):+.4f}  "
        f"min={min(r.ic_signal for r in rows):+.4f}  "
        f"max={max(r.ic_signal for r in rows):+.4f}  "
        f"frac_positive={sum(1 for r in rows if r.ic_signal > 0) / len(rows):.2%}  "
        f"frac_IC_gt_0.03={sum(1 for r in rows if r.ic_signal > 0.03) / len(rows):.2%}"
    )
    print()

    df = pd.DataFrame([asdict(r) for r in rows])
    feature_cols = [
        "rv_mean",
        "corr_mean",
        "disp_mean",
        "trend_signed",
        "trend_abs",
        "ricci_mean",
        "ricci_std",
    ]
    print("=== Spearman ρ: rolling IC_signal vs regime feature ===")
    corr_res: dict[str, dict[str, float]] = {}
    for col in feature_cols:
        rho, p = spearmanr(df["ic_signal"], df[col])
        corr_res[col] = {"rho": float(rho), "p": float(p)}
        marker = " ***" if p < 0.01 else (" *" if p < 0.05 else "")
        print(f"  {col:<14}  ρ={rho:+.3f}  p={p:.4f}{marker}")
    print()

    # Quartile analysis on the most-correlated feature
    best_feat = max(corr_res, key=lambda k: abs(corr_res[k]["rho"]))
    print(f"=== Quartile bins by {best_feat} ===")
    q = pd.qcut(df[best_feat], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    grouped = df.groupby(q, observed=True)["ic_signal"].agg(["mean", "median", "count"])
    print(grouped.to_string())
    print()

    out = {
        "window_sec": _WINDOW_SEC,
        "step_sec": _STEP_SEC,
        "n_windows": len(rows),
        "rows": [asdict(r) for r in rows],
        "feature_correlations": corr_res,
        "best_feature": best_feat,
        "quartile_bins": {
            str(k): {m: float(v) for m, v in grouped.loc[k].items()} for k in grouped.index
        },
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/L2_WALK_FORWARD.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print("wrote results/L2_WALK_FORWARD.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
