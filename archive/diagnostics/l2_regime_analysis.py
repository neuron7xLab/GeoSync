#!/usr/bin/env python3
"""Characterize the regime structure of the collected L2 substrate.

For each of K adjacent disjoint time blocks, measure both:

    (a) OUTCOME variables
        * IC_signal of Ricci κ_min vs 3-min forward mid-return
        * residual IC (orthogonal to baselines)
        * permutation p-value

    (b) PREDICTORS  (regime features, independent of Ricci)
        * realized volatility (pooled 60s-rolling, mean per block)
        * cross-asset correlation (mean off-diagonal of corr(mid_returns))
        * cross-asset dispersion (std across symbols of per-sec returns)
        * signed trend (mean log-return per second, full block)
        * |trend| magnitude
        * spread level (pooled median bps)
        * Ricci κ_min mean + std within block (signal-internal)

Regress OUTCOME on PREDICTORS → identify discriminator that separates
PROCEED blocks from KILL blocks. Emit JSON per-block + correlation matrix.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from research.microstructure.killtest import (
    _load_parquets as load_parquets,
)
from research.microstructure.killtest import (
    build_feature_frame,
    cross_sectional_ricci_signal,
    run_killtest,
    slice_features,
)
from research.microstructure.l2_schema import DEFAULT_SYMBOLS


@dataclass
class BlockRow:
    block: int
    start: int
    end: int
    n: int
    ic_signal: float
    residual_ic: float
    residual_p: float
    perm_p: float
    verdict: str
    rv_mean: float
    corr_mean: float
    disp_mean: float
    trend_signed: float
    trend_abs: float
    ricci_mean: float
    ricci_std: float


def _features_per_block(feat: object, start: int, end: int) -> dict[str, float]:
    from research.microstructure.killtest import FeatureFrame  # noqa: PLC0415

    assert isinstance(feat, FeatureFrame)
    sub = slice_features(feat, start, end)

    # 1-second log returns per symbol (first row treated as 0 via hstack zero)
    log_mid = np.log(sub.mid)
    ret = np.vstack([np.zeros((1, sub.n_symbols)), np.diff(log_mid, axis=0)])

    # Realized vol: rolling 60s std per symbol, pooled mean across time+symbols
    rv = pd.DataFrame(ret).rolling(window=60, min_periods=30).std().to_numpy()
    rv_mean = float(np.nanmean(rv))

    # Cross-asset mean correlation (full block corr matrix, off-diagonal mean)
    if sub.n_symbols >= 2 and sub.n_rows > 30:
        c = np.corrcoef(ret.T)
        c = np.nan_to_num(c, nan=0.0)
        mask = ~np.eye(sub.n_symbols, dtype=bool)
        corr_mean = float(c[mask].mean())
    else:
        corr_mean = float("nan")

    # Dispersion: per-tick std across symbols, averaged
    disp = ret.std(axis=1)
    disp_mean = float(np.nanmean(disp))

    # Signed trend: sum of log_mid diffs per symbol, averaged
    trend = float((log_mid[-1] - log_mid[0]).mean())
    return {
        "rv_mean": rv_mean,
        "corr_mean": corr_mean,
        "disp_mean": disp_mean,
        "trend_signed": trend,
        "trend_abs": abs(trend),
    }


def main() -> int:
    data_dir = Path("data/binance_l2_perp")
    frames = load_parquets(data_dir, DEFAULT_SYMBOLS)
    features = build_feature_frame(frames, DEFAULT_SYMBOLS)
    print(f"substrate: n_rows={features.n_rows}  n_symbols={features.n_symbols}")
    print()

    k_blocks = 8
    block_size = features.n_rows // k_blocks
    rows: list[BlockRow] = []
    for k in range(k_blocks):
        start = k * block_size
        end = (k + 1) * block_size if k < k_blocks - 1 else features.n_rows
        sub = slice_features(features, start, end)
        v = run_killtest(sub)
        feat_block = _features_per_block(features, start, end)
        ricci = cross_sectional_ricci_signal(sub.ofi)
        ricci_mean = float(np.nanmean(ricci))
        ricci_std = float(np.nanstd(ricci))
        row = BlockRow(
            block=k,
            start=start,
            end=end,
            n=int(sub.n_rows),
            ic_signal=float(v.ic_signal),
            residual_ic=float(v.residual_ic),
            residual_p=float(v.residual_ic_pvalue),
            perm_p=float(v.null_test_pvalues["permutation_shuffle"]),
            verdict=v.verdict,
            rv_mean=feat_block["rv_mean"],
            corr_mean=feat_block["corr_mean"],
            disp_mean=feat_block["disp_mean"],
            trend_signed=feat_block["trend_signed"],
            trend_abs=feat_block["trend_abs"],
            ricci_mean=ricci_mean,
            ricci_std=ricci_std,
        )
        rows.append(row)
        print(
            f"block={k}  IC={row.ic_signal:+.4f}  rv={row.rv_mean:.6f}  "
            f"corr={row.corr_mean:+.3f}  disp={row.disp_mean:.6f}  "
            f"trend={row.trend_signed:+.5f}  κ_mean={row.ricci_mean:+.4f}  "
            f"κ_std={row.ricci_std:.4f}  verdict={row.verdict}"
        )
    print()

    # Spearman rank corr between IC_signal and each regime feature
    df = pd.DataFrame([asdict(r) for r in rows])
    numeric_cols = [
        "rv_mean",
        "corr_mean",
        "disp_mean",
        "trend_signed",
        "trend_abs",
        "ricci_mean",
        "ricci_std",
    ]
    print("=== Spearman rank correlation: IC_signal vs regime features ===")
    corr_results: dict[str, dict[str, float]] = {}
    for col in numeric_cols:
        if df[col].notna().sum() < 4:
            continue
        rho, p = spearmanr(df["ic_signal"], df[col])
        corr_results[col] = {"rho": float(rho), "p": float(p)}
        print(f"  {col:<14}  ρ={rho:+.3f}  p={p:.3f}")
    print()

    out = {
        "n_blocks": k_blocks,
        "n_substrate_rows": features.n_rows,
        "blocks": [asdict(r) for r in rows],
        "ic_vs_feature_correlations": corr_results,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/REGIME_ANALYSIS.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print("wrote results/REGIME_ANALYSIS.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
