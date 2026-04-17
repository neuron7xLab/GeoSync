#!/usr/bin/env python3
"""Regime-conditional gate: does filtering by rolling realized vol rescue IC?

Walk-forward analysis identified rolling realized vol as the strongest
regime discriminator (Spearman ρ=+0.352, p=0.008 on 56 rolling windows,
Q1 IC median +0.027 vs Q4 median +0.137).

This script:
    1. Load the collected L2 substrate.
    2. Compute `rolling_rv_regime` (and `rolling_corr_regime` for comparison)
       at a few window sizes.
    3. At several quantile thresholds (keep top 75% / 50% / 25%), run
       `run_killtest(regime_mask=mask)`. Compare IC_conditional against
       the unconditional baseline.
    4. Report fraction of time regime is ON + IC uplift + residual IC.

If conditional IC dramatically > unconditional AND regime is ON a
non-trivial fraction of time → the regime filter is the next architectural
evolution.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.microstructure.killtest import (
    _load_parquets as load_parquets,
)
from research.microstructure.killtest import (
    build_feature_frame,
    run_killtest,
)
from research.microstructure.l2_schema import DEFAULT_SYMBOLS
from research.microstructure.regime import (
    regime_mask_from_quantile,
    rolling_corr_regime,
    rolling_rv_regime,
)


def _row(
    name: str,
    mask: np.ndarray | None,
    features: object,
) -> dict[str, float | str]:
    from research.microstructure.killtest import FeatureFrame  # noqa: PLC0415

    assert isinstance(features, FeatureFrame)
    f: FeatureFrame = features
    mask_bool = mask.astype(bool) if mask is not None else None
    v = run_killtest(f, regime_mask=mask_bool)
    frac_on = float(mask_bool.sum() / f.n_rows) if mask_bool is not None else 1.0
    return {
        "name": name,
        "frac_on": frac_on,
        "ic_signal": float(v.ic_signal) if np.isfinite(v.ic_signal) else float("nan"),
        "residual_ic": float(v.residual_ic) if np.isfinite(v.residual_ic) else float("nan"),
        "residual_p": float(v.residual_ic_pvalue),
        "perm_p": float(v.null_test_pvalues["permutation_shuffle"]),
        "verdict": v.verdict,
    }


def main() -> int:
    data_dir = Path("data/binance_l2_perp")
    frames = load_parquets(data_dir, DEFAULT_SYMBOLS)
    features = build_feature_frame(frames, DEFAULT_SYMBOLS)
    print(f"substrate: n_rows={features.n_rows}  n_symbols={features.n_symbols}\n")

    results: list[dict[str, float | str]] = []

    # --- Unconditional baseline ---
    results.append(_row("UNCONDITIONAL", None, features))

    # --- RV regime (primary, identified by walk-forward) ---
    for window_rows in (180, 300, 600):
        score = rolling_rv_regime(features, window_rows=window_rows)
        for q in (0.25, 0.50, 0.75):
            mask = regime_mask_from_quantile(score, quantile=q)
            results.append(
                _row(f"rv_w{window_rows}_q{int(q * 100):02d}", mask, features),
            )

    # --- Correlation regime (comparison) ---
    for window_rows in (180, 300, 600):
        score = rolling_corr_regime(features, window_rows=window_rows)
        for q in (0.25, 0.50, 0.75):
            mask = regime_mask_from_quantile(score, quantile=q)
            results.append(
                _row(f"corr_w{window_rows}_q{int(q * 100):02d}", mask, features),
            )

    # --- Combined: rv AND corr ---
    for window_rows in (300,):
        rv = rolling_rv_regime(features, window_rows=window_rows)
        cr = rolling_corr_regime(features, window_rows=window_rows)
        for q in (0.50,):
            rv_mask = regime_mask_from_quantile(rv, quantile=q)
            cr_mask = regime_mask_from_quantile(cr, quantile=q)
            mask = rv_mask & cr_mask
            results.append(
                _row(f"rv_AND_corr_w{window_rows}_q50", mask, features),
            )

    header = (
        f"{'name':<26} {'frac_on':>8} {'IC':>9} {'residual':>10} {'res_p':>7} {'perm_p':>7} verdict"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        assert isinstance(r["name"], str)
        assert isinstance(r["verdict"], str)
        print(
            f"{r['name']:<26} {float(r['frac_on']):>8.3f} "
            f"{float(r['ic_signal']):>+9.4f} {float(r['residual_ic']):>+10.4f} "
            f"{float(r['residual_p']):>7.4f} {float(r['perm_p']):>7.4f} {r['verdict']}"
        )

    Path("results").mkdir(exist_ok=True)
    Path("results/L2_REGIME_CONDITIONAL.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print("\nwrote results/L2_REGIME_CONDITIONAL.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
