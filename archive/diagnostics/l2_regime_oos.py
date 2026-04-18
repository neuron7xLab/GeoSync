#!/usr/bin/env python3
"""True OOS test of the regime-conditional gate.

The regime_conditional script uses full-window quantile thresholds, which
have look-ahead bias (the threshold is computed on data that includes
the test region). This script eliminates that:

    1. Split substrate 50/50 in time: train (first half) + test (second).
    2. Compute rolling_rv_regime on TRAIN only.
    3. Derive quantile thresholds from TRAIN's finite scores.
    4. Apply those thresholds to TEST's rolling_rv_regime (computed on
       test data alone — no information from train flows in).
    5. Run run_killtest on test, unconditionally and with each mask.
    6. Compare IC_conditional vs IC_unconditional on TEST.

If conditional OOS IC materially > unconditional OOS IC, the regime
filter generalizes. If not, the full-window lift was an artifact of
threshold fit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.microstructure.killtest import (
    FeatureFrame,
    build_feature_frame,
    run_killtest,
    slice_features,
)
from research.microstructure.killtest import (
    _load_parquets as load_parquets,
)
from research.microstructure.l2_schema import DEFAULT_SYMBOLS
from research.microstructure.regime import (
    regime_mask_from_score,
    rolling_rv_regime,
)

_WINDOW_ROWS: int = 300
_QUANTILES: tuple[float, ...] = (0.25, 0.50, 0.75)


def _row(
    name: str,
    mask: np.ndarray | None,
    features: FeatureFrame,
) -> dict[str, float | str]:
    mask_bool = mask.astype(bool) if mask is not None else None
    v = run_killtest(features, regime_mask=mask_bool)
    frac_on = float(mask_bool.sum() / features.n_rows) if mask_bool is not None else 1.0
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
    print(f"substrate: n_rows={features.n_rows}  n_symbols={features.n_symbols}")

    mid = features.n_rows // 2
    train = slice_features(features, 0, mid)
    test = slice_features(features, mid, features.n_rows)
    print(f"train rows: {train.n_rows}   test rows: {test.n_rows}\n")

    # Calibrate thresholds on train alone.
    train_score = rolling_rv_regime(train, window_rows=_WINDOW_ROWS)
    train_finite = train_score[np.isfinite(train_score)]
    if train_finite.size == 0:
        print("train had zero finite rolling-rv scores; aborting")
        return 2
    thresholds = {q: float(np.quantile(train_finite, q)) for q in _QUANTILES}
    print("thresholds from TRAIN:")
    for q, thr in thresholds.items():
        print(f"  quantile q{int(q * 100):02d}  threshold={thr:.8f}")
    print()

    # Apply to test (compute test's own score — no train contamination).
    test_score = rolling_rv_regime(test, window_rows=_WINDOW_ROWS)
    results: list[dict[str, float | str]] = []
    results.append(_row("TEST_UNCONDITIONAL", None, test))
    for q, thr in thresholds.items():
        mask = regime_mask_from_score(test_score, threshold=thr)
        results.append(_row(f"TEST_q{int(q * 100):02d}_thr_from_train", mask, test))

    # Also compute train-own IC for reference
    results.append(_row("TRAIN_UNCONDITIONAL", None, train))

    header = (
        f"{'name':<32} {'frac_on':>8} {'IC':>9} {'residual':>10} {'res_p':>7} {'perm_p':>7} verdict"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        assert isinstance(r["name"], str)
        assert isinstance(r["verdict"], str)
        print(
            f"{r['name']:<32} {float(r['frac_on']):>8.3f} "
            f"{float(r['ic_signal']):>+9.4f} {float(r['residual_ic']):>+10.4f} "
            f"{float(r['residual_p']):>7.4f} {float(r['perm_p']):>7.4f} {r['verdict']}"
        )

    Path("results").mkdir(exist_ok=True)
    Path("results/L2_REGIME_OOS.json").write_text(
        json.dumps(
            {
                "window_rows": _WINDOW_ROWS,
                "train_rows": train.n_rows,
                "test_rows": test.n_rows,
                "thresholds_from_train": thresholds,
                "results": results,
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    print("\nwrote results/L2_REGIME_OOS.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
