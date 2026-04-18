#!/usr/bin/env python3
"""Cross-session OOS test of the regime filter.

The 50/50 split in `l2_regime_oos.py` tests generalization across two
halves of one 5-hour session. This script tests the stronger claim:
does the threshold calibrated on session 1 still lift IC on a
different session with independent market conditions?

Usage:
    python scripts/l2_regime_cross_session.py \\
        --train-dir data/binance_l2_perp \\
        --test-dir data/binance_l2_perp_v2

Train session → derive rv-threshold quantiles.
Test session  → compute rolling_rv, apply threshold, run killtest.

If conditional IC on session 2 > unconditional IC on session 2
by the same magnitude as within session 1 → regime filter is
production-ready.  If the uplift collapses across sessions → the
threshold was session-specific, not a genuine regime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from research.microstructure.killtest import (
    FeatureFrame,
    build_feature_frame,
    run_killtest,
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


def _load_feature_frame(data_dir: Path, label: str) -> FeatureFrame:
    frames = load_parquets(data_dir, DEFAULT_SYMBOLS)
    if not frames:
        print(f"no parquet shards in {data_dir}", file=sys.stderr)
        sys.exit(2)
    features = build_feature_frame(frames, DEFAULT_SYMBOLS)
    print(f"{label}: n_rows={features.n_rows}  n_symbols={features.n_symbols}  dir={data_dir}")
    return features


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("data/binance_l2_perp"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/binance_l2_perp_v2"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/L2_REGIME_CROSS_SESSION.json"),
    )
    args = parser.parse_args()

    train = _load_feature_frame(Path(args.train_dir), "train")
    test = _load_feature_frame(Path(args.test_dir), "test")
    print()

    # Calibrate threshold on train session alone.
    train_score = rolling_rv_regime(train, window_rows=_WINDOW_ROWS)
    train_finite = train_score[np.isfinite(train_score)]
    if train_finite.size == 0:
        print("train session produced zero finite rolling-rv scores", file=sys.stderr)
        return 2
    thresholds = {q: float(np.quantile(train_finite, q)) for q in _QUANTILES}
    print("thresholds from TRAIN session:")
    for q, thr in thresholds.items():
        print(f"  q{int(q * 100):02d}   threshold={thr:.8f}")
    print()

    test_score = rolling_rv_regime(test, window_rows=_WINDOW_ROWS)
    test_finite = test_score[np.isfinite(test_score)]
    if test_finite.size > 0:
        print(
            "TEST-session rv distribution:  "
            f"min={test_finite.min():.6f}  q25={np.quantile(test_finite, 0.25):.6f}  "
            f"median={np.median(test_finite):.6f}  "
            f"q75={np.quantile(test_finite, 0.75):.6f}  max={test_finite.max():.6f}"
        )
    print()

    results: list[dict[str, float | str]] = []
    results.append(_row("TEST_UNCONDITIONAL", None, test))
    results.append(_row("TRAIN_UNCONDITIONAL_REFERENCE", None, train))
    for q, thr in thresholds.items():
        mask = regime_mask_from_score(test_score, threshold=thr)
        results.append(_row(f"TEST_q{int(q * 100):02d}_thr_from_train={thr:.6f}", mask, test))

    header = (
        f"{'name':<46} {'frac_on':>8} {'IC':>9} {'residual':>10} {'res_p':>7} {'perm_p':>7} verdict"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        assert isinstance(r["name"], str)
        assert isinstance(r["verdict"], str)
        print(
            f"{r['name']:<46} {float(r['frac_on']):>8.3f} "
            f"{float(r['ic_signal']):>+9.4f} {float(r['residual_ic']):>+10.4f} "
            f"{float(r['residual_p']):>7.4f} {float(r['perm_p']):>7.4f} {r['verdict']}"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "window_rows": _WINDOW_ROWS,
                "train_dir": str(args.train_dir),
                "test_dir": str(args.test_dir),
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
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
