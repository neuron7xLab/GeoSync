#!/usr/bin/env python3
"""Recursive + cyclic reality check on the collected L2 substrate.

Uses only the existing primitives (`build_feature_frame`, `slice_features`,
`run_killtest`, `run_killtest_split`). No new dataclasses, no new modules,
no new gate logic. Two orthogonal views:

1. RECURSIVE BISECTION (depth-first): at depth d, each cell is a 1/(2**d)
   contiguous slice of the full window. Reports IC + residual_IC at every
   cell. Signal is `deep` iff it survives every leaf at some depth.
2. CYCLIC BLOCKS (breadth): split the full window into K adjacent disjoint
   blocks of equal size; report IC trajectory across them. Signal is
   `stable` iff IC sign + magnitude are preserved across blocks.

Reality = what both views say simultaneously.
"""

from __future__ import annotations

from pathlib import Path

from research.microstructure.killtest import (
    _load_parquets as load_parquets,
)
from research.microstructure.killtest import (
    build_feature_frame,
    run_killtest,
    slice_features,
)
from research.microstructure.l2_schema import DEFAULT_SYMBOLS

_MIN_ROWS_PER_CELL = 1500
_MAX_DEPTH = 3
_CYCLIC_K = 8


def _recurse(features_obj: object, path: str, depth: int, results: list[dict[str, object]]) -> None:
    from research.microstructure.killtest import FeatureFrame  # noqa: PLC0415

    assert isinstance(features_obj, FeatureFrame)
    features: FeatureFrame = features_obj

    if features.n_rows < _MIN_ROWS_PER_CELL:
        results.append(
            {
                "path": path,
                "depth": depth,
                "n_samples": features.n_rows,
                "ic_signal": float("nan"),
                "residual_ic": float("nan"),
                "residual_p": float("nan"),
                "note": "too_small",
            }
        )
        return

    v = run_killtest(features)
    results.append(
        {
            "path": path,
            "depth": depth,
            "n_samples": v.n_samples,
            "ic_signal": v.ic_signal,
            "residual_ic": v.residual_ic,
            "residual_p": v.residual_ic_pvalue,
            "verdict": v.verdict,
            "reasons_count": len(v.reasons),
        }
    )

    if depth >= _MAX_DEPTH:
        return
    mid = features.n_rows // 2
    left = slice_features(features, 0, mid)
    right = slice_features(features, mid, features.n_rows)
    _recurse(left, f"{path}L", depth + 1, results)
    _recurse(right, f"{path}R", depth + 1, results)


def main() -> int:
    data_dir = Path("data/binance_l2_perp")
    frames = load_parquets(data_dir, DEFAULT_SYMBOLS)
    features = build_feature_frame(frames, DEFAULT_SYMBOLS)
    print(f"substrate: n_rows={features.n_rows}  n_symbols={features.n_symbols}")
    print()

    # --- 1. Recursive bisection ---
    print("=" * 74)
    print("RECURSIVE BISECTION TREE  (depth 0 = full; L/R = halves at each split)")
    print("=" * 74)
    tree: list[dict[str, object]] = []
    _recurse(features, "·", 0, tree)
    print(f"{'path':<10} {'depth':<6} {'n':<7} {'IC':>8} {'residual':>10} {'p':>8} {'verdict':<10}")
    for row in tree:
        p = row["path"]
        d = row["depth"]
        n = row["n_samples"]
        if row.get("note") == "too_small":
            print(f"{p:<10} {d:<6} {n:<7}   — too small for stable IC —")
            continue
        ic = row["ic_signal"]
        rr = row["residual_ic"]
        pv = row["residual_p"]
        vd = row["verdict"]
        assert isinstance(p, str) and isinstance(d, int) and isinstance(n, int)
        assert isinstance(ic, float) and isinstance(rr, float) and isinstance(pv, float)
        assert isinstance(vd, str)
        print(f"{p:<10} {d:<6} {n:<7} {ic:>+8.4f} {rr:>+10.4f} {pv:>8.4f} {vd:<10}")
    print()

    # --- 2. Cyclic K blocks ---
    print("=" * 74)
    print(f"CYCLIC BLOCKS  (K={_CYCLIC_K} adjacent disjoint windows)")
    print("=" * 74)
    block = features.n_rows // _CYCLIC_K
    print(
        f"{'block':<6} {'start':<6} {'end':<6} {'n':<7} {'IC':>8} {'residual':>10} {'p':>8} {'verdict':<10}"
    )
    ic_series: list[float] = []
    for k in range(_CYCLIC_K):
        start = k * block
        end = (k + 1) * block if k < _CYCLIC_K - 1 else features.n_rows
        sub = slice_features(features, start, end)
        if sub.n_rows < _MIN_ROWS_PER_CELL:
            print(f"{k:<6} {start:<6} {end:<6} {sub.n_rows:<7}   — too small —")
            continue
        v = run_killtest(sub)
        ic_series.append(v.ic_signal)
        print(
            f"{k:<6} {start:<6} {end:<6} {v.n_samples:<7} "
            f"{v.ic_signal:>+8.4f} {v.residual_ic:>+10.4f} {v.residual_ic_pvalue:>8.4f} "
            f"{v.verdict:<10}"
        )
    print()
    if ic_series:
        n_pos = sum(1 for ic in ic_series if ic > 0)
        avg = sum(ic_series) / len(ic_series)
        print(
            f"summary: {n_pos}/{len(ic_series)} blocks positive IC  avg={avg:+.4f}  "
            f"min={min(ic_series):+.4f}  max={max(ic_series):+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
