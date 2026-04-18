#!/usr/bin/env python3
"""Rolling-calibration walk-forward: most production-realistic OOS test.

Slide two adjacent windows through the substrate:

    |---- CALIBRATION (W rows) ----|---- EVALUATION (E rows) ----|
                                    ^ threshold applied here
    |~>  shift by E, repeat

At each step:
    * compute rolling_rv_regime on CALIBRATION slice → derive quantile thresholds
    * compute rolling_rv_regime on EVALUATION slice  → apply each threshold
    * run `run_killtest` on EVALUATION, unconditionally + for each threshold
    * collect IC_unconditional, IC_conditional, frac_on, uplift

If uplift > 0 in the majority of steps AND aggregate IC_conditional
substantially exceeds aggregate IC_unconditional, the threshold is
robust under production-style rolling recalibration.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from research.microstructure.killtest import (
    _load_parquets as load_parquets,
)
from research.microstructure.killtest import (
    build_feature_frame,
    run_killtest,
    slice_features,
)
from research.microstructure.l2_schema import DEFAULT_SYMBOLS
from research.microstructure.regime import (
    regime_mask_from_score,
    rolling_rv_regime,
)

_CALIB_ROWS: int = 3600  # 60 min
_EVAL_ROWS: int = 1800  # 30 min
_WINDOW_ROWS: int = 300
_QUANTILES: tuple[float, ...] = (0.50, 0.75)


@dataclass
class Step:
    step: int
    calib_start: int
    calib_end: int
    eval_start: int
    eval_end: int
    ic_unconditional: float
    ic_q50: float
    ic_q75: float
    frac_on_q50: float
    frac_on_q75: float
    uplift_q50: float
    uplift_q75: float
    thr_q50: float
    thr_q75: float


def main() -> int:
    data_dir = Path("data/binance_l2_perp")
    frames = load_parquets(data_dir, DEFAULT_SYMBOLS)
    features = build_feature_frame(frames, DEFAULT_SYMBOLS)
    print(f"substrate: n_rows={features.n_rows}")
    n = features.n_rows
    steps: list[Step] = []
    idx = 0
    while True:
        calib_start = idx
        calib_end = calib_start + _CALIB_ROWS
        eval_start = calib_end
        eval_end = eval_start + _EVAL_ROWS
        if eval_end > n:
            break

        calib = slice_features(features, calib_start, calib_end)
        ev = slice_features(features, eval_start, eval_end)

        train_score = rolling_rv_regime(calib, window_rows=_WINDOW_ROWS)
        train_finite = train_score[np.isfinite(train_score)]
        if train_finite.size == 0:
            idx += _EVAL_ROWS
            continue
        thr = {q: float(np.quantile(train_finite, q)) for q in _QUANTILES}

        test_score = rolling_rv_regime(ev, window_rows=_WINDOW_ROWS)

        v_un = run_killtest(ev)
        v_q50 = run_killtest(
            ev, regime_mask=regime_mask_from_score(test_score, threshold=thr[0.50])
        )
        v_q75 = run_killtest(
            ev, regime_mask=regime_mask_from_score(test_score, threshold=thr[0.75])
        )

        mask_q50 = regime_mask_from_score(test_score, threshold=thr[0.50])
        mask_q75 = regime_mask_from_score(test_score, threshold=thr[0.75])

        ic_un = float(v_un.ic_signal) if np.isfinite(v_un.ic_signal) else float("nan")
        ic_q50 = float(v_q50.ic_signal) if np.isfinite(v_q50.ic_signal) else float("nan")
        ic_q75 = float(v_q75.ic_signal) if np.isfinite(v_q75.ic_signal) else float("nan")
        uplift_q50 = ic_q50 - ic_un if np.isfinite(ic_un) and np.isfinite(ic_q50) else float("nan")
        uplift_q75 = ic_q75 - ic_un if np.isfinite(ic_un) and np.isfinite(ic_q75) else float("nan")

        steps.append(
            Step(
                step=len(steps),
                calib_start=calib_start,
                calib_end=calib_end,
                eval_start=eval_start,
                eval_end=eval_end,
                ic_unconditional=ic_un,
                ic_q50=ic_q50,
                ic_q75=ic_q75,
                frac_on_q50=float(mask_q50.sum() / ev.n_rows),
                frac_on_q75=float(mask_q75.sum() / ev.n_rows),
                uplift_q50=uplift_q50,
                uplift_q75=uplift_q75,
                thr_q50=thr[0.50],
                thr_q75=thr[0.75],
            )
        )
        idx += _EVAL_ROWS

    # Print per-step
    header = (
        f"{'step':<4} {'ev_range':<13} {'IC_un':>7} {'IC_q50':>7} {'IC_q75':>7} "
        f"{'up50':>6} {'up75':>6} {'on50':>5} {'on75':>5}"
    )
    print(header)
    print("-" * len(header))
    for s in steps:
        print(
            f"{s.step:<4} {s.eval_start:>5}-{s.eval_end:<5} "
            f"{s.ic_unconditional:>+7.4f} {s.ic_q50:>+7.4f} {s.ic_q75:>+7.4f} "
            f"{s.uplift_q50:>+6.3f} {s.uplift_q75:>+6.3f} "
            f"{s.frac_on_q50:>5.2%} {s.frac_on_q75:>5.2%}"
        )
    print()

    def _agg(xs: list[float]) -> dict[str, float]:
        clean = [x for x in xs if np.isfinite(x)]
        if not clean:
            return {"mean": float("nan"), "median": float("nan"), "pos_frac": float("nan")}
        return {
            "mean": float(statistics.mean(clean)),
            "median": float(statistics.median(clean)),
            "pos_frac": float(sum(1 for x in clean if x > 0) / len(clean)),
        }

    ic_un_stats = _agg([s.ic_unconditional for s in steps])
    ic_q50_stats = _agg([s.ic_q50 for s in steps])
    ic_q75_stats = _agg([s.ic_q75 for s in steps])
    up50_stats = _agg([s.uplift_q50 for s in steps])
    up75_stats = _agg([s.uplift_q75 for s in steps])

    print("=== AGGREGATE across steps ===")
    print(
        f"IC unconditional: mean={ic_un_stats['mean']:+.4f}  "
        f"median={ic_un_stats['median']:+.4f}  pos_frac={ic_un_stats['pos_frac']:.2%}"
    )
    print(
        f"IC q50:           mean={ic_q50_stats['mean']:+.4f}  "
        f"median={ic_q50_stats['median']:+.4f}  pos_frac={ic_q50_stats['pos_frac']:.2%}"
    )
    print(
        f"IC q75:           mean={ic_q75_stats['mean']:+.4f}  "
        f"median={ic_q75_stats['median']:+.4f}  pos_frac={ic_q75_stats['pos_frac']:.2%}"
    )
    print(
        f"uplift q50:       mean={up50_stats['mean']:+.4f}  "
        f"median={up50_stats['median']:+.4f}  pos_frac={up50_stats['pos_frac']:.2%}"
    )
    print(
        f"uplift q75:       mean={up75_stats['mean']:+.4f}  "
        f"median={up75_stats['median']:+.4f}  pos_frac={up75_stats['pos_frac']:.2%}"
    )

    out = {
        "calib_rows": _CALIB_ROWS,
        "eval_rows": _EVAL_ROWS,
        "window_rows": _WINDOW_ROWS,
        "steps": [asdict(s) for s in steps],
        "aggregate": {
            "ic_unconditional": ic_un_stats,
            "ic_q50": ic_q50_stats,
            "ic_q75": ic_q75_stats,
            "uplift_q50": up50_stats,
            "uplift_q75": up75_stats,
        },
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/L2_REGIME_WALKFORWARD.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print("\nwrote results/L2_REGIME_WALKFORWARD.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
