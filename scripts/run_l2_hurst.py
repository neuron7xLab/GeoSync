#!/usr/bin/env python3
"""Hurst exponent CLI — scale-free cross-check for spectral β.

Consumes one L2 substrate; produces a Hurst JSON that independently
confirms or contradicts the spectral redness finding (PR #271, β=+1.80):

    β ≈ 2.0  ↔  H ≈ 1.5  (Brownian / random-walk persistence)

Writes results/L2_HURST.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from research.microstructure.hurst import (
    DEFAULT_MAX_SCALE_FRAC,
    DEFAULT_MIN_SCALE,
    DEFAULT_N_SCALES,
    dfa_hurst,
)
from research.microstructure.killtest import (
    _load_parquets as load_parquets,
)
from research.microstructure.killtest import (
    build_feature_frame,
    cross_sectional_ricci_signal,
)
from research.microstructure.l2_schema import DEFAULT_SYMBOLS

_log = logging.getLogger("l2_hurst")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/binance_l2_perp"))
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/L2_HURST.json"),
    )
    parser.add_argument("--min-scale", type=int, default=DEFAULT_MIN_SCALE)
    parser.add_argument("--max-scale-frac", type=float, default=DEFAULT_MAX_SCALE_FRAC)
    parser.add_argument("--n-scales", type=int, default=DEFAULT_N_SCALES)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    symbols = tuple(s.strip().upper() for s in str(args.symbols).split(",") if s.strip())
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        _log.error("data dir does not exist: %s", data_dir)
        return 2

    frames = load_parquets(data_dir, symbols)
    if not frames:
        _log.error("no parquet shards in %s", data_dir)
        return 2
    try:
        features = build_feature_frame(frames, symbols)
    except ValueError as exc:
        _log.error("insufficient overlap: %s", exc)
        return 2

    signal = cross_sectional_ricci_signal(features.ofi)
    report = dfa_hurst(
        signal,
        min_scale=int(args.min_scale),
        max_scale_frac=float(args.max_scale_frac),
        n_scales=int(args.n_scales),
    )

    payload: dict[str, Any] = {
        "n_rows": features.n_rows,
        "n_symbols": features.n_symbols,
        "min_scale": int(args.min_scale),
        "max_scale_frac": float(args.max_scale_frac),
        "n_scales": int(args.n_scales),
        "report": asdict(report),
    }
    body = json.dumps(payload, indent=2, sort_keys=True, default=str)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    print(body)

    _log.info(
        "Hurst H=%.4f R²=%.4f verdict=%s (n_scales=%d, n=%d)",
        report.hurst_exponent,
        report.r_squared,
        report.verdict,
        len(report.scales),
        report.n_samples_used,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
