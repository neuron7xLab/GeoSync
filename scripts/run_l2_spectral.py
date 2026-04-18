#!/usr/bin/env python3
"""Spectral CLI — Welch PSD + dominant period + redness slope on κ_min."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from research.microstructure.killtest import (
    _load_parquets as load_parquets,
)
from research.microstructure.killtest import (
    build_feature_frame,
    cross_sectional_ricci_signal,
)
from research.microstructure.l2_schema import DEFAULT_SYMBOLS
from research.microstructure.spectral import (
    DEFAULT_SEGMENT_SEC,
    spectral_report,
)

_log = logging.getLogger("l2_spectral")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/binance_l2_perp"))
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/L2_SPECTRAL.json"),
    )
    parser.add_argument("--segment-sec", type=int, default=DEFAULT_SEGMENT_SEC)
    parser.add_argument("--fs-hz", type=float, default=1.0)
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
    report = spectral_report(
        signal,
        fs_hz=float(args.fs_hz),
        segment_sec=int(args.segment_sec),
    )

    # Trim the stored PSD arrays to avoid a mega-json; expose top-N
    # highest-power bins plus the summary.
    psd_arr = list(report.psd)
    freq_arr = list(report.frequencies_hz)
    top_n = min(32, len(psd_arr))
    sort_idx = sorted(range(len(psd_arr)), key=lambda i: -psd_arr[i])[:top_n]
    top_bins = [
        {
            "freq_hz": freq_arr[i],
            "period_sec": (1.0 / freq_arr[i] if freq_arr[i] > 0 else None),
            "psd": psd_arr[i],
        }
        for i in sort_idx
    ]

    payload = {
        "n_rows": features.n_rows,
        "fs_hz": float(args.fs_hz),
        "segment_sec": int(args.segment_sec),
        "dominant_period_sec": report.dominant_period_sec,
        "dominant_peak_power": report.dominant_peak_power,
        "redness_slope_beta": report.redness_slope_beta,
        "redness_intercept": report.redness_intercept,
        "regime_verdict": report.regime_verdict,
        "top_power_bins": top_bins,
        "n_psd_bins": len(psd_arr),
        # Full arrays omitted from the tracked JSON — too large. Recomputable
        # by re-running this CLI against the same substrate.
    }

    body = json.dumps(payload, indent=2, sort_keys=True, default=str)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    print(body)
    _log.info(
        "dominant_period=%ss peak_power=%s  redness β=%.3f verdict=%s",
        report.dominant_period_sec,
        report.dominant_peak_power,
        report.redness_slope_beta,
        report.regime_verdict,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
