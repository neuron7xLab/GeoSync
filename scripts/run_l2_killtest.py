#!/usr/bin/env python3
"""Run the fail-fast L2 kill test against collected Binance-perp shards.

Thin CLI wrapper around `research.microstructure.killtest`. Loads parquet
shards, builds feature frame, executes gate, writes verdict JSON.

Exit codes:
    0 — verdict produced (PROCEED or KILL); check JSON body
    2 — insufficient data / bad arguments
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from research.microstructure.killtest import (
    _load_parquets as load_parquets,
)
from research.microstructure.killtest import (
    build_feature_frame,
    run_killtest,
    run_killtest_split,
    split_verdict_to_json,
    verdict_to_json,
)
from research.microstructure.l2_schema import DEFAULT_SYMBOLS

_log = logging.getLogger("l2_killtest")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/binance_l2_perp"),
        help="Directory containing per-symbol parquet shards",
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated list of symbols to include",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/L2_KILLTEST_VERDICT.json"),
        help="Path to write verdict JSON",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=None,
        help="If set (e.g. 0.5), run train/test split OOS gate instead of single-window gate",
    )
    parser.add_argument(
        "--retention-gate",
        type=float,
        default=0.5,
        help="Minimum IC(test)/IC(train) ratio for split PROCEED (default 0.5)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
    )
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
        _log.error("no parquet shards found in %s", data_dir)
        return 2
    _log.info("loaded %d symbol frames: %s", len(frames), sorted(frames.keys()))

    try:
        features = build_feature_frame(frames, symbols)
    except ValueError as exc:
        _log.error("insufficient overlap: %s", exc)
        return 2

    _log.info(
        "feature frame: %d rows × %d symbols",
        features.n_rows,
        features.n_symbols,
    )

    if args.split is not None:
        split_verdict = run_killtest_split(
            features,
            split_at_fraction=float(args.split),
            retention_gate=float(args.retention_gate),
        )
        json_body = split_verdict_to_json(split_verdict)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_body, encoding="utf-8")
        print(json_body)
        _log.info(
            "split verdict: %s — retention=%.3f — reasons: %s",
            split_verdict.verdict,
            split_verdict.ic_retention,
            split_verdict.reasons or "none",
        )
        return 0

    verdict = run_killtest(features)
    json_body = verdict_to_json(verdict)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json_body, encoding="utf-8")

    print(json_body)
    _log.info("verdict: %s — reasons: %s", verdict.verdict, verdict.reasons or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
