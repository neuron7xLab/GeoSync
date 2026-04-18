#!/usr/bin/env python3
"""Conditional Transfer Entropy CLI over the symbol OFI panel.

Takes BTCUSDT OFI (or the index-average OFI if BTC unavailable) as the
common-factor conditioner Z. For every ordered non-BTC pair (A, B),
computes TE(A→B | Z) and TE(B→A | Z). Addresses the common-factor
critique of the unconditional TE finding (PR #274): is pairwise flow
genuine A↔B coupling, or just common-response to market-wide drift?
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from research.microstructure.conditional_transfer_entropy import (
    DEFAULT_LAG_ROWS,
    DEFAULT_N_BINS,
    DEFAULT_N_SURROGATES,
    conditional_transfer_entropy,
)
from research.microstructure.killtest import (
    _load_parquets as load_parquets,
)
from research.microstructure.killtest import (
    build_feature_frame,
)
from research.microstructure.l2_schema import DEFAULT_SYMBOLS

_log = logging.getLogger("l2_conditional_te")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/binance_l2_perp"))
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/L2_CONDITIONAL_TE.json"),
    )
    parser.add_argument(
        "--conditioner",
        default="BTCUSDT",
        help="Symbol to use as common-factor Z (default BTCUSDT). "
        "Use 'basket' to condition on row-mean OFI across all symbols.",
    )
    parser.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS)
    parser.add_argument("--lag-rows", type=int, default=DEFAULT_LAG_ROWS)
    parser.add_argument("--n-surrogates", type=int, default=DEFAULT_N_SURROGATES)
    parser.add_argument("--seed", type=int, default=42)
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

    cond_name = str(args.conditioner).upper()
    if cond_name == "BASKET":
        z = features.ofi.mean(axis=1)
        non_cond_indices = list(range(features.n_symbols))
    else:
        if cond_name not in features.symbols:
            _log.error("conditioner %s not in universe %s", cond_name, features.symbols)
            return 2
        cond_idx = features.symbols.index(cond_name)
        z = features.ofi[:, cond_idx]
        non_cond_indices = [i for i in range(features.n_symbols) if i != cond_idx]

    pairs: list[dict[str, Any]] = []
    verdict_counts: dict[str, int] = {}
    for i_pos, i in enumerate(non_cond_indices):
        for j in non_cond_indices[i_pos + 1 :]:
            sym_a = features.symbols[i]
            sym_b = features.symbols[j]
            report = conditional_transfer_entropy(
                features.ofi[:, i],
                features.ofi[:, j],
                np.asarray(z, dtype=np.float64),
                n_bins=int(args.n_bins),
                lag_rows=int(args.lag_rows),
                n_surrogates=int(args.n_surrogates),
                seed=int(args.seed),
            )
            pairs.append(
                {
                    "symbol_x": sym_a,
                    "symbol_y": sym_b,
                    "report": asdict(report),
                }
            )
            verdict_counts[report.verdict] = verdict_counts.get(report.verdict, 0) + 1
            _log.info(
                "%s↔%s  TE=%.4f→CTE=%.4f  Δ=%.1f%%  p=%.3f  %s",
                sym_a,
                sym_b,
                report.te_unconditional_y_to_x_nats,
                report.te_conditional_y_to_x_nats,
                (
                    100.0 * report.reduction_fraction
                    if np.isfinite(report.reduction_fraction)
                    else float("nan")
                ),
                report.p_value_conditional,
                report.verdict,
            )

    payload: dict[str, Any] = {
        "n_rows": features.n_rows,
        "n_symbols": features.n_symbols,
        "conditioner": cond_name,
        "n_bins": int(args.n_bins),
        "lag_rows": int(args.lag_rows),
        "n_surrogates": int(args.n_surrogates),
        "seed": int(args.seed),
        "n_pairs": len(pairs),
        "verdict_counts": verdict_counts,
        "pairs": pairs,
    }
    body = json.dumps(payload, indent=2, sort_keys=True, default=str)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    print(body)

    _log.info("verdict summary: %s", verdict_counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
