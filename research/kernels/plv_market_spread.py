"""Task 12: PLV between market phase and spread phase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import hilbert


def _phase(arr: np.ndarray) -> np.ndarray:
    centered = arr - np.mean(arr)
    return np.angle(hilbert(centered))


def _plv(phi1: np.ndarray, phi2: np.ndarray) -> float:
    return float(abs(np.mean(np.exp(1j * (phi1 - phi2)))))


def run(input_csv: Path, output_json: Path, n: int = 1000, seed: int = 42) -> dict:
    df = pd.read_csv(input_csv)
    midr = pd.Series(df["mid_returns"]).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    spr = pd.Series(df["spread"]).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    m = min(len(midr), len(spr))
    midr, spr = midr[:m], spr[:m]

    split = int(0.7 * m)
    phi_m = _phase(midr[split:])
    phi_s = _phase(spr[split:])
    obs = _plv(phi_m, phi_s)

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n):
        surr = rng.permutation(phi_s)
        if _plv(phi_m, surr) >= obs:
            count += 1
    p = (count + 1) / (n + 1)

    verdict = {
        "plv": round(obs, 6),
        "p_value": round(float(p), 6),
        "FINAL": "SIGNAL_READY" if obs > 0 and p < 0.05 else "REJECT",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return verdict


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", type=Path, default=Path("data/dukascopy/xauusd_l2_hourly.csv"))
    p.add_argument("--output-json", type=Path, default=Path("results/plv_spread_market_verdict.json"))
    args = p.parse_args()
    run(args.input_csv, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
