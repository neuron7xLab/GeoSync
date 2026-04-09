#!/usr/bin/env python3
# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Live-market backtest for GeoSync on real EURUSD data.

Production-oriented utility:
- data-source policy: prefer local cache CSV (`--cache-csv`) for reproducible
  runs; if missing, fallback to Yahoo Finance (`yfinance`)
- runs a deterministic rolling-window GeoSync signal pipeline
- executes via EventDrivenBacktestEngine
- exports artifacts (equity + signals) for audit and visualization
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
    yf = None
    YFINANCE_IMPORT_ERROR = exc
else:
    YFINANCE_IMPORT_ERROR = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if not hasattr(_dt, "UTC"):  # pragma: no cover - Python < 3.11 compat shim
    _dt.UTC = _dt.timezone.utc  # type: ignore[attr-defined]

from backtest.event_driven import EventDrivenBacktestEngine  # noqa: E402
from core.indicators.kuramoto_ricci_composite import GeoSyncCompositeEngine  # noqa: E402


@dataclass(slots=True)
class LiveConfig:
    ticker: str = "EURUSD=X"
    interval: str = "1h"
    start: str = "2023-01-01"
    end: str = "2025-01-01"
    window: int = 200
    initial_capital: float = 100_000.0
    r_long_threshold: float = 0.70
    r_flat_threshold: float = 0.30
    min_entry_signal: float = 0.0
    min_confidence: float = 0.50
    exit_threshold: float = 0.50
    cache_csv: Path | None = None
    output_dir: Path = Path("backtest/output")


def _download_prices(cfg: LiveConfig) -> pd.DataFrame:
    """Load market data with deterministic source policy.

    Priority:
    1) local cache CSV if provided and exists
    2) Yahoo Finance download fallback

    If fallback is needed but `yfinance` is unavailable, raise actionable error.
    """
    if cfg.cache_csv and cfg.cache_csv.exists():
        cached = pd.read_csv(cfg.cache_csv, index_col=0, parse_dates=True)
        if "close" in cached.columns and not cached.empty:
            return cached.sort_index()
        raise ValueError(f"Cache file is invalid: {cfg.cache_csv}")

    if yf is None:
        cache_hint = (
            f"Cache CSV '{cfg.cache_csv}' was requested but not found. "
            if cfg.cache_csv is not None
            else ""
        )
        raise ModuleNotFoundError(
            f"{cache_hint}yfinance is not installed. Install it with "
            "`pip install yfinance` or provide --cache-csv with pre-downloaded data."
        ) from YFINANCE_IMPORT_ERROR

    raw = yf.download(
        cfg.ticker,
        start=cfg.start,
        end=cfg.end,
        interval=cfg.interval,
        progress=False,
        auto_adjust=False,
    )
    if raw.empty:
        raise RuntimeError(
            f"No data downloaded for {cfg.ticker} {cfg.interval} {cfg.start}->{cfg.end}"
        )

    close_col = ("Close", cfg.ticker) if isinstance(raw.columns, pd.MultiIndex) else "Close"
    close = raw[close_col].rename("close").astype(float).dropna()
    df = close.to_frame()
    df["volume"] = 1_000.0

    if cfg.cache_csv:
        cfg.cache_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg.cache_csv)
    return df


def _build_signal_series(market_df: pd.DataFrame, cfg: LiveConfig) -> pd.DataFrame:
    prices = market_df["close"].to_numpy(dtype=np.float64, copy=False)
    signal = np.zeros(prices.shape[0], dtype=np.float64)
    r_vals = np.full(prices.shape[0], np.nan, dtype=np.float64)
    conf_vals = np.full(prices.shape[0], np.nan, dtype=np.float64)
    entry_vals = np.full(prices.shape[0], np.nan, dtype=np.float64)
    risk_vals = np.full(prices.shape[0], np.nan, dtype=np.float64)
    phase_vals = ["warmup"] * prices.shape[0]
    block_reason = ["warmup"] * prices.shape[0]

    engine = GeoSyncCompositeEngine()

    for i in range(cfg.window, prices.shape[0]):
        window_df = market_df.iloc[i - cfg.window : i]
        snap = engine.analyze_market(window_df)

        r_vals[i] = float(snap.kuramoto_R)
        conf_vals[i] = float(snap.confidence)
        entry_vals[i] = float(snap.entry_signal)
        risk_vals[i] = float(snap.risk_multiplier)
        phase_vals[i] = snap.phase.value

        prev = signal[i - 1]
        entry_ready = (
            snap.kuramoto_R >= cfg.r_long_threshold
            and snap.entry_signal > cfg.min_entry_signal
            and snap.confidence >= cfg.min_confidence
        )
        exit_ready = (
            snap.kuramoto_R <= cfg.r_flat_threshold or snap.exit_signal > cfg.exit_threshold
        )

        if entry_ready:
            signal[i] = float(np.clip(snap.risk_multiplier, 0.0, 1.0))
            block_reason[i] = "entry_ok"
        elif exit_ready:
            signal[i] = 0.0
            block_reason[i] = "exit_gate"
        else:
            signal[i] = prev
            if prev > 0.0:
                block_reason[i] = "hold_open"
            elif snap.kuramoto_R < cfg.r_long_threshold:
                block_reason[i] = "blocked_r_threshold"
            elif snap.confidence < cfg.min_confidence:
                block_reason[i] = "blocked_confidence"
            elif snap.entry_signal <= cfg.min_entry_signal:
                block_reason[i] = "blocked_entry_signal"
            else:
                block_reason[i] = f"blocked_phase_{snap.phase.value}"

    out = pd.DataFrame(
        {
            "close": prices,
            "signal": signal,
            "kuramoto_R": r_vals,
            "confidence": conf_vals,
            "entry_signal": entry_vals,
            "risk_multiplier": risk_vals,
            "phase": phase_vals,
            "block_reason": block_reason,
        },
        index=market_df.index,
    )
    return out


def _run_backtest(signal_df: pd.DataFrame, cfg: LiveConfig):
    prices = signal_df["close"].to_numpy(dtype=np.float64, copy=False)
    signals = signal_df["signal"].to_numpy(dtype=np.float64, copy=False)

    def signal_fn(_: np.ndarray) -> np.ndarray:
        return signals

    engine = EventDrivenBacktestEngine()
    return engine.run(
        prices,
        signal_fn,
        initial_capital=cfg.initial_capital,
        strategy_name="geosync_kuramoto_live_eurusd",
    )


def _save_artifacts(signal_df: pd.DataFrame, result, cfg: LiveConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    signal_path = cfg.output_dir / "signals.csv"
    signal_df.to_csv(signal_path)

    if result.equity_curve is not None:
        eq = pd.Series(result.equity_curve, index=signal_df.index, name="equity")
        eq.to_csv(cfg.output_dir / "equity_curve.csv")

    cfg_dict = asdict(cfg)
    cfg_dict["cache_csv"] = str(cfg.cache_csv) if cfg.cache_csv is not None else None
    cfg_dict["output_dir"] = str(cfg.output_dir)
    summary_payload = {
        "config": cfg_dict,
        "bars": int(len(signal_df)),
        "trades": int(result.trades),
        "pnl": float(result.pnl),
        "return_pct": float(result.pnl / cfg.initial_capital * 100.0),
        "max_drawdown": float(result.max_dd),
        "commission_cost": float(result.commission_cost),
        "entry_events": int((signal_df["block_reason"] == "entry_ok").sum()),
        "sharpe": None,
    }
    perf = getattr(result, "performance", None)
    if perf is not None and hasattr(perf, "sharpe"):
        summary_payload["sharpe"] = float(perf.sharpe)
    (cfg.output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _nan_stats(series: pd.Series) -> dict[str, float | None]:
    clean = series.dropna()
    if clean.empty:
        return {"count": 0, "p05": None, "p50": None, "p95": None, "mean": None}
    return {
        "count": int(clean.shape[0]),
        "p05": float(clean.quantile(0.05)),
        "p50": float(clean.quantile(0.50)),
        "p95": float(clean.quantile(0.95)),
        "mean": float(clean.mean()),
    }


def _write_diagnostic_report(signal_df: pd.DataFrame, result, cfg: LiveConfig) -> None:
    phase_counts = signal_df["phase"].value_counts().to_dict()
    block_counts = signal_df["block_reason"].value_counts().to_dict()
    r_stats = _nan_stats(signal_df["kuramoto_R"])
    conf_stats = _nan_stats(signal_df["confidence"])
    entry_stats = _nan_stats(signal_df["entry_signal"])
    risk_stats = _nan_stats(signal_df["risk_multiplier"])
    entries = int((signal_df["block_reason"] == "entry_ok").sum())

    report = [
        "# Diagnostic report: live EURUSD backtest",
        "",
        "## Root cause",
        (
            "- Primary bottleneck: `transition` regimes were producing near-zero"
            " practical entries because historical mapping was effectively too"
            " conservative for persistent EURUSD transition states."
        ),
        (
            "- Consequence: even when Kuramoto synchronization (`R`) was high,"
            " execution gating was blocked by low/zero `entry_signal` for long"
            " stretches."
        ),
        "",
        "## Patch",
        "- Added transition-aware entry mapping in composite engine with fail-closed constraints (`R > Rp`, `temporal_ricci < 0`).",
        "- Added per-bar no-entry reason diagnostics in live runner.",
        "- Added audit artifacts (`signals.csv`, `equity_curve.csv`, `summary.json`, this report).",
        "",
        "## Distributions",
        f"- kuramoto_R: {json.dumps(r_stats, ensure_ascii=False)}",
        f"- confidence: {json.dumps(conf_stats, ensure_ascii=False)}",
        f"- entry_signal: {json.dumps(entry_stats, ensure_ascii=False)}",
        f"- risk_multiplier: {json.dumps(risk_stats, ensure_ascii=False)}",
        "",
        "## Counters",
        f"- phase_counts: {json.dumps(phase_counts, ensure_ascii=False)}",
        f"- block_reason_counts: {json.dumps(block_counts, ensure_ascii=False)}",
        f"- entry_events: {entries}",
        f"- trades: {result.trades}",
        f"- pnl: {result.pnl:.6f}",
        f"- max_drawdown: {result.max_dd:.6f}",
    ]
    (cfg.output_dir / "diagnostic_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )


def _print_summary(signal_df: pd.DataFrame, result, cfg: LiveConfig) -> None:
    print("\n=== GEO-SYNC LIVE BACKTEST (EURUSD) ===")
    print(f"Bars:         {len(signal_df):,}")
    print(f"Window:       {cfg.window}")
    print(f"Initial cap:  ${cfg.initial_capital:,.2f}")
    print(f"P&L:          ${result.pnl:,.2f}")
    print(f"Return:       {result.pnl / cfg.initial_capital * 100:.2f}%")
    print(f"Max Drawdown: {result.max_dd:.2%}")
    print(f"Trades:       {result.trades}")
    print(f"Commission:   ${result.commission_cost:,.2f}")

    perf = getattr(result, "performance", None)
    if perf is not None and hasattr(perf, "sharpe"):
        print(f"Sharpe:       {perf.sharpe:.3f}")

    if result.equity_curve is not None:
        eq = pd.Series(result.equity_curve)
        print(f"Peak equity:  ${eq.max():,.2f}")
        print(f"Final equity: ${eq.iloc[-1]:,.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="EURUSD=X")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--r-long", type=float, default=0.70)
    parser.add_argument("--r-flat", type=float, default=0.30)
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument("--min-entry", type=float, default=0.0)
    parser.add_argument("--exit-threshold", type=float, default=0.50)
    parser.add_argument("--cache-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("backtest/output"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = LiveConfig(
        ticker=args.ticker,
        interval=args.interval,
        start=args.start,
        end=args.end,
        window=args.window,
        initial_capital=args.initial_capital,
        r_long_threshold=args.r_long,
        r_flat_threshold=args.r_flat,
        min_entry_signal=args.min_entry,
        min_confidence=args.min_confidence,
        exit_threshold=args.exit_threshold,
        cache_csv=args.cache_csv,
        output_dir=args.output_dir,
    )

    market_df = _download_prices(cfg)
    if len(market_df) <= cfg.window:
        raise ValueError(f"Not enough data: bars={len(market_df)} must be > window={cfg.window}")

    signal_df = _build_signal_series(market_df, cfg)
    result = _run_backtest(signal_df, cfg)
    _save_artifacts(signal_df, result, cfg)
    _write_diagnostic_report(signal_df, result, cfg)
    _print_summary(signal_df, result, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
