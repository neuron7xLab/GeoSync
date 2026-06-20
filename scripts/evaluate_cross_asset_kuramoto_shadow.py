"""Shadow evaluator — aggregates daily evidence + paper-state into the
live scoreboard and the drift/envelope engine.

Deterministic, offline, append-only on the scoreboard and on the
predictive envelope. No signal logic is re-run here; this module only
consumes the shadow runner's output and the spike paper-trader's
append-only evidence ledger.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SHADOW_DIR = REPO / "results" / "cross_asset_kuramoto" / "shadow_validation"
DAILY_ROOT = SHADOW_DIR / "daily"
LIVE_SCOREBOARD = SHADOW_DIR / "live_scoreboard.csv"
PREDICTIVE_ENVELOPE = SHADOW_DIR / "predictive_envelope.csv"
DRIFT_NOTE = SHADOW_DIR / "DRIFT_NOTE.md"
LIVE_STATE = SHADOW_DIR / "LIVE_STATE.json"
OPS_INCIDENTS = SHADOW_DIR / "operational_incidents.csv"
INCIDENT_LOGGER = REPO / "scripts" / "log_shadow_operational_incident.py"
PAPER_EQUITY = Path.home() / "spikes" / "cross_asset_sync_regime" / "paper_state" / "equity.csv"
DEMO_EQUITY = REPO / "results" / "cross_asset_kuramoto" / "demo" / "equity_curve.csv"

ENVELOPE_SEED = 20260422
ENVELOPE_BLOCK_LEN = 20
ENVELOPE_N_PATHS = 500
ENVELOPE_HORIZON_BARS = 90
BARS_PER_YEAR = 252.0
DEMO_MAX_DD = 0.1676  # from risk_metrics.csv
DD_GATE_MULT = 1.5  # G4 · 1.5x demo OOS max DD

SCOREBOARD_COLUMNS = (
    "eval_date",
    "live_bars_completed",
    "cumulative_net_return",
    "annualized_return_live",
    "annualized_vol_live",
    "sharpe_live",
    "max_dd_live",
    "turnover_ann_live",
    "hit_rate_live",
    "avg_win_live",
    "avg_loss_live",
    "cost_drag_bps_live",
    "benchmark_cum_return",
    "benchmark_sharpe_live",
    "relative_return_vs_benchmark",
    "predictive_envelope_quantile",
    "status_label",
    "gate_decision",
)

STATUS_VOCAB = {
    "BUILDING_SAMPLE",
    "WITHIN_EXPECTATION",
    "UNDERWATCH",
    "OUTSIDE_EXPECTATION",
    "OPERATIONALLY_UNSAFE",
}
GATE_VOCAB = {
    "CONTINUE_SHADOW",
    "ESCALATE_REVIEW",
    "NO_DEPLOY",
    "DEPLOYMENT_CANDIDATE_PENDING_OWNER",
}


class LedgerContinuityError(RuntimeError):
    """Raised when the live paper ledger has missing business-day bars."""


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _incident_logger_module() -> Any:
    spec = importlib.util.spec_from_file_location("shadow_incident_logger", INCIDENT_LOGGER)
    assert spec is not None, f"spec_from_file_location returned None for {INCIDENT_LOGGER}"
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _normalised_ledger_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    parsed = pd.to_datetime(df["date"], utc=True, errors="raise")
    dates = parsed.dt.tz_convert(None).dt.normalize()
    return pd.DatetimeIndex(sorted(pd.Series(dates).dropna().unique()))


def _missing_business_dates_within_window(
    df: pd.DataFrame,
    *,
    horizon_bars: int = ENVELOPE_HORIZON_BARS,
) -> list[str]:
    """Return missing business dates inside the observed 90-bar ledger window.

    The shadow panel uses ``panel_calendar_freq=B``. This check does not
    touch upstream panel construction; it only verifies that the append-only
    paper ledger is sequential over the last validation window. Weekends are
    excluded by construction, market holidays are not inferred here because
    the project calendar contract is business-day frequency, not an exchange
    holiday calendar.
    """
    if df.empty or "date" not in df.columns:
        return []

    actual = _normalised_ledger_dates(df)
    if len(actual) < 2:
        return []

    start = actual[max(0, len(actual) - horizon_bars)]
    end = actual[-1]
    expected = pd.bdate_range(start=start, end=end, freq="B")
    missing = expected.difference(actual)
    return [d.strftime("%Y-%m-%d") for d in missing]


def _record_missing_bar_and_fail_closed(
    missing_dates: list[str],
    *,
    incidents_path: Path = OPS_INCIDENTS,
    detected_at: str | None = None,
) -> None:
    detected = detected_at or _now_utc()
    first_missing = missing_dates[0]
    gap_size = len(missing_dates)
    logger = _incident_logger_module()
    logger.append_incident(
        timestamp=detected,
        scheduled_time=f"{first_missing}T22:00:00Z",
        actual_run_time=detected,
        bar_date=first_missing,
        scheduled_date=first_missing,
        detected_at=detected,
        gap_size_bars=gap_size,
        reason="missing_bar",
        recovery_status="fail_closed_continuity_broken "
        f"missing_dates={';'.join(missing_dates)}",
        path=incidents_path,
    )
    raise LedgerContinuityError(
        f"missing_bar continuity gap: scheduled_date={first_missing} "
        f"gap_size_bars={gap_size}"
    )


# --------------------------------------------------------------------- #
# Predictive envelope
# --------------------------------------------------------------------- #


def _demo_oos_log_returns() -> np.ndarray:
    df = pd.read_csv(DEMO_EQUITY)
    # equity_curve.csv has strategy_cumret on exp-scale; take log diff
    cum = df["strategy_cumret"].to_numpy()
    log = np.log(np.maximum(cum, 1e-12))
    r = np.diff(log)
    return r[np.isfinite(r)]


def _build_envelope(
    oos: np.ndarray,
    n_paths: int,
    block_len: int,
    horizon: int,
    seed: int,
) -> pd.DataFrame:
    """Block-bootstrap cumulative log-return paths.

    Returns a DataFrame with columns ``p05 p25 p50 p75 p95`` and one
    row per forward bar index (0-based) up to ``horizon``.
    """
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(horizon / block_len))
    n = len(oos)
    if n < block_len:
        raise ValueError(f"insufficient OOS bars ({n}) for block length {block_len}")
    sims = np.zeros((n_paths, horizon))
    for i in range(n_paths):
        pieces: list[np.ndarray] = []
        for _ in range(nb):
            start = int(rng.integers(0, n - block_len + 1))
            pieces.append(oos[start : start + block_len])
        path = np.concatenate(pieces)[:horizon]
        sims[i] = np.cumsum(path)
    q = np.quantile(sims, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)
    out = pd.DataFrame(
        {
            "forward_bar": np.arange(1, horizon + 1),
            "p05": q[0],
            "p25": q[1],
            "p50": q[2],
            "p75": q[3],
            "p95": q[4],
        }
    )
    return out


def _write_envelope_if_missing(envelope: pd.DataFrame) -> None:
    if PREDICTIVE_ENVELOPE.exists():
        # append-only: do not overwrite existing envelope
        return
    envelope.to_csv(PREDICTIVE_ENVELOPE, index=False, lineterminator="\n")


def _write_drift_note(oos_len: int) -> None:
    if DRIFT_NOTE.exists():
        return
    DRIFT_NOTE.write_text(
        "# Shadow validation · Drift / envelope method\n\n"
        "## Method\n\n"
        f"Block-bootstrap of the demo-ready OOS integrated log-return\n"
        f"series ({oos_len} bars; sourced read-only from "
        "`results/cross_asset_kuramoto/demo/equity_curve.csv`). No new\n"
        "backtest is run; the OOS stream is the already-validated one.\n\n"
        "## Parameters (locked)\n\n"
        f"- seed = `{ENVELOPE_SEED}`\n"
        f"- block length = **{ENVELOPE_BLOCK_LEN}** bars (≈ one\n"
        "  trading-month block; chosen once and frozen)\n"
        f"- number of bootstrap paths = **{ENVELOPE_N_PATHS}**\n"
        f"- forward horizon = **{ENVELOPE_HORIZON_BARS}** bars\n"
        "- quantiles reported: p05, p25, p50, p75, p95\n\n"
        "## Why this is descriptive, not optimisation\n\n"
        "1. No parameter of the strategy is changed, selected, or varied.\n"
        "2. The envelope is a non-parametric summary of the **validated**\n"
        "   OOS return distribution with its own short-range structure\n"
        "   preserved by block sampling.\n"
        "3. The envelope is used only to label the live cumulative path\n"
        "   relative to historical expectation. It has no feedback loop\n"
        "   into signal generation.\n\n"
        "## Reproducibility\n\n"
        "Given the same demo OOS stream, the same seed, the same block\n"
        "length, and the same number of paths, the envelope is\n"
        "bit-identical across runs. The test suite asserts this.\n"
    )


# --------------------------------------------------------------------- #
# Live scoreboard
# --------------------------------------------------------------------- #


def _load_live_ledger(
    paper_equity: Path,
    *,
    incidents_path: Path = OPS_INCIDENTS,
    detected_at: str | None = None,
) -> pd.DataFrame:
    """Read the spike paper-trader's append-only equity.csv ledger.

    Fail-closed when the last 90-bar business-day window is not sequential.
    A missing date means shadow evidence is discontinuous; continuing would
    silently promote a corrupt validation horizon.
    """
    if not paper_equity.exists():
        return pd.DataFrame()
    df = pd.read_csv(paper_equity, parse_dates=["date"])
    if df.empty:
        return df
    # Deduplicate by date: keep the LAST row per date (idempotent re-ticks)
    df = df.sort_values(["date", "day_n"]).drop_duplicates("date", keep="last")
    df = df.reset_index(drop=True)
    missing = _missing_business_dates_within_window(df)
    if missing:
        _record_missing_bar_and_fail_closed(
            missing,
            incidents_path=incidents_path,
            detected_at=detected_at,
        )
    return df


def _load_pipeline_status_for_latest() -> dict[str, Any]:
    """Pull the most recent daily run's pipeline_status.csv."""
    if not DAILY_ROOT.is_dir():
        return {}
    dirs = sorted([p for p in DAILY_ROOT.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not dirs:
        return {}
    ps = dirs[-1] / "pipeline_status.csv"
    if not ps.is_file():
        return {}
    return pd.read_csv(ps).iloc[0].to_dict()


def _compute_live_metrics(live: pd.DataFrame) -> dict[str, Any]:
    empty_metrics: dict[str, Any] = {
        "live_bars_completed": 0,
        "cumulative_net_return": 0.0,
        "annualized_return_live": float("nan"),
        "annualized_vol_live": float("nan"),
        "sharpe_live": float("nan"),
        "max_dd_live": float("nan"),
        "turnover_ann_live": float("nan"),
        "hit_rate_live": float("nan"),
        "avg_win_live": float("nan"),
        "avg_loss_live": float("nan"),
        "cost_drag_bps_live": float("nan"),
    }
    # Guard against an empty or schema-less ledger (paper-state absent on
    # CI runners, or the spike has not yet written its first tick).
    if live.empty or "net_ret" not in live.columns:
        return empty_metrics
    r = live["net_ret"].astype(float).to_numpy()
    n = int(len(r))
    if n == 0:
        return empty_metrics
    ann_ret = float(np.mean(r) * BARS_PER_YEAR)
    ann_vol = float(np.std(r, ddof=1) * np.sqrt(BARS_PER_YEAR)) if n > 1 else float("nan")
    sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else float("nan")
    eq = live["equity"].astype(float).to_numpy()
    peak = np.maximum.accumulate(eq)
    dd = 1.0 - eq / peak
    max_dd = float(dd.max())
    tov = live["turnover"].astype(float).to_numpy()
    tov_ann = float(np.mean(tov) * BARS_PER_YEAR) if n > 0 else float("nan")
    wins = r[r > 0]
    losses = r[r < 0]
    hit = float((r > 0).mean())
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    cost = live["cost"].astype(float).to_numpy()
    cost_drag_ann_bps = float(np.mean(cost) * BARS_PER_YEAR * 10_000.0)
    return {
        "live_bars_completed": n,
        "cumulative_net_return": float(np.exp(r.sum()) - 1.0),
        "annualized_return_live": ann_ret,
        "annualized_vol_live": ann_vol,
        "sharpe_live": sharpe,
        "max_dd_live": max_dd,
        "turnover_ann_live": tov_ann,
        "hit_rate_live": hit,
        "avg_win_live": avg_win,
        "avg_loss_live": avg_loss,
        "cost_drag_bps_live": cost_drag_ann_bps,
    }


def _envelope_position(live_cum_log: float, bar_n: int, envelope: pd.DataFrame) -> str:
    """Label live cumulative log-return against the envelope at ``bar_n``."""
    row = envelope[envelope["forward_bar"] == bar_n]
    if row.empty:
        return "out_of_horizon"
    p5 = float(row["p05"].iloc[0])
    p25 = float(row["p25"].iloc[0])
    p75 = float(row["p75"].iloc[0])
    p95 = float(row["p95"].iloc[0])
    if live_cum_log < p5:
        return "below_p05"
    if live_cum_log < p25:
        return "p05_p25"
    if live_cum_log <= p75:
        return "p25_p75"
    if live_cum_log <= p95:
        return "p75_p95"
    return "above_p95"


def _benchmark_cum(live: pd.DataFrame) -> tuple[float, float]:
    """Equal-weight buy-and-hold benchmark cum-return and Sharpe from equity ledger.

    The paper-state equity.csv includes a ``btc_equity`` column as the
    spike's built-in BTC benchmark; we use that (not a re-computation)
    to stay read-only.
    """
    if "btc_equity" not in live.columns or live.empty:
        return float("nan"), float("nan")
    btc_eq = live["btc_equity"].astype(float).to_numpy()
    btc_cum = float(btc_eq[-1] - 1.0)
    btc_ret = np.diff(np.log(np.maximum(btc_eq, 1e-12)))
    if len(btc_ret) < 2:
        return btc_cum, float("nan")
    mu = btc_ret.mean() * BARS_PER_YEAR
    sd = btc_ret.std(ddof=1) * np.sqrt(BARS_PER_YEAR)
    sh = mu / sd if sd > 0 else float("nan")
    return btc_cum, sh


# --------------------------------------------------------------------- #
# Gate engine
# --------------------------------------------------------------------- #


def _latest_pipeline_unsafe() -> bool:
    ps = _load_pipeline_status_for_latest()
    return bool(ps.get("operationally_unsafe", False))


def _any_invariant_fail() -> bool:
    if not DAILY_ROOT.is_dir():
        return False
    for d in DAILY_ROOT.iterdir():
        inv = d / "invariant_status.csv"
        if not inv.exists():
            continue
        df = pd.read_csv(inv)
        if (df["status"].astype(str).str.contains("FAIL")).any():
            return True
    return False


def _envelope_sub_p05_streak(board: pd.DataFrame) -> int:
    """Trailing run of evaluation rows where envelope position is below_p05."""
    if board.empty:
        return 0
    tail = board["predictive_envelope_quantile"].astype(str).tolist()
    n = 0
    for s in reversed(tail):
        if s == "below_p05":
            n += 1
        else:
            break
    return n


def _decide_status_and_gate(
    bars: int,
    metrics: dict[str, Any],
    env_pos: str,
    op_unsafe: bool,
    inv_fail: bool,
    sub_p05_streak: int,
) -> tuple[str, str]:
    # Hard operational short-circuits
    if op_unsafe or inv_fail:
        return "OPERATIONALLY_UNSAFE", "ESCALATE_REVIEW"

    # Risk gate G4
    dd_live = metrics.get("max_dd_live") or 0.0
    if np.isfinite(dd_live) and dd_live > DEMO_MAX_DD * DD_GATE_MULT:
        return "OUTSIDE_EXPECTATION", "ESCALATE_REVIEW"

    # Drift gate G3
    if env_pos == "below_p05" and sub_p05_streak >= 20:
        if bars >= 60:
            return "OUTSIDE_EXPECTATION", "NO_DEPLOY"
        return "OUTSIDE_EXPECTATION", "ESCALATE_REVIEW"

    if bars < 20:
        return "BUILDING_SAMPLE", "CONTINUE_SHADOW"

    # 90-bar truth gate
    if bars >= 90:
        if env_pos in {"p25_p75", "p75_p95"} and metrics.get("sharpe_live", float("nan")) > 0:
            return "WITHIN_EXPECTATION", "DEPLOYMENT_CANDIDATE_PENDING_OWNER"
        if env_pos in {"p05_p25"}:
            return "UNDERWATCH", "CONTINUE_SHADOW"
        return "OUTSIDE_EXPECTATION", "NO_DEPLOY"

    # Interior bars
    if env_pos == "below_p05":
        return "OUTSIDE_EXPECTATION", "ESCALATE_REVIEW"
    if env_pos == "p05_p25":
        return "UNDERWATCH", "CONTINUE_SHADOW"
    return "WITHIN_EXPECTATION", "CONTINUE_SHADOW"


# --------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------- #


def _append_scoreboard(row: dict[str, Any]) -> None:
    new = not LIVE_SCOREBOARD.exists()
    with LIVE_SCOREBOARD.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SCOREBOARD_COLUMNS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in SCOREBOARD_COLUMNS})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Shadow validation evaluator")
    ap.add_argument("--paper-equity", type=Path, default=PAPER_EQUITY)
    ap.add_argument("--rebuild-envelope", action="store_true")
    args = ap.parse_args(argv)

    SHADOW_DIR.mkdir(parents=True, exist_ok=True)

    # Predictive envelope (append-only; regenerated only on --rebuild-envelope)
    oos = _demo_oos_log_returns()
    envelope = _build_envelope(
        oos,
        n_paths=ENVELOPE_N_PATHS,
        block_len=ENVELOPE_BLOCK_LEN,
        horizon=ENVELOPE_HORIZON_BARS,
        seed=ENVELOPE_SEED,
    )
    if args.rebuild_envelope and PREDICTIVE_ENVELOPE.exists():
        PREDICTIVE_ENVELOPE.unlink()
    _write_envelope_if_missing(envelope)
    _write_drift_note(len(oos))

    # Live ledger from spike paper-state
    live = _load_live_ledger(args.paper_equity)
    metrics = _compute_live_metrics(live)
    bars = metrics["live_bars_completed"]

    bench_cum, bench_sh = _benchmark_cum(live)

    # Envelope position at the current live bar
    live_cum_log = float(np.log(1.0 + metrics["cumulative_net_return"])) if bars > 0 else 0.0
    env_pos = _envelope_position(live_cum_log, bars, envelope) if bars > 0 else "n/a"

    # Existing scoreboard (to measure below-p05 streak)
    board = (
        pd.read_csv(LIVE_SCOREBOARD)
        if LIVE_SCOREBOARD.exists()
        else pd.DataFrame(columns=list(SCOREBOARD_COLUMNS))
    )
    streak = _envelope_sub_p05_streak(board)
    if env_pos == "below_p05":
        streak += 1
    else:
        streak = 0

    op_unsafe = _latest_pipeline_unsafe()
    inv_fail = _any_invariant_fail()
    status, gate = _decide_status_and_gate(
        bars=bars,
        metrics=metrics,
        env_pos=env_pos,
        op_unsafe=op_unsafe,
        inv_fail=inv_fail,
        sub_p05_streak=streak,
    )

    assert status in STATUS_VOCAB, f"status {status!r} not in {sorted(STATUS_VOCAB)}"
    assert gate in GATE_VOCAB, f"gate {gate!r} not in {sorted(GATE_VOCAB)}"

    eval_row = {
        "eval_date": _now_utc()[:10],
        "live_bars_completed": bars,
        "cumulative_net_return": round(metrics["cumulative_net_return"], 6),
        "annualized_return_live": (
            round(metrics["annualized_return_live"], 6)
            if np.isfinite(metrics["annualized_return_live"])
            else ""
        ),
        "annualized_vol_live": (
            round(metrics["annualized_vol_live"], 6)
            if np.isfinite(metrics["annualized_vol_live"])
            else ""
        ),
        "sharpe_live": (
            round(metrics["sharpe_live"], 4) if np.isfinite(metrics["sharpe_live"]) else ""
        ),
        "max_dd_live": (
            round(metrics["max_dd_live"], 6) if np.isfinite(metrics["max_dd_live"]) else ""
        ),
        "turnover_ann_live": (
            round(metrics["turnover_ann_live"], 4)
            if np.isfinite(metrics["turnover_ann_live"])
            else ""
        ),
        "hit_rate_live": (
            round(metrics["hit_rate_live"], 4) if np.isfinite(metrics["hit_rate_live"]) else ""
        ),
        "avg_win_live": (
            round(metrics["avg_win_live"], 6) if np.isfinite(metrics["avg_win_live"]) else ""
        ),
        "avg_loss_live": (
            round(metrics["avg_loss_live"], 6) if np.isfinite(metrics["avg_loss_live"]) else ""
        ),
        "cost_drag_bps_live": (
            round(metrics["cost_drag_bps_live"], 2)
            if np.isfinite(metrics["cost_drag_bps_live"])
            else ""
        ),
        "benchmark_cum_return": round(bench_cum, 6) if np.isfinite(bench_cum) else "",
        "benchmark_sharpe_live": round(bench_sh, 4) if np.isfinite(bench_sh) else "",
        "relative_return_vs_benchmark": (
            round(metrics["cumulative_net_return"] - bench_cum, 6) if np.isfinite(bench_cum) else ""
        ),
        "predictive_envelope_quantile": env_pos,
        "status_label": status,
        "gate_decision": gate,
    }
    _append_scoreboard(eval_row)

    # Update LIVE_STATE
    LIVE_STATE.write_text(
        json.dumps(
            {
                "last_eval_utc": _now_utc(),
                "live_bars": bars,
                "cumulative_net_return": round(metrics["cumulative_net_return"], 6),
                "sharpe_live": (
                    round(metrics["sharpe_live"], 4)
                    if np.isfinite(metrics["sharpe_live"])
                    else None
                ),
                "max_dd_live": (
                    round(metrics["max_dd_live"], 6)
                    if np.isfinite(metrics["max_dd_live"])
                    else None
                ),
                "predictive_envelope_quantile": env_pos,
                "status_label": status,
                "gate_decision": gate,
                "operationally_unsafe": op_unsafe,
                "any_invariant_fail": inv_fail,
            },
            indent=2,
        )
    )

    print(json.dumps({"eval": eval_row}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
