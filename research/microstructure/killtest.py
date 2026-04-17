"""Fail-fast kill test for GeoSync/Ricci on real L2 substrate.

Single-file, minimal scope: load L2 parquet shards → derive four core features
(OFI, queue imbalance, cross-sectional correlation graph, Forman-Ricci κ_min)
→ test against one target (3-min forward mid-price log return) → compare to
three baselines (plain mid-return, realized vol, plain own-OFI) → run null
tests (permutation + circular shift) → orthogonality residual → emit binary
VERDICT.

Gate thresholds (conservative, fail-fast first pass):
    IC_signal >= 0.03 (Spearman, pooled across symbols)
    IC_signal > max(IC_baselines)
    Residual IC (after OLS on baselines) > 0 with permutation p < 0.05
    Stable lead: IC > 0 at horizons 1, 2, 3, 4, 5 minutes (all)

If any gate fails → VERDICT = KILL. Otherwise → PROCEED.

Seed = 42 everywhere. Determinism contract (INV-HPC1): same inputs → same
verdict bit-identical.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.typing import NDArray
from scipy.stats import spearmanr

from core.physics.forman_ricci import FormanRicciCurvature
from research.microstructure.l2_schema import N_LEVELS, l2_schema

SEED: int = 42
_GRID_MS: int = 1_000
_TARGET_HORIZONS_SEC: tuple[int, ...] = (60, 120, 180, 240, 300)
_PRIMARY_HORIZON_SEC: int = 180
_CORR_WINDOW_SEC: int = 300
_CORR_STEP_SEC: int = 30
_RICCI_THRESHOLD: float = 0.5
_IC_GATE: float = 0.03
_PERM_TRIALS: int = 500
_PERM_PVALUE_GATE: float = 0.05


@dataclass(frozen=True)
class FeatureFrame:
    """Aligned per-symbol feature panels on a fixed 1-second grid."""

    timestamps_ms: NDArray[np.int64]
    symbols: tuple[str, ...]
    mid: NDArray[np.float64]
    ofi: NDArray[np.float64]
    queue_imbalance: NDArray[np.float64]

    @property
    def n_rows(self) -> int:
        return int(self.mid.shape[0])

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)


@dataclass
class GateVerdict:
    verdict: str
    reasons: list[str]
    ic_signal: float
    ic_baselines: dict[str, float]
    residual_ic: float
    residual_ic_pvalue: float
    horizon_ic: dict[int, float]
    null_test_pvalues: dict[str, float]
    n_samples: int
    n_symbols: int
    seed: int = SEED
    metadata: dict[str, Any] = field(default_factory=dict)


def _load_parquets(data_dir: Path, symbols: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Load and concat all parquet shards per symbol under `data_dir`."""
    schema = l2_schema()
    frames: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        shards = sorted(data_dir.glob(f"{sym}_hour_*.parquet"))
        if not shards:
            continue
        tables = [pq.read_table(p, schema=schema) for p in shards]
        df = pd.concat([t.to_pandas() for t in tables], ignore_index=True)
        df = df.drop_duplicates(subset=["ts_event", "update_id"])
        df = df.sort_values("ts_event").reset_index(drop=True)
        frames[sym] = df
    return frames


def _to_grid(df: pd.DataFrame, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Downsample to 1-second grid: last observation per second."""
    idx = pd.to_datetime(df["ts_event"], unit="ms", utc=True)
    panel = df.set_index(idx)
    grid_idx = pd.date_range(
        start=pd.to_datetime(start_ms, unit="ms", utc=True),
        end=pd.to_datetime(end_ms, unit="ms", utc=True),
        freq="1s",
    )
    resampled = panel.resample("1s").last()
    resampled = resampled.reindex(grid_idx).ffill(limit=30)
    return resampled


def _compute_mid(df: pd.DataFrame) -> pd.Series:
    return (df["bid_px_1"] + df["ask_px_1"]) * 0.5


def _compute_ofi(df: pd.DataFrame) -> pd.Series:
    """Order Flow Imbalance at L1 (Cont-Kukanov-Stoikov, 2014).

    e_n = ΔB · 1[bid_px_n >= bid_px_{n-1}] - bid_sz_{n-1} · 1[bid_px_n < bid_px_{n-1}]
        - ΔA · 1[ask_px_n <= ask_px_{n-1}] + ask_sz_{n-1} · 1[ask_px_n > ask_px_{n-1}]

    where ΔB = bid_sz_n - bid_sz_{n-1} when bid_px_n == bid_px_{n-1}, else bid_sz_n;
    symmetric for asks.
    """
    bid_px = df["bid_px_1"].astype(float).to_numpy()
    bid_sz = df["bid_sz_1"].astype(float).to_numpy()
    ask_px = df["ask_px_1"].astype(float).to_numpy()
    ask_sz = df["ask_sz_1"].astype(float).to_numpy()

    n = len(bid_px)
    ofi = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if bid_px[i] > bid_px[i - 1]:
            bid_term = bid_sz[i]
        elif bid_px[i] < bid_px[i - 1]:
            bid_term = -bid_sz[i - 1]
        else:
            bid_term = bid_sz[i] - bid_sz[i - 1]

        if ask_px[i] < ask_px[i - 1]:
            ask_term = ask_sz[i]
        elif ask_px[i] > ask_px[i - 1]:
            ask_term = -ask_sz[i - 1]
        else:
            ask_term = ask_sz[i] - ask_sz[i - 1]

        ofi[i] = bid_term - ask_term
    return pd.Series(ofi, index=df.index, name="ofi")


def _compute_queue_imbalance(df: pd.DataFrame) -> pd.Series:
    bid_sz = df["bid_sz_1"].astype(float)
    ask_sz = df["ask_sz_1"].astype(float)
    denom = bid_sz + ask_sz
    qi = (bid_sz - ask_sz) / denom.replace(0.0, np.nan)
    return qi.fillna(0.0).clip(-1.0, 1.0).rename("qi")


def build_feature_frame(
    frames: dict[str, pd.DataFrame],
    symbols: tuple[str, ...],
) -> FeatureFrame:
    if not frames:
        raise ValueError("no symbol frames loaded — cannot build feature frame")
    symbols_present = tuple(s for s in symbols if s in frames)
    if len(symbols_present) < 3:
        raise ValueError(
            f"need at least 3 symbols with data, got {len(symbols_present)}: {symbols_present}"
        )

    start_ms = max(int(frames[s]["ts_event"].iloc[0]) for s in symbols_present)
    end_ms = min(int(frames[s]["ts_event"].iloc[-1]) for s in symbols_present)
    if end_ms - start_ms < _CORR_WINDOW_SEC * 1000 * 4:
        raise ValueError(
            f"overlap too short: {(end_ms - start_ms) / 1000:.1f}s "
            f"(need >= {_CORR_WINDOW_SEC * 4}s)"
        )

    grid_panels: dict[str, pd.DataFrame] = {
        s: _to_grid(frames[s], start_ms, end_ms) for s in symbols_present
    }

    ref_idx = grid_panels[symbols_present[0]].index
    n_rows = len(ref_idx)
    n_sym = len(symbols_present)

    mid = np.full((n_rows, n_sym), np.nan, dtype=np.float64)
    ofi = np.zeros((n_rows, n_sym), dtype=np.float64)
    qi = np.zeros((n_rows, n_sym), dtype=np.float64)

    for k, sym in enumerate(symbols_present):
        panel = grid_panels[sym]
        mid[:, k] = _compute_mid(panel).to_numpy()
        ofi[:, k] = _compute_ofi(panel).to_numpy()
        qi[:, k] = _compute_queue_imbalance(panel).to_numpy()

    mask = np.isfinite(mid).all(axis=1)
    mid = mid[mask]
    ofi = ofi[mask]
    qi = qi[mask]
    timestamps_ms = (ref_idx[mask].astype("int64") // 1_000_000).to_numpy()

    return FeatureFrame(
        timestamps_ms=timestamps_ms,
        symbols=symbols_present,
        mid=mid,
        ofi=ofi,
        queue_imbalance=qi,
    )


def cross_sectional_ricci_signal(
    ofi_panel: NDArray[np.float64],
    window: int = _CORR_WINDOW_SEC,
    step: int = _CORR_STEP_SEC,
    threshold: float = _RICCI_THRESHOLD,
) -> NDArray[np.float64]:
    """Rolling κ_min of Forman-Ricci on OFI cross-sectional correlation graph.

    Returns array of shape (T,) aligned to input rows; rows before the first
    full window are NaN.
    """
    n, m = ofi_panel.shape
    if m < 3:
        return np.full(n, np.nan)
    fr = FormanRicciCurvature(threshold=threshold)
    out = np.full(n, np.nan, dtype=np.float64)
    for end in range(window, n, step):
        block = ofi_panel[end - window : end]
        if not np.all(np.isfinite(block)):
            continue
        std = block.std(axis=0)
        if np.any(std < 1e-12):
            continue
        corr_raw = np.corrcoef(block.T)
        corr = np.nan_to_num(
            np.asarray(corr_raw, dtype=np.float64), nan=0.0, posinf=1.0, neginf=-1.0
        )
        result = fr.compute_from_correlation(corr)
        fill_to = min(end + step, n)
        out[end:fill_to] = result.kappa_min
    return out


def _forward_log_return(mid_panel: NDArray[np.float64], horizon_rows: int) -> NDArray[np.float64]:
    log_mid = np.log(mid_panel)
    fwd = np.full_like(log_mid, np.nan)
    if horizon_rows >= log_mid.shape[0]:
        return fwd
    fwd[:-horizon_rows] = log_mid[horizon_rows:] - log_mid[:-horizon_rows]
    return fwd


def _pooled_ic(signal_panel: NDArray[np.float64], target_panel: NDArray[np.float64]) -> float:
    s_flat = signal_panel.ravel()
    t_flat = target_panel.ravel()
    mask = np.isfinite(s_flat) & np.isfinite(t_flat)
    if mask.sum() < 50:
        return float("nan")
    s = s_flat[mask]
    t = t_flat[mask]
    if float(np.std(s)) == 0.0 or float(np.std(t)) == 0.0:
        return float("nan")
    rho, _ = spearmanr(s, t)
    return float(rho) if np.isfinite(rho) else float("nan")


def _permutation_pvalue(
    signal_panel: NDArray[np.float64],
    target_panel: NDArray[np.float64],
    observed_ic: float,
    *,
    trials: int = _PERM_TRIALS,
    seed: int = SEED,
    mode: str = "shuffle",
) -> float:
    rng = np.random.default_rng(seed)
    n_rows = signal_panel.shape[0]
    count = 0
    trials_done = 0
    for _ in range(trials):
        if mode == "shuffle":
            perm = rng.permutation(n_rows)
            shuffled = signal_panel[perm]
        elif mode == "circular":
            shift = int(rng.integers(1, n_rows - 1))
            shuffled = np.roll(signal_panel, shift, axis=0)
        else:
            raise ValueError(f"unknown mode: {mode}")
        ic = _pooled_ic(shuffled, target_panel)
        if not np.isfinite(ic):
            continue
        if abs(ic) >= abs(observed_ic):
            count += 1
        trials_done += 1
    if trials_done == 0:
        return 1.0
    return (count + 1) / (trials_done + 1)


def _residualize(
    signal_panel: NDArray[np.float64],
    baselines_panel: dict[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    """OLS residuals of signal ~ baselines, preserving NaN positions."""
    s_flat = signal_panel.ravel().astype(np.float64)
    b_cols = [b.ravel().astype(np.float64) for b in baselines_panel.values()]
    stacked = np.column_stack(b_cols) if b_cols else np.zeros((s_flat.shape[0], 0))
    mask = (
        np.isfinite(s_flat) & np.all(np.isfinite(stacked), axis=1)
        if b_cols
        else np.isfinite(s_flat)
    )
    residual = np.full_like(s_flat, np.nan)
    if mask.sum() < 50 or stacked.shape[1] == 0:
        return residual.reshape(signal_panel.shape)
    X = np.column_stack([np.ones(mask.sum()), stacked[mask]])
    y = s_flat[mask]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    residual[mask] = y - fitted
    return residual.reshape(signal_panel.shape)


def run_killtest(
    features: FeatureFrame,
    *,
    primary_horizon_sec: int = _PRIMARY_HORIZON_SEC,
    horizons_sec: tuple[int, ...] = _TARGET_HORIZONS_SEC,
    ic_gate: float = _IC_GATE,
    pvalue_gate: float = _PERM_PVALUE_GATE,
    seed: int = SEED,
) -> GateVerdict:
    """Execute the full fail-fast gate and emit a binary verdict."""
    ricci_signal_1d = cross_sectional_ricci_signal(features.ofi)
    ricci_panel = np.repeat(ricci_signal_1d[:, None], features.n_symbols, axis=1)
    target = _forward_log_return(features.mid, primary_horizon_sec)

    ic_signal = _pooled_ic(ricci_panel, target)

    ret_1s = np.vstack(
        [
            np.zeros((1, features.n_symbols)),
            np.diff(np.log(features.mid), axis=0),
        ]
    )
    realized_vol = pd.DataFrame(ret_1s).rolling(window=60, min_periods=30).std().to_numpy()
    baselines: dict[str, NDArray[np.float64]] = {
        "plain_return": ret_1s,
        "realized_vol": realized_vol,
        "plain_ofi": features.ofi,
    }

    ic_baselines: dict[str, float] = {}
    for name, b in baselines.items():
        ic_baselines[name] = _pooled_ic(b, target)

    residual_signal = _residualize(ricci_panel, baselines)
    residual_ic = _pooled_ic(residual_signal, target)
    residual_pvalue = _permutation_pvalue(
        residual_signal, target, residual_ic, seed=seed, mode="shuffle"
    )

    null_pvalues: dict[str, float] = {
        "permutation_shuffle": _permutation_pvalue(
            ricci_panel, target, ic_signal, seed=seed, mode="shuffle"
        ),
        "circular_shift": _permutation_pvalue(
            ricci_panel, target, ic_signal, seed=seed + 1, mode="circular"
        ),
    }

    horizon_ic: dict[int, float] = {}
    for h in horizons_sec:
        tgt_h = _forward_log_return(features.mid, h)
        horizon_ic[h] = _pooled_ic(ricci_panel, tgt_h)

    reasons: list[str] = []
    if not np.isfinite(ic_signal) or ic_signal < ic_gate:
        reasons.append(f"IC_signal={ic_signal:.4f} < gate={ic_gate:.4f}")
    finite_baselines = [v for v in ic_baselines.values() if np.isfinite(v)]
    best_baseline = max(finite_baselines) if finite_baselines else 0.0
    if np.isfinite(ic_signal) and ic_signal <= best_baseline:
        reasons.append(f"IC_signal={ic_signal:.4f} does not beat best baseline={best_baseline:.4f}")
    if not np.isfinite(residual_ic) or residual_ic <= 0.0:
        reasons.append(f"residual_IC={residual_ic:.4f} <= 0 (no orthogonal edge)")
    if residual_pvalue > pvalue_gate:
        reasons.append(f"residual permutation p={residual_pvalue:.3f} > gate={pvalue_gate:.3f}")
    unstable = [h for h, ic in horizon_ic.items() if not np.isfinite(ic) or ic <= 0.0]
    if unstable:
        reasons.append(f"unstable lead: non-positive IC at horizons {unstable}")
    for null_name, p in null_pvalues.items():
        if p > pvalue_gate:
            reasons.append(f"{null_name} p={p:.3f} > gate={pvalue_gate:.3f}")

    verdict = "PROCEED" if not reasons else "KILL"

    return GateVerdict(
        verdict=verdict,
        reasons=reasons,
        ic_signal=float(ic_signal) if np.isfinite(ic_signal) else float("nan"),
        ic_baselines={
            k: float(v) if np.isfinite(v) else float("nan") for k, v in ic_baselines.items()
        },
        residual_ic=float(residual_ic) if np.isfinite(residual_ic) else float("nan"),
        residual_ic_pvalue=float(residual_pvalue),
        horizon_ic={
            int(h): float(ic) if np.isfinite(ic) else float("nan") for h, ic in horizon_ic.items()
        },
        null_test_pvalues={k: float(v) for k, v in null_pvalues.items()},
        n_samples=int(features.n_rows),
        n_symbols=int(features.n_symbols),
        seed=seed,
        metadata={
            "primary_horizon_sec": primary_horizon_sec,
            "ic_gate": ic_gate,
            "pvalue_gate": pvalue_gate,
            "corr_window_sec": _CORR_WINDOW_SEC,
            "corr_step_sec": _CORR_STEP_SEC,
            "ricci_threshold": _RICCI_THRESHOLD,
            "n_levels": N_LEVELS,
        },
    )


def verdict_to_json(verdict: GateVerdict) -> str:
    return json.dumps(asdict(verdict), indent=2, sort_keys=True, default=str)
