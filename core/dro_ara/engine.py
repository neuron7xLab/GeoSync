# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""DRO-ARA v7 — Deterministic Recursive Observer + Action Result Acceptor.

Measures statistical regime of a price series via Hurst (H) on log-returns (DFA-1),
confirms stationarity via lag-augmented ADF on log-returns (AIC lag selection,
Ng & Perron 2001), and emits a deterministic regime + trading signal through a
bounded ARA feedback loop. Both statistical tests operate on the same transform
(∆ log price) — the convention was aligned in PR #345 (RFC-stationarity).

Public invariants (never relaxed):

* ``gamma = 2*H + 1``   — derived from H, never assigned independently.
* ``rs = max(0, 1 - |gamma - 1|)`` — risk scalar ∈ [0, 1].
* Regime is INVALID when ADF rejects stationarity OR DFA R² < 0.90.
* Signal LONG requires CRITICAL regime AND rs > 0.33 AND trend ∈ {CONVERGING,
  STABLE}.
* Degenerate input (NaN/Inf, constant, wrong rank, too short) raises
  ``ValueError`` — fail-closed, never silent.

References:
    Dickey & Fuller (1979); Ng & Perron (2001); Peng et al. (1994, DFA);
    MacKinnon (1994, ADF critical values).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Iterator, NamedTuple

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Regime",
    "Signal",
    "State",
    "derive_gamma",
    "risk_scalar",
    "classify",
    "geosync_observe",
    "MAX_DEPTH",
    "MIN_WINDOW",
    "R2_MIN",
    "EPSILON_H",
    "STABLE_RUNS",
    "H_CRITICAL",
    "H_DRIFT",
    "ADF_CV_5PCT",
    "ADF_MAX_LAGS",
    "RS_LONG_THRESH",
]

MAX_DEPTH: Final[int] = 32
MIN_WINDOW: Final[int] = 64
MIN_BOXES: Final[int] = 4
R2_MIN: Final[float] = 0.90
EPSILON_H: Final[float] = 0.02
STABLE_RUNS: Final[int] = 3
H_CRITICAL: Final[float] = 0.45
H_DRIFT: Final[float] = 0.55
ADF_CV_5PCT: Final[float] = -2.86
ADF_MAX_LAGS: Final[int] = 4
RS_LONG_THRESH: Final[float] = 0.33


class Regime(str, Enum):
    CRITICAL = "CRITICAL"
    TRANSITION = "TRANSITION"
    DRIFT = "DRIFT"
    INVALID = "INVALID"


class Signal(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"
    REDUCE = "REDUCE"


def _adf_stationary(x: NDArray[np.float64]) -> bool:
    """Lag-augmented ADF test with AIC lag selection.

    Model: Δxₜ = β·xₜ₋₁ + Σ γₖ·Δxₜ₋ₖ + εₜ. H₀: β = 0 (unit root). Reject H₀ when
    t_β < ADF_CV_5PCT → series declared stationary. Returns True when rejected.
    """
    if len(x) < 20:
        return True
    dy = np.diff(x)
    best_aic = np.inf
    best_t = 0.0

    for k in range(0, ADF_MAX_LAGS + 1):
        min_idx = k + 1
        y_ = dy[min_idx:]
        xl_ = x[min_idx:-1] - x[min_idx:-1].mean()

        if k == 0:
            Z = xl_.reshape(-1, 1)
        else:
            lags = np.column_stack([dy[min_idx - j : -j] for j in range(1, k + 1)])
            Z = np.column_stack([xl_, lags])

        ZtZ = Z.T @ Z + np.eye(Z.shape[1]) * 1e-10
        coef = np.linalg.solve(ZtZ, Z.T @ y_)
        res = y_ - Z @ coef
        s2 = float(np.sum(res**2) / max(len(res) - Z.shape[1], 1))

        aic = len(y_) * np.log(s2 + 1e-12) + 2 * Z.shape[1]
        if aic < best_aic:
            best_aic = aic
            se_beta = float(np.sqrt(s2 * np.linalg.inv(ZtZ)[0, 0]))
            best_t = float(coef[0] / (se_beta + 1e-12))

    return bool(best_t < ADF_CV_5PCT)


def _validate(x: NDArray[np.float64] | np.ndarray, min_len: int) -> NDArray[np.float64]:
    arr: NDArray[np.float64] = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"1-D required, got {arr.shape}")
    if len(arr) < min_len:
        raise ValueError(f"need ≥{min_len}, got {len(arr)}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("NaN/Inf in input")
    if np.all(arr == arr[0]):
        raise ValueError("constant series")
    return arr


def _hurst_dfa(x: NDArray[np.float64]) -> tuple[float, float]:
    """DFA-1 on log-returns. Returns (H, R²).

    Log-returns are detrended by construction (linear drift in prices → constant
    in log-returns → zero-mean after demeaning), so the H estimate is unbiased
    across drift magnitudes.
    """
    r = np.diff(np.log(np.abs(x) + 1e-12))
    y = np.cumsum(r - r.mean())
    N = len(y)
    min_box = max(8, N // 64)
    max_box = N // MIN_BOXES
    if max_box < min_box:
        return 0.5, 0.0
    sizes = np.unique(np.round(np.geomspace(min_box, max_box, 16)).astype(int))
    sizes = sizes[(sizes >= min_box) & (sizes <= max_box)]
    if len(sizes) < 3:
        return 0.5, 0.0

    Fv_list: list[float] = []
    sv_list: list[int] = []
    for n in sizes:
        nb = N // n
        if nb < MIN_BOXES:
            continue
        seg = y[: nb * n].reshape(nb, n)
        t_ = np.arange(n, dtype=float)
        t_ -= t_.mean()
        s_ = seg - seg.mean(axis=1, keepdims=True)
        sl = (s_ @ t_) / (t_ @ t_)
        res = s_ - sl[:, None] * t_[None, :]
        Fn = float(np.sqrt(np.mean(res**2)))
        if Fn > 0 and np.isfinite(Fn):
            Fv_list.append(Fn)
            sv_list.append(int(n))

    if len(Fv_list) < 3:
        return 0.5, 0.0
    Fv = np.asarray(Fv_list, dtype=np.float64)
    if not np.all(Fv > 0):
        return 0.5, 0.0

    ln = np.log(np.asarray(sv_list, dtype=np.float64))
    lF = np.log(Fv)
    H_raw, b = np.polyfit(ln, lF, 1)
    ss_res = float(np.sum((lF - (H_raw * ln + b)) ** 2))
    ss_tot = float(np.sum((lF - lF.mean()) ** 2))
    r2 = float(np.clip(1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0, 0.0, 1.0))
    return float(np.clip(H_raw, 0.01, 0.99)), r2


def derive_gamma(x: NDArray[np.float64] | np.ndarray) -> tuple[float, float, float]:
    """Return (gamma, H, r2) with invariant gamma = 2·H + 1."""
    arr: NDArray[np.float64] = np.asarray(x, dtype=np.float64)
    H, r2 = _hurst_dfa(arr)
    return round(2 * H + 1, 6), round(H, 6), round(r2, 6)


def risk_scalar(gamma: float) -> float:
    return round(max(0.0, 1.0 - abs(gamma - 1.0)), 6)


def classify(gamma: float, r2: float, stationary: bool) -> Regime:
    if not stationary or r2 < R2_MIN:
        return Regime.INVALID
    H = (gamma - 1.0) / 2.0
    if H < H_CRITICAL:
        return Regime.CRITICAL
    if H < H_DRIFT:
        return Regime.TRANSITION
    return Regime.DRIFT


@dataclass(frozen=True)
class State:
    gamma: float
    H: float
    r2: float
    regime: Regime
    rs: float
    stationary: bool

    @classmethod
    def from_window(cls, x: NDArray[np.float64] | np.ndarray) -> "State":
        arr: NDArray[np.float64] = np.asarray(x, dtype=np.float64)
        # INV-DRO3 convention fix (PR #345 RFC): ADF runs on log-returns, not
        # raw prices. Aligns with DFA input (engine.py:138) — was a
        # near-tautology on I(1) asset prices when tested on levels.
        log_returns = np.diff(np.log(np.abs(arr) + 1e-12))
        stat = _adf_stationary(log_returns)
        g, H, r2 = derive_gamma(arr)
        reg = classify(g, r2, stat)
        rs = risk_scalar(g) if reg != Regime.INVALID else 0.0
        return cls(gamma=g, H=H, r2=r2, regime=reg, rs=rs, stationary=stat)


def _windows(p: NDArray[np.float64], w: int, s: int) -> Iterator[NDArray[np.float64]]:
    for i in range(0, len(p) - w, s):
        yield p[i : i + w]


class _ARA(NamedTuple):
    pred: float
    regime: Regime
    errors: tuple[float, ...]
    stable: int


def _ara_step(a: _ARA, actual: State, alpha: float) -> _ARA:
    err = abs(actual.gamma - a.pred)
    stable = a.stable + 1 if err < EPSILON_H else 0
    pred = alpha * actual.gamma + (1 - alpha) * a.pred
    return _ARA(round(pred, 6), actual.regime, a.errors + (round(err, 6),), stable)


def _converged(a: _ARA) -> bool:
    return a.stable >= STABLE_RUNS


def _free_energy(a: _ARA) -> float:
    if not a.errors:
        return float("inf")
    return round(float(np.mean(a.errors[-STABLE_RUNS:])), 6)


def _trend(a: _ARA) -> str | None:
    if len(a.errors) < 3:
        return None
    w = a.errors[-4:]
    if w[-1] < w[0] * 0.7:
        return "CONVERGING"
    if w[-1] > w[0] * 1.3:
        return "DIVERGING"
    return "STABLE"


def _dro(price: NDArray[np.float64], window: int, step: int, alpha: float) -> tuple[State, _ARA]:
    gen = _windows(price, window, step)
    s = State.from_window(next(gen))
    ara = _ARA(s.gamma, s.regime, (), 0)
    for depth, w in enumerate(gen):
        if depth >= MAX_DEPTH:
            break
        s = State.from_window(w)
        ara = _ara_step(ara, s, alpha)
        if _converged(ara):
            break
    return s, ara


def geosync_observe(
    price: NDArray[np.float64] | np.ndarray,
    window: int = 512,
    step: int = 64,
) -> dict[str, Any]:
    """Public API: observe a price series and emit a regime + signal verdict.

    :param price: 1-D finite non-constant price array with length ≥ window+step.
    :param window: DFA window size; ≥ 512 recommended so DFA spans ≥ 3 octaves
        and R² typically > 0.90.
    :param step: stride between DRO iterations; EMA α = 2/(N+1) where
        N = (len(price) − window) // step.

    Invariants enforced on return:

    * ``gamma`` == 2·``H`` + 1 (rounded)
    * ``risk_scalar`` ∈ [0, 1]
    * ``regime`` ∈ {CRITICAL, TRANSITION, DRIFT, INVALID}
    * ``signal`` ∈ {LONG, SHORT, HOLD, REDUCE}
    """
    arr = _validate(price, min_len=window + step)
    N = (len(arr) - window) // step
    alpha = 2.0 / (N + 1)
    s, ara = _dro(arr, window, step, alpha)
    trend = _trend(ara)

    if s.regime == Regime.CRITICAL and s.rs > RS_LONG_THRESH and trend in ("CONVERGING", "STABLE"):
        signal = Signal.LONG
    elif s.regime == Regime.DRIFT and trend == "DIVERGING":
        signal = Signal.SHORT
    elif trend in ("CONVERGING", "STABLE"):
        signal = Signal.HOLD
    else:
        signal = Signal.REDUCE

    return {
        "gamma": s.gamma,
        "H": s.H,
        "r2_dfa": s.r2,
        "regime": s.regime.value,
        "risk_scalar": s.rs,
        "stationary": s.stationary,
        "signal": signal.value,
        "free_energy": _free_energy(ara),
        "ara_steps": len(ara.errors),
        "converged": _converged(ara),
        "trend": trend,
        "alpha_ema": round(alpha, 6),
    }
