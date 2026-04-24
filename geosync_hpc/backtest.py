# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Backtesting harness implementing SABRE CAL."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Deque, Iterator, List, Protocol

import numpy as np
import pandas as pd

from .conformal import ConformalCQR
from .evaluation import cvar, sharpe
from .execution import Execution
from .features import FeatureStore
from .logging import Logger
from .policy import Policy
from .quantile import QuantileModels
from .regime import RegimeModel
from .risk import Guardrails


class Resettable(Protocol):
    def reset(self) -> None: ...


@dataclass
class RuntimeStateManager:
    components: tuple[Resettable, ...]

    def reset_all(self) -> None:
        for component in self.components:
            component.reset()


@dataclass
class DequeState:
    history: Deque[float]

    def reset(self) -> None:
        self.history.clear()


@dataclass
class LoggerState:
    params: dict[str, str]
    logger: Logger | None = None

    def reset(self) -> None:
        if self.logger is not None:
            try:
                self.logger.end()
            except Exception:
                pass
        self.logger = Logger(params=self.params)


@dataclass(frozen=True)
class TradeStep:
    mid: float
    spread_frac: float
    costs: float
    target: float
    cur_pos: float
    fill_price: float
    pnl: float


@dataclass(frozen=True)
class RuntimeState:
    ret_hist: tuple[float, ...]
    exec_state: dict
    cqr_state: dict
    guard_peak: float | None
    guard_cooldown: int


class BacktestSession:
    _BASE_REQUIRED_COLUMNS = ("mid", "bid", "ask", "bid_size", "ask_size", "last", "last_size")

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.lookbacks = cfg["features"]["lookbacks"]
        self.fs = FeatureStore(cfg["features"]["fracdiff_d"], cfg["features"].get("ofi_window", 20))
        self.reg = RegimeModel(tuple(cfg["regime"]["bins"]))
        self.qm = QuantileModels(cfg["quantile"]["low_q"], cfg["quantile"]["high_q"])
        self.cqr = ConformalCQR(
            cfg["conformal"]["alpha"],
            cfg["conformal"]["decay"],
            cfg["conformal"]["window"],
            online_window=cfg["conformal"].get("online_window", 2000),
        )
        self.policy = Policy(
            cfg["policy"]["max_pos"],
            cfg["policy"]["kelly_shrink"],
            risk_gamma=cfg["policy"].get("risk_gamma", 10.0),
            cvar_alpha=cfg["policy"].get("cvar_alpha", 0.95),
            cvar_window=cfg["policy"].get("cvar_window", 1000),
        )
        self.exec = Execution(
            cfg["execution"]["fee_bps"],
            cfg["execution"]["impact_coeff"],
            cfg["execution"].get("impact_model", "square_root"),
            cfg["execution"].get("queue_fill_p", 0.85),
            seed=cfg.get("seed", 7),
        )
        self.guard = Guardrails(
            cfg["risk"]["intraday_dd_limit"],
            cfg["risk"]["loss_streak_cooldown"],
            cfg["risk"]["vola_spike_mult"],
            cfg["risk"].get("exposure_cap", 1.0),
        )
        self.logger = Logger(
            params={"impact_model": cfg["execution"].get("impact_model", "square_root")}
        )
        self.buffer_frac = (cfg["conformal"].get("buffer_bps", 0.0) or 0.0) * 1e-4
        self.horizon = int(cfg.get("target", {}).get("horizon", 0))
        self.online_update = bool(cfg["conformal"].get("online_update", False))
        self._ret_hist: Deque[float] = deque(maxlen=int(2 * self.policy.cvar_window))
        self._logger_params = {"impact_model": cfg["execution"].get("impact_model", "square_root")}
        self._logger_state = LoggerState(params=self._logger_params, logger=self.logger)
        self._ret_hist_state = DequeState(history=self._ret_hist)
        self._max_position_jump_mult = float(cfg["risk"].get("max_position_jump_mult", 2.0))
        self._state_manager = RuntimeStateManager(
            components=(
                self.fs,
                self.reg,
                self.guard,
                self.exec,
                self.cqr,
                self._ret_hist_state,
                self._logger_state,
            )
        )

    def _reset_runtime_state(self) -> None:
        """Reset mutable streaming state before each independent run."""
        self._state_manager.reset_all()
        self.logger = self._logger_state.logger or Logger(params=self._logger_params)

    @staticmethod
    def _validate_finite_frame(df: pd.DataFrame, cols: list[str], label: str) -> None:
        subset = df[cols]
        finite_mask = np.isfinite(subset.to_numpy(dtype=float))
        if not finite_mask.all():
            raise ValueError(f"Non-finite values detected in {label}: {cols}")

    @staticmethod
    def validate_finite(label: str, **values: float) -> None:
        bad = [k for k, v in values.items() if not np.isfinite(v)]
        if bad:
            raise ValueError(f"Non-finite values detected in {label}: {bad}")

    @contextmanager
    def logger_context(self) -> Iterator[Logger]:
        self._logger_state.reset()
        self.logger = self._logger_state.logger or Logger(params=self._logger_params)
        try:
            yield self.logger
        finally:
            self.logger.end()

    def get_state(self) -> RuntimeState:
        return RuntimeState(
            ret_hist=tuple(self._ret_hist),
            exec_state=self.exec.get_state(),
            cqr_state=self.cqr.get_state(),
            guard_peak=self.guard.peak,
            guard_cooldown=self.guard.cooldown,
        )

    def set_state(self, state: RuntimeState) -> None:
        self._ret_hist.clear()
        self._ret_hist.extend(state.ret_hist)
        self.exec.set_state(state.exec_state)
        self.cqr.set_state(state.cqr_state)
        self.guard.peak = state.guard_peak
        self.guard.cooldown = state.guard_cooldown

    def _validate_inputs(
        self, df: pd.DataFrame, feat_cols: list[str], y_col: str, spread_col: str, vol_col: str
    ) -> None:
        required = set(self._BASE_REQUIRED_COLUMNS)
        required.update({y_col, spread_col, vol_col})
        required.update(feat_cols)
        missing = sorted(col for col in required if col not in df.columns)
        if missing:
            raise ValueError(f"Missing required columns for backtest: {missing}")
        if len(df) < 2:
            raise ValueError("Backtest requires at least 2 rows of data.")

    def _assert_step_invariants(self, step: TradeStep) -> None:
        vals = {
            "mid": step.mid,
            "spread_frac": step.spread_frac,
            "costs": step.costs,
            "target": step.target,
            "cur_pos": step.cur_pos,
            "fill_price": step.fill_price,
            "pnl": step.pnl,
        }
        bad = [k for k, v in vals.items() if not np.isfinite(v)]
        if bad:
            raise ValueError(f"Non-finite runtime values detected: {bad}")
        cap = float(self.guard.exposure_cap)
        if abs(step.target) > cap + 1e-12:
            raise ValueError(f"Target position {step.target} exceeds exposure cap {cap}.")
        if step.costs < 0.0:
            raise ValueError(f"Negative costs detected: {step.costs}")
        max_jump = self._max_position_jump_mult * cap
        if abs(step.target - step.cur_pos) > max_jump + 1e-12:
            raise ValueError(
                f"Non-physical position jump detected: {step.cur_pos} -> {step.target}"
            )
        slip_bound = abs(step.spread_frac) * abs(step.mid)
        if abs(step.fill_price - step.mid) > slip_bound + 1e-9:
            raise ValueError(
                f"Fill price deviation {step.fill_price - step.mid} exceeds expected spread-bound {slip_bound}."
            )

    def fit_quantiles(self, X_fit: pd.DataFrame, y_fit: pd.Series) -> None:
        self._validate_finite_frame(X_fit, list(X_fit.columns), "fit_quantiles.X_fit")
        if not np.isfinite(y_fit.to_numpy(dtype=float)).all():
            raise ValueError("Non-finite values detected in fit_quantiles.y_fit")
        self.qm.fit(X_fit, y_fit)

    def calibrate_conformal(self, X_cal: pd.DataFrame, y_cal: pd.Series) -> None:
        self._validate_finite_frame(X_cal, list(X_cal.columns), "calibrate_conformal.X_cal")
        if not np.isfinite(y_cal.to_numpy(dtype=float)).all():
            raise ValueError("Non-finite values detected in calibrate_conformal.y_cal")
        Lc, Uc = [], []
        for i in range(len(X_cal)):
            x_dict = dict(zip(self.qm.cols or [], X_cal.iloc[i].values))
            L, M, U = self.qm.predict_all(x_dict)
            Lc.append(L)
            Uc.append(U)
        l_arr = np.asarray(Lc, dtype=float)
        u_arr = np.asarray(Uc, dtype=float)
        y_arr = y_cal.to_numpy(dtype=float)
        mask = np.isfinite(l_arr) & np.isfinite(u_arr) & np.isfinite(y_arr)
        self.cqr.fit_calibrate(l_arr[mask], u_arr[mask], y_arr[mask])

    def run(
        self,
        df: pd.DataFrame,
        feat_cols: list[str],
        y_col: str,
        spread_col: str = "spread",
        vol_col: str = "vol10",
        save_csv: str | None = None,
    ) -> pd.DataFrame:
        self._validate_inputs(df, feat_cols, y_col, spread_col, vol_col)
        self._reset_runtime_state()
        pos = 0.0
        eq = 0.0
        equity = [0.0]
        self.guard.start_session(equity[0])
        loss_streak = 0
        vola_hist: list[float] = []
        rows: list[dict[str, float]] = []
        covered = 0.0
        cov_count = 0
        rv_ref = max(1e-9, df[vol_col].iloc[: max(10, len(df) // 5)].mean())
        L_pred_hist: List[float] = []
        U_pred_hist: List[float] = []

        with self.logger_context():
            for i in range(len(df) - 1):
                row = df.iloc[i]
                snap_row = {
                    "mid": row["mid"],
                    "bid": row["bid"],
                    "ask": row["ask"],
                    "bid_size": row["bid_size"],
                    "ask_size": row["ask_size"],
                    "last": row["last"],
                    "last_size": row["last_size"],
                }
                self.fs.update(snap_row)
                feats = self.fs.snapshot(self.lookbacks)
                if feats is None:
                    continue
                reg = self.reg.update(feats)

                xrow = dict(zip(feat_cols, df[feat_cols].iloc[i].values))
                L, M, U = self.qm.predict_all(xrow)
                self.validate_finite("quantile_predictions", L=L, M=M, U=U)
                L_pred_hist.append(L)
                U_pred_hist.append(U)

                rv_t = float(df[vol_col].iloc[i])
                self.validate_finite("volatility_input", rv_t=rv_t)
                self.cqr.dynamic_alpha(rv_t, rv_ref)
                Lc, Uc = self.cqr.interval(L, U)
                yt = float(row[y_col])
                self.validate_finite("coverage_inputs", yt=yt, Lc=Lc, Uc=Uc)
                if Lc <= yt <= Uc:
                    covered += 1.0
                cov_count += 1
                cov = covered / cov_count
                self.logger.log_metric("coverage", cov, step=i)
                self.logger.log_metric("alpha_eff", self.cqr.alpha, step=i)

                notional_frac = min(1.0, abs(1.0 - pos))
                costs = self.exec.costs(df[spread_col].iloc[i], rv_t, notional_frac=notional_frac)
                self.logger.log_metric("qhat", self.cqr.qhat or 0.0, step=i)
                self.logger.log_metric("costs", costs, step=i)

                proposed = self.policy.decide(Lc, M, Uc, costs, self.buffer_frac, self._ret_hist)
                checks = self.guard.check(
                    equity,
                    feats.get("rv", 0.0),
                    float(np.mean(vola_hist[-200:])) if vola_hist else 0.0,
                    loss_streak,
                    proposed,
                )
                target = checks["throttle"] * checks["pos_cap"]
                fill_price = self.exec.fill(feats["mid"], df[spread_col].iloc[i], target, pos)

                pnl = (target - pos) * (df["mid"].iloc[i + 1] - fill_price) - abs(target - pos) * (
                    costs * feats["mid"]
                )
                self._assert_step_invariants(
                    TradeStep(
                        mid=feats["mid"],
                        spread_frac=df[spread_col].iloc[i],
                        costs=costs,
                        target=target,
                        cur_pos=pos,
                        fill_price=fill_price,
                        pnl=pnl,
                    )
                )
                pos = target
                eq += pnl
                equity.append(eq)
                loss_streak = (loss_streak + 1) if pnl < 0 else 0
                vola_hist.append(feats.get("rv", 0.0))
                ret_norm = pnl / max(1e-9, feats["mid"])
                self._ret_hist.append(ret_norm)

                rows.append(
                    {
                        "ts": df.index[i],
                        "mid": feats["mid"],
                        "pos": pos,
                        "pnl": pnl,
                        "eq": eq,
                        "L": Lc,
                        "M": M,
                        "U": Uc,
                        "costs": costs,
                        "regime": reg["regime"],
                        "spread": feats["spread"],
                        "eff_spread": feats.get("eff_spread", np.nan),
                        "ofi": feats.get("ofi_sh", np.nan),
                        "lambda": feats.get("kyle_lambda", np.nan),
                    }
                )

                if i % 500 == 0 and i > 0:
                    self.logger.log_metric("equity", eq, step=i)
                    self.logger.log_metric(
                        "sharpe_partial", sharpe(np.diff(np.array(equity))), step=i
                    )

                if self.online_update and self.horizon > 0 and i >= self.horizon:
                    idx = i - self.horizon
                    if idx < len(L_pred_hist) and idx < len(U_pred_hist):
                        y_true = float(df[y_col].iloc[idx])
                        L_hist = float(L_pred_hist[idx])
                        U_hist = float(U_pred_hist[idx])
                        self.validate_finite(
                            "online_update_inputs", y_true=y_true, L_hist=L_hist, U_hist=U_hist
                        )
                        self.cqr.update_online(L_hist, U_hist, y_true)

        res = pd.DataFrame(rows)
        if save_csv and not res.empty:
            try:
                res.to_csv(save_csv, index=False)
                self.logger.log_artifact(save_csv)
            except Exception:
                pass
        r = res["pnl"].values if not res.empty else np.array([])
        self.logger.log_metric("sharpe", sharpe(r))
        self.logger.log_metric("cvar95", cvar(r, 0.95))
        return res


class BacktesterCAL(BacktestSession):
    """Backward-compatible alias for legacy API usage."""
