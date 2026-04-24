# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Runtime state dataclasses for deterministic backtesting sessions."""

from __future__ import annotations

from dataclasses import dataclass


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
    l_pred_hist: tuple[float, ...]
    u_pred_hist: tuple[float, ...]
    equity: tuple[float, ...]
    vola_hist: tuple[float, ...]
    loss_streak: int
    pos: float
    eq: float
    exec_state: dict
    cqr_state: dict
    guard_peak: float
    guard_cooldown: int
    guard_session_started: bool
