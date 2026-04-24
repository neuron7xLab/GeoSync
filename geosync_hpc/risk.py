# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Risk guardrails used during backtests."""

from __future__ import annotations

import numpy as np


class Guardrails:
    def __init__(
        self,
        intraday_dd_limit: float = 0.02,
        loss_streak_cooldown: int = 4,
        vola_spike_mult: float = 2.5,
        exposure_cap: float = 1.0,
    ) -> None:
        self.dd_limit = intraday_dd_limit
        self.cooldown_streak = loss_streak_cooldown
        self.vola_mult = vola_spike_mult
        self.exposure_cap = exposure_cap
        self.peak = 0.0
        self._session_started = False
        self.cooldown = 0

    def check(
        self,
        equity_curve: list[float],
        vola: float,
        vola_avg: float,
        loss_streak: int,
        proposed_pos: float,
    ) -> dict[str, float | bool]:
        if len(equity_curve) == 0:
            return {
                "halt": False,
                "throttle": 1.0,
                "pos_cap": np.clip(proposed_pos, -self.exposure_cap, self.exposure_cap),
            }
        assert self._session_started, "Guardrails.start_session() must be called before check()."
        eq = float(equity_curve[-1])
        self.peak = max(self.peak, eq)
        denom = max(abs(self.peak), 1e-9)
        dd = max(0.0, (self.peak - eq) / denom)
        halt = dd > self.dd_limit or loss_streak >= self.cooldown_streak
        throttle = 0.5 if vola > self.vola_mult * max(1e-9, vola_avg) else 1.0
        if halt and self.cooldown == 0:
            self.cooldown = 60
        if self.cooldown > 0:
            self.cooldown -= 1
            throttle = 0.0
        pos_cap = float(np.clip(proposed_pos, -self.exposure_cap, self.exposure_cap))
        return {"halt": halt, "throttle": throttle, "pos_cap": pos_cap}

    def start_session(self, starting_equity: float) -> None:
        """Initialize run-local drawdown baseline."""
        self.peak = float(starting_equity)
        self._session_started = True
        self.cooldown = 0

    def reset(self) -> None:
        """Reset drawdown/cooldown memory for an independent backtest."""
        self.peak = 0.0
        self._session_started = False
        self.cooldown = 0
