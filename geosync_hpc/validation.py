# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Validation service for runtime contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .state import TradeStep


class ValidationService:
    @staticmethod
    def finite_frame(df: pd.DataFrame, cols: list[str], label: str) -> None:
        subset = df[cols]
        finite_mask = np.isfinite(subset.to_numpy(dtype=float))
        if not finite_mask.all():
            raise ValueError(f"Non-finite values detected in {label}: {cols}")

    @staticmethod
    def finite_values(label: str, **values: float) -> None:
        bad = [k for k, v in values.items() if not np.isfinite(v)]
        if bad:
            raise ValueError(f"Non-finite values detected in {label}: {bad}")

    @staticmethod
    def trade_step(step: TradeStep, exposure_cap: float, max_position_jump_mult: float) -> None:
        vals = {
            "mid": step.mid,
            "spread_frac": step.spread_frac,
            "costs": step.costs,
            "target": step.target,
            "cur_pos": step.cur_pos,
            "fill_price": step.fill_price,
            "pnl": step.pnl,
        }
        ValidationService.finite_values("trade_step", **vals)
        if abs(step.target) > exposure_cap + 1e-12:
            raise ValueError(f"Target position {step.target} exceeds exposure cap {exposure_cap}.")
        if step.costs < 0.0:
            raise ValueError(f"Negative costs detected: {step.costs}")
        max_jump = max_position_jump_mult * exposure_cap
        if abs(step.target - step.cur_pos) > max_jump + 1e-12:
            raise ValueError(
                f"Non-physical position jump detected: {step.cur_pos} -> {step.target}"
            )
        slip_bound = abs(step.spread_frac) * abs(step.mid)
        if abs(step.fill_price - step.mid) > slip_bound + 1e-9:
            raise ValueError(
                f"Fill price deviation {step.fill_price - step.mid} exceeds expected spread-bound {slip_bound}."
            )
