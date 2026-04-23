# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Runtime safety tests for stateful HPC primitives."""

from __future__ import annotations

from geosync_hpc.execution import Execution
from geosync_hpc.risk import Guardrails


def test_guardrails_do_not_trigger_drawdown_halt_when_peak_nonpositive() -> None:
    guard = Guardrails(intraday_dd_limit=0.01, loss_streak_cooldown=10)
    # Negative equity without a positive peak should not create synthetic DD halts.
    out = guard.check(
        equity_curve=[-100.0], vola=0.1, vola_avg=0.1, loss_streak=0, proposed_pos=0.5
    )
    assert out["halt"] is False
    assert out["throttle"] == 1.0


def test_guardrails_cooldown_counts_down_without_being_rearmed_every_step() -> None:
    guard = Guardrails(intraday_dd_limit=0.01, loss_streak_cooldown=1)
    # Trigger cooldown once.
    guard.check(equity_curve=[1.0], vola=0.1, vola_avg=0.1, loss_streak=1, proposed_pos=0.5)
    # During cooldown, no fresh halt trigger should occur.
    for _ in range(59):
        out = guard.check(
            equity_curve=[1.0],
            vola=0.1,
            vola_avg=0.1,
            loss_streak=0,
            proposed_pos=0.5,
        )
        assert out["throttle"] == 0.0
    out = guard.check(equity_curve=[1.0], vola=0.1, vola_avg=0.1, loss_streak=0, proposed_pos=0.5)
    assert out["throttle"] == 1.0


def test_execution_clamps_queue_fill_probability_to_valid_range() -> None:
    e_high = Execution(queue_fill_p=10.0, seed=1)
    e_low = Execution(queue_fill_p=-2.0, seed=1)
    assert e_high.queue_fill_p == 1.0
    assert e_low.queue_fill_p == 0.0
