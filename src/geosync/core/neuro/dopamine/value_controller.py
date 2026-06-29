# SPDX-License-Identifier: MIT
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ValueVector:
    human: float
    system: float
    repository: float
    research: float


@dataclass(frozen=True)
class CognitiveValueStep:
    score: float
    delta: float
    bounded_rpe: float
    priority: float
    gate: str


def _unit(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def cognitive_value(vector: ValueVector) -> float:
    return (
        0.35 * _unit(vector.human)
        + 0.30 * _unit(vector.system)
        + 0.20 * _unit(vector.repository)
        + 0.15 * _unit(vector.research)
    )


def cognitive_value_step(previous: ValueVector, current: ValueVector) -> CognitiveValueStep:
    previous_score = cognitive_value(previous)
    score = cognitive_value(current)
    delta = score - previous_score
    bounded_rpe = math.tanh(3.0 * delta)
    priority = min(1.0, max(0.0, 0.5 + 0.5 * bounded_rpe))
    if bounded_rpe > 0.05:
        gate = "EXPAND"
    elif bounded_rpe < -0.05:
        gate = "REDUCE"
    else:
        gate = "HOLD"
    return CognitiveValueStep(
        score=score,
        delta=delta,
        bounded_rpe=bounded_rpe,
        priority=priority,
        gate=gate,
    )
