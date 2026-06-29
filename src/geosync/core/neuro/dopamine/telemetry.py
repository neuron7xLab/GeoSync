# SPDX-License-Identifier: MIT
from __future__ import annotations

import math

METRIC_NAMES = (
    "dopamine.rpe_finite_rate",
    "dopamine.output_finite_rate",
    "dopamine.output_bound_violation_count",
    "dopamine.config_validation_pass_rate",
    "dopamine.schema_runtime_parity_failures",
    "dopamine.contract_violation_count",
    "dopamine.null_survival_rate",
    "dopamine.artifact_completeness_rate",
    "dopamine.claim_promotion_block_count",
    "dopamine.backtest_parity_error_rate",
    "dopamine.p95_step_latency_ms",
    "dopamine.p99_step_latency_ms",
    "dopamine.max_memory_mb",
    "dopamine.cognitive_value_score",
    "dopamine.cognitive_value_delta",
    "dopamine.cognitive_value_priority",
)


def metric_catalog() -> dict[str, dict[str, str]]:
    return {name: {"owner": "geosync.dopamine", "unit": "ratio|count|ms|MiB"} for name in METRIC_NAMES}


def bounded_value(human: float, system: float, repository: float, research: float) -> float:
    def unit(value: float) -> float:
        if not math.isfinite(value):
            return 0.0
        return min(1.0, max(0.0, value))

    return (
        0.35 * unit(human)
        + 0.30 * unit(system)
        + 0.20 * unit(repository)
        + 0.15 * unit(research)
    )


def bounded_value_delta(previous_score: float, current_score: float) -> dict[str, float]:
    previous = min(1.0, max(0.0, previous_score))
    current = min(1.0, max(0.0, current_score))
    delta = current - previous
    rpe = math.tanh(3.0 * delta)
    return {"delta": delta, "bounded_rpe": rpe, "priority": min(1.0, max(0.0, 0.5 + 0.5 * rpe))}
