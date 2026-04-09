# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Backtesting utilities, strategies, and performance analytics.

Keep package import fail-closed for optional integrations: importing
``backtest`` should not require all transitive runtime dependencies.
"""
from __future__ import annotations

from core.utils.optional_dependency import MissingOptionalDependency

from .engine import LatencyConfig, OrderBookConfig
from .performance import PerformanceReport, compute_performance_metrics, export_performance_report

try:  # optional research helpers
    from .dopamine_td import (
        DopamineTDParams,
        dopamine_td_signal,
        run_dopamine_backtest,
        run_vectorized_dopamine_td,
    )
except Exception as exc:  # pragma: no cover - optional dependency chain
    DopamineTDParams = MissingOptionalDependency("DopamineTDParams", str(exc))  # type: ignore[assignment]
    dopamine_td_signal = MissingOptionalDependency("dopamine_td_signal", str(exc))  # type: ignore[assignment]
    run_dopamine_backtest = MissingOptionalDependency("run_dopamine_backtest", str(exc))  # type: ignore[assignment]
    run_vectorized_dopamine_td = MissingOptionalDependency("run_vectorized_dopamine_td", str(exc))  # type: ignore[assignment]

try:  # optional synthetic scenario suite
    from .synthetic import (
        ControlledExperiment,
        LiquidityShock,
        OrderBookDepthConfig,
        OrderBookDepthProfile,
        StrategyEvaluation,
        StructuralBreak,
        SyntheticScenario,
        SyntheticScenarioConfig,
        SyntheticScenarioGenerator,
        VolatilityShift,
    )
except Exception as exc:  # pragma: no cover - optional dependency chain
    ControlledExperiment = MissingOptionalDependency("ControlledExperiment", str(exc))  # type: ignore[assignment]
    LiquidityShock = MissingOptionalDependency("LiquidityShock", str(exc))  # type: ignore[assignment]
    OrderBookDepthConfig = MissingOptionalDependency("OrderBookDepthConfig", str(exc))  # type: ignore[assignment]
    OrderBookDepthProfile = MissingOptionalDependency("OrderBookDepthProfile", str(exc))  # type: ignore[assignment]
    StrategyEvaluation = MissingOptionalDependency("StrategyEvaluation", str(exc))  # type: ignore[assignment]
    StructuralBreak = MissingOptionalDependency("StructuralBreak", str(exc))  # type: ignore[assignment]
    SyntheticScenario = MissingOptionalDependency("SyntheticScenario", str(exc))  # type: ignore[assignment]
    SyntheticScenarioConfig = MissingOptionalDependency("SyntheticScenarioConfig", str(exc))  # type: ignore[assignment]
    SyntheticScenarioGenerator = MissingOptionalDependency("SyntheticScenarioGenerator", str(exc))  # type: ignore[assignment]
    VolatilityShift = MissingOptionalDependency("VolatilityShift", str(exc))  # type: ignore[assignment]

__all__ = [
    "LatencyConfig",
    "OrderBookConfig",
    "ControlledExperiment",
    "LiquidityShock",
    "OrderBookDepthConfig",
    "OrderBookDepthProfile",
    "StrategyEvaluation",
    "StructuralBreak",
    "SyntheticScenario",
    "SyntheticScenarioConfig",
    "SyntheticScenarioGenerator",
    "VolatilityShift",
    "PerformanceReport",
    "compute_performance_metrics",
    "export_performance_report",
    "DopamineTDParams",
    "dopamine_td_signal",
    "run_dopamine_backtest",
    "run_vectorized_dopamine_td",
]
