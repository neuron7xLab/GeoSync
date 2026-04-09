# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Backtesting utilities, strategies, and performance analytics.

Keep package import fail-closed for optional integrations: importing
``backtest`` should not require all transitive runtime dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _MissingOptionalDependency:
    symbol: str
    reason: str

    def _raise(self) -> None:
        raise ImportError(
            f"{self.symbol} is unavailable because an optional dependency failed "
            f"to import: {self.reason}"
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._raise()

    def __getattr__(self, _name: str) -> Any:
        self._raise()

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
    DopamineTDParams = _MissingOptionalDependency("DopamineTDParams", str(exc))  # type: ignore[assignment]
    dopamine_td_signal = _MissingOptionalDependency("dopamine_td_signal", str(exc))  # type: ignore[assignment]
    run_dopamine_backtest = _MissingOptionalDependency("run_dopamine_backtest", str(exc))  # type: ignore[assignment]
    run_vectorized_dopamine_td = _MissingOptionalDependency("run_vectorized_dopamine_td", str(exc))  # type: ignore[assignment]

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
    ControlledExperiment = _MissingOptionalDependency("ControlledExperiment", str(exc))  # type: ignore[assignment]
    LiquidityShock = _MissingOptionalDependency("LiquidityShock", str(exc))  # type: ignore[assignment]
    OrderBookDepthConfig = _MissingOptionalDependency("OrderBookDepthConfig", str(exc))  # type: ignore[assignment]
    OrderBookDepthProfile = _MissingOptionalDependency("OrderBookDepthProfile", str(exc))  # type: ignore[assignment]
    StrategyEvaluation = _MissingOptionalDependency("StrategyEvaluation", str(exc))  # type: ignore[assignment]
    StructuralBreak = _MissingOptionalDependency("StructuralBreak", str(exc))  # type: ignore[assignment]
    SyntheticScenario = _MissingOptionalDependency("SyntheticScenario", str(exc))  # type: ignore[assignment]
    SyntheticScenarioConfig = _MissingOptionalDependency("SyntheticScenarioConfig", str(exc))  # type: ignore[assignment]
    SyntheticScenarioGenerator = _MissingOptionalDependency("SyntheticScenarioGenerator", str(exc))  # type: ignore[assignment]
    VolatilityShift = _MissingOptionalDependency("VolatilityShift", str(exc))  # type: ignore[assignment]

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
