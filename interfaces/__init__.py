# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Interface definitions for GeoSync subsystems."""
from __future__ import annotations

from core.utils.optional_dependency import MissingOptionalDependency
from interfaces.backtest import BacktestEngine

try:  # optional interface groups can depend on runtime connectors
    from interfaces.execution import PositionSizer, RiskController
except Exception as exc:  # pragma: no cover - optional dependency chain
    PositionSizer = MissingOptionalDependency("PositionSizer", str(exc))  # type: ignore[assignment]
    RiskController = MissingOptionalDependency("RiskController", str(exc))  # type: ignore[assignment]

try:
    from interfaces.ingestion import AsyncDataIngestionService, DataIngestionService
except Exception as exc:  # pragma: no cover - optional dependency chain
    AsyncDataIngestionService = MissingOptionalDependency("AsyncDataIngestionService", str(exc))  # type: ignore[assignment]
    DataIngestionService = MissingOptionalDependency("DataIngestionService", str(exc))  # type: ignore[assignment]

__all__ = [
    "AsyncDataIngestionService",
    "BacktestEngine",
    "DataIngestionService",
    "PositionSizer",
    "RiskController",
]
