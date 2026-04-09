# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Interface definitions for GeoSync subsystems."""
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

from interfaces.backtest import BacktestEngine

try:  # optional interface groups can depend on runtime connectors
    from interfaces.execution import PositionSizer, RiskController
except Exception as exc:  # pragma: no cover - optional dependency chain
    PositionSizer = _MissingOptionalDependency("PositionSizer", str(exc))  # type: ignore[assignment]
    RiskController = _MissingOptionalDependency("RiskController", str(exc))  # type: ignore[assignment]

try:
    from interfaces.ingestion import AsyncDataIngestionService, DataIngestionService
except Exception as exc:  # pragma: no cover - optional dependency chain
    AsyncDataIngestionService = _MissingOptionalDependency("AsyncDataIngestionService", str(exc))  # type: ignore[assignment]
    DataIngestionService = _MissingOptionalDependency("DataIngestionService", str(exc))  # type: ignore[assignment]

__all__ = [
    "AsyncDataIngestionService",
    "BacktestEngine",
    "DataIngestionService",
    "PositionSizer",
    "RiskController",
]
