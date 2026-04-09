# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Utilities for explicit optional-dependency placeholders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MissingOptionalDependency:
    """Placeholder that raises a descriptive ImportError when accessed."""

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
