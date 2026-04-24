# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Quantile regression ensemble for conditional intervals."""

from __future__ import annotations

import importlib.util

import numpy as np

_HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
if _HAS_SKLEARN:
    from sklearn.ensemble import GradientBoostingRegressor


class _ConstantQuantileRegressor:
    """Deterministic fallback when sklearn is unavailable."""

    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self._q = 0.0

    def fit(self, X, y) -> "_ConstantQuantileRegressor":
        del X
        y_arr = np.asarray(y, dtype=float)
        if y_arr.size == 0:
            self._q = 0.0
        else:
            self._q = float(np.quantile(y_arr, self.alpha))
        return self

    def predict(self, X) -> np.ndarray:
        n = len(X)
        return np.full(shape=(n,), fill_value=self._q, dtype=float)


class QuantileModels:
    def __init__(
        self,
        low_q: float = 0.2,
        high_q: float = 0.8,
        seed: int = 7,
        allow_fallback: bool = False,
    ) -> None:
        if not _HAS_SKLEARN and not allow_fallback:
            raise RuntimeError(
                "scikit-learn is required for QuantileModels in production mode. "
                "Set allow_fallback=True only for constrained test/smoke environments."
            )
        if _HAS_SKLEARN:
            self.low = GradientBoostingRegressor(loss="quantile", alpha=low_q, random_state=seed)
            self.med = GradientBoostingRegressor(loss="quantile", alpha=0.5, random_state=seed)
            self.high = GradientBoostingRegressor(loss="quantile", alpha=high_q, random_state=seed)
        else:
            self.low = _ConstantQuantileRegressor(low_q)
            self.med = _ConstantQuantileRegressor(0.5)
            self.high = _ConstantQuantileRegressor(high_q)
        self.cols: list[str] | None = None
        self.fitted = False

    def fit(self, X, y):
        self.cols = list(X.columns)
        self.low.fit(X, y)
        self.med.fit(X, y)
        self.high.fit(X, y)
        self.fitted = True
        return self

    def predict_all(self, x_row: dict[str, float]) -> tuple[float, float, float]:
        if not self.cols:
            raise ValueError("QuantileModels must be fitted before prediction.")
        x = np.array([x_row.get(c, 0.0) for c in self.cols]).reshape(1, -1)
        low = float(self.low.predict(x)[0])
        med = float(self.med.predict(x)[0])
        high = float(self.high.predict(x)[0])
        return low, med, high
