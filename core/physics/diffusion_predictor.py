# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""T7 — Graph diffusion volatility front predictor.

Propagation equation:
    ρ(t) = exp(-D · L(κ) · t) · ρ₀

where:
    L(κ)  = graph Laplacian with Ricci-modulated diffusion
    D_ij  = D₀ · exp(κ_ij)   (Ricci curvature scaling)
    ρ₀    = initial volatility density (recent realised vol per asset)

Volatility front = assets where ρ_i(t+1) > θ.

Backtest metric: ROC-AUC for front prediction vs realised vol spikes
at horizon t+1 to t+3.

Validated framework: Kikuchi (2025) graph diffusion for information
propagation. Financial backtest of volatility prediction is novel.

The key insight: information (and volatility) spreads faster on
positively curved edges (well-connected neighborhoods) and slower
on negatively curved edges (bottleneck regions). This is empirically
validated in network science (Sandhu 2016).

References:
    Kikuchi "Graph diffusion for network analysis" (2025)
    Chung "Spectral Graph Theory" (1997)
    Sandhu et al. "Graph curvature for cancer networks" Sci. Rep. (2016)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm


@dataclass(frozen=True, slots=True)
class VolatilityFrontPrediction:
    """Prediction of volatility front propagation."""

    predicted_front: list[int]        # asset indices predicted to spike
    density_t1: NDArray[np.float64]   # propagated density at t+1
    density_t3: NDArray[np.float64]   # propagated density at t+3
    initial_density: NDArray[np.float64]
    threshold: float


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Backtest evaluation of front prediction."""

    n_windows: int
    roc_auc_t1: float
    roc_auc_t3: float
    precision_t1: float
    recall_t1: float
    mean_lead_time: float  # average steps prediction leads actual spike


class DiffusionVolatilityPredictor:
    """Predict volatility propagation via graph diffusion.

    Parameters
    ----------
    D_0 : float
        Base diffusion coefficient (default 1.0).
    threshold_quantile : float
        Quantile for defining "high volatility" (default 0.75).
    horizons : tuple[float, ...]
        Prediction horizons in time units (default (1.0, 3.0)).
    """

    def __init__(
        self,
        D_0: float = 1.0,
        threshold_quantile: float = 0.75,
        horizons: tuple[float, ...] = (1.0, 3.0),
    ) -> None:
        if D_0 <= 0:
            raise ValueError(f"D_0 must be > 0, got {D_0}")
        if not 0 < threshold_quantile < 1:
            raise ValueError(f"threshold_quantile in (0,1), got {threshold_quantile}")
        self._D_0 = D_0
        self._thresh_q = threshold_quantile
        self._horizons = horizons

    def _build_laplacian(
        self,
        adjacency: NDArray[np.float64],
        curvature: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Build curvature-weighted Laplacian."""
        A = np.asarray(adjacency, dtype=np.float64)
        if curvature is not None:
            kappa = np.asarray(curvature, dtype=np.float64)
            W = A * self._D_0 * np.exp(kappa)
        else:
            W = A * self._D_0
        W = 0.5 * (W + W.T)
        np.fill_diagonal(W, 0.0)
        D = np.diag(W.sum(axis=1))
        return D - W

    def _propagate(
        self,
        rho_0: NDArray[np.float64],
        L: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """ρ(t) = exp(-L·t) · ρ₀ with renormalisation."""
        if t <= 0:
            return rho_0.copy()
        propagator = expm(-L * t)
        rho = propagator @ rho_0
        rho = np.maximum(rho, 0.0)
        total = rho.sum()
        if total > 0:
            rho /= total
        return rho

    def predict(
        self,
        realized_vol: NDArray[np.float64],
        adjacency: NDArray[np.float64],
        curvature: NDArray[np.float64] | None = None,
    ) -> VolatilityFrontPrediction:
        """Predict volatility front from current state.

        Parameters
        ----------
        realized_vol : (N,) current realised volatility per asset.
        adjacency : (N, N) correlation-based adjacency.
        curvature : (N, N) Ricci curvature matrix (optional).

        Returns
        -------
        VolatilityFrontPrediction.
        """
        vol = np.asarray(realized_vol, dtype=np.float64)
        n = vol.size

        # Normalise vol to probability density
        rho_0 = np.maximum(vol, 0.0)
        total = rho_0.sum()
        if total > 0:
            rho_0 = rho_0 / total
        else:
            rho_0 = np.ones(n) / n

        L = self._build_laplacian(adjacency, curvature)

        rho_t1 = self._propagate(rho_0, L, self._horizons[0])
        rho_t3 = self._propagate(
            rho_0, L, self._horizons[-1] if len(self._horizons) > 1 else self._horizons[0]
        )

        # Threshold for front detection
        threshold = float(np.quantile(rho_t1, self._thresh_q))
        front = np.where(rho_t1 > threshold)[0].tolist()

        return VolatilityFrontPrediction(
            predicted_front=front,
            density_t1=rho_t1,
            density_t3=rho_t3,
            initial_density=rho_0,
            threshold=threshold,
        )

    def backtest(
        self,
        prices: NDArray[np.float64],
        adjacency_series: list[NDArray[np.float64]] | None = None,
        curvature_series: list[NDArray[np.float64]] | None = None,
        vol_window: int = 10,
        correlation_window: int = 30,
        spike_threshold_quantile: float = 0.80,
    ) -> BacktestResult:
        """Backtest volatility front prediction on historical data.

        Parameters
        ----------
        prices : (T, N) price history.
        adjacency_series : optional pre-computed adjacency per window.
        curvature_series : optional pre-computed curvature per window.
        vol_window : int, window for realised vol computation.
        correlation_window : int, window for adjacency construction.
        spike_threshold_quantile : float, quantile for "actual spike".

        Returns
        -------
        BacktestResult with ROC-AUC metrics.
        """
        prices = np.asarray(prices, dtype=np.float64)
        T, N = prices.shape

        returns = np.abs(
            np.diff(prices, axis=0) / np.maximum(np.abs(prices[:-1]), 1e-12)
        )

        start = max(vol_window, correlation_window)
        n_windows = T - start - 3  # need 3 steps ahead

        if n_windows < 5:
            return BacktestResult(
                n_windows=0, roc_auc_t1=0.5, roc_auc_t3=0.5,
                precision_t1=0.0, recall_t1=0.0, mean_lead_time=0.0,
            )

        predictions_t1 = []
        actuals_t1 = []
        predictions_t3 = []
        actuals_t3 = []

        for t in range(n_windows):
            idx = start + t

            # Realised vol
            vol = np.std(returns[idx - vol_window:idx], axis=0)

            # Build adjacency from correlation
            if adjacency_series is not None and t < len(adjacency_series):
                adj = adjacency_series[t]
            else:
                ret_window = returns[idx - correlation_window:idx]
                with np.errstate(invalid="ignore"):
                    corr = np.corrcoef(ret_window, rowvar=False)
                corr = np.nan_to_num(corr, nan=0.0)
                adj = np.abs(corr)
                adj[adj < 0.3] = 0.0
                np.fill_diagonal(adj, 0.0)

            curv = (
                curvature_series[t]
                if curvature_series is not None and t < len(curvature_series)
                else None
            )

            # Predict
            pred = self.predict(vol, adj, curv)

            # Actual future vol
            if idx + 1 < returns.shape[0]:
                actual_vol_t1 = returns[idx]
                spike_thresh = np.quantile(actual_vol_t1, spike_threshold_quantile)
                actual_spike_t1 = (actual_vol_t1 > spike_thresh).astype(float)
                pred_score_t1 = pred.density_t1
                predictions_t1.append(pred_score_t1)
                actuals_t1.append(actual_spike_t1)

            if idx + 3 < returns.shape[0]:
                actual_vol_t3 = np.max(returns[idx:idx + 3], axis=0)
                spike_thresh_3 = np.quantile(actual_vol_t3, spike_threshold_quantile)
                actual_spike_t3 = (actual_vol_t3 > spike_thresh_3).astype(float)
                pred_score_t3 = pred.density_t3
                predictions_t3.append(pred_score_t3)
                actuals_t3.append(actual_spike_t3)

        # Compute ROC-AUC
        auc_t1 = self._compute_auc(predictions_t1, actuals_t1)
        auc_t3 = self._compute_auc(predictions_t3, actuals_t3)

        # Precision/Recall for t+1
        prec_t1, rec_t1 = self._precision_recall(predictions_t1, actuals_t1)

        return BacktestResult(
            n_windows=n_windows,
            roc_auc_t1=auc_t1,
            roc_auc_t3=auc_t3,
            precision_t1=prec_t1,
            recall_t1=rec_t1,
            mean_lead_time=1.0,  # by construction, prediction is 1 step ahead
        )

    @staticmethod
    def _compute_auc(
        predictions: list[NDArray],
        actuals: list[NDArray],
    ) -> float:
        """Simple ROC-AUC via Mann-Whitney U statistic."""
        if not predictions or not actuals:
            return 0.5

        pred = np.concatenate(predictions)
        actual = np.concatenate(actuals)

        pos = pred[actual > 0.5]
        neg = pred[actual <= 0.5]

        if pos.size == 0 or neg.size == 0:
            return 0.5

        # Mann-Whitney U
        n_pos = pos.size
        n_neg = neg.size
        u = 0.0
        for p in pos:
            u += np.sum(p > neg) + 0.5 * np.sum(p == neg)

        return float(u / (n_pos * n_neg))

    @staticmethod
    def _precision_recall(
        predictions: list[NDArray],
        actuals: list[NDArray],
    ) -> tuple[float, float]:
        """Precision and recall at median threshold."""
        if not predictions or not actuals:
            return 0.0, 0.0

        pred = np.concatenate(predictions)
        actual = np.concatenate(actuals)

        thresh = np.median(pred)
        pred_binary = (pred > thresh).astype(float)

        tp = np.sum(pred_binary * actual)
        fp = np.sum(pred_binary * (1 - actual))
        fn = np.sum((1 - pred_binary) * actual)

        precision = tp / max(tp + fp, 1e-12)
        recall = tp / max(tp + fn, 1e-12)
        return float(precision), float(recall)


__all__ = [
    "DiffusionVolatilityPredictor",
    "VolatilityFrontPrediction",
    "BacktestResult",
]
