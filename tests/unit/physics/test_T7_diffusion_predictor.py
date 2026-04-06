# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""T7 — Graph diffusion volatility front predictor tests."""

import numpy as np
import pytest

from core.physics.diffusion_predictor import (
    DiffusionVolatilityPredictor,
    VolatilityFrontPrediction,
)


@pytest.fixture
def predictor() -> DiffusionVolatilityPredictor:
    return DiffusionVolatilityPredictor(D_0=1.0, threshold_quantile=0.75)


@pytest.fixture
def simple_adjacency():
    """5-asset chain: 0—1—2—3—4."""
    n = 5
    adj = np.zeros((n, n))
    for i in range(n - 1):
        adj[i, i + 1] = adj[i + 1, i] = 0.5
    return adj


class TestVolatilityPropagation:
    """Volatility should spread from source along edges."""

    def test_front_from_single_source(self, predictor, simple_adjacency):
        vol = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        pred = predictor.predict(vol, simple_adjacency)
        assert isinstance(pred, VolatilityFrontPrediction)
        # After propagation, density should spread from node 0
        assert pred.density_t1[0] > 0
        assert pred.density_t1[1] > pred.density_t1[4]  # closer gets more

    def test_probability_conservation(self, predictor, simple_adjacency):
        vol = np.array([0.5, 0.2, 0.1, 0.1, 0.1])
        pred = predictor.predict(vol, simple_adjacency)
        assert abs(pred.density_t1.sum() - 1.0) < 1e-10
        assert abs(pred.density_t3.sum() - 1.0) < 1e-10

    def test_longer_horizon_more_diffused(self, predictor, simple_adjacency):
        vol = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        pred = predictor.predict(vol, simple_adjacency)
        # At t=3, density should be more spread than t=1
        std_t1 = np.std(pred.density_t1)
        std_t3 = np.std(pred.density_t3)
        assert std_t3 < std_t1, "Longer time → more uniform → lower std"


class TestCurvatureEffect:
    """Positive curvature → faster diffusion."""

    def test_positive_curvature_spreads_faster(self, predictor, simple_adjacency):
        vol = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        n = simple_adjacency.shape[0]

        curv_pos = np.full((n, n), 1.0)
        np.fill_diagonal(curv_pos, 0.0)
        curv_neg = np.full((n, n), -1.0)
        np.fill_diagonal(curv_neg, 0.0)

        pred_pos = predictor.predict(vol, simple_adjacency, curv_pos)
        pred_neg = predictor.predict(vol, simple_adjacency, curv_neg)

        # Positive curvature → node 0 loses density faster
        assert pred_pos.density_t1[0] < pred_neg.density_t1[0]


class TestFrontDetection:
    def test_front_indices(self, predictor, simple_adjacency):
        vol = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        pred = predictor.predict(vol, simple_adjacency)
        assert isinstance(pred.predicted_front, list)
        assert len(pred.predicted_front) > 0

    def test_threshold_affects_front_size(self):
        p_strict = DiffusionVolatilityPredictor(threshold_quantile=0.9)
        p_loose = DiffusionVolatilityPredictor(threshold_quantile=0.5)

        adj = np.ones((4, 4)) * 0.5
        np.fill_diagonal(adj, 0.0)
        vol = np.array([0.5, 0.3, 0.1, 0.1])

        front_strict = p_strict.predict(vol, adj).predicted_front
        front_loose = p_loose.predict(vol, adj).predicted_front
        assert len(front_strict) <= len(front_loose)


class TestBacktest:
    def test_basic_backtest(self, predictor):
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.normal(0, 1, (100, 5)), axis=0)

        result = predictor.backtest(
            prices, vol_window=5, correlation_window=15,
        )
        assert result.n_windows > 0
        assert 0 <= result.roc_auc_t1 <= 1
        assert 0 <= result.roc_auc_t3 <= 1
        assert 0 <= result.precision_t1 <= 1
        assert 0 <= result.recall_t1 <= 1

    def test_insufficient_data(self, predictor):
        result = predictor.backtest(np.ones((10, 3)))
        assert result.n_windows == 0
        assert result.roc_auc_t1 == 0.5  # random baseline


class TestDeterminism:
    def test_deterministic(self, predictor, simple_adjacency):
        vol = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
        p1 = predictor.predict(vol, simple_adjacency)
        p2 = predictor.predict(vol, simple_adjacency)
        np.testing.assert_array_equal(p1.density_t1, p2.density_t1)


class TestInputValidation:
    def test_bad_D0(self):
        with pytest.raises(ValueError):
            DiffusionVolatilityPredictor(D_0=0)

    def test_bad_quantile(self):
        with pytest.raises(ValueError):
            DiffusionVolatilityPredictor(threshold_quantile=0)
