# SPDX-License-Identifier: MIT
"""T4 — Higher-order Kuramoto with triadic interaction tests."""

import numpy as np
import pytest

from core.physics.higher_order_kuramoto import (
    HigherOrderKuramotoEngine,
    HigherOrderKuramotoResult,
    build_triangle_index,
    find_triangles,
)


@pytest.fixture
def engine() -> HigherOrderKuramotoEngine:
    return HigherOrderKuramotoEngine(
        sigma1=1.0, sigma2=0.5, dt=0.01, steps=200,
        correlation_threshold=0.3,
    )


@pytest.fixture
def complete_corr_4():
    """K4 complete graph correlation."""
    n = 4
    corr = np.full((n, n), 0.8)
    np.fill_diagonal(corr, 1.0)
    return corr


class TestTriangleDetection:
    def test_complete_graph(self):
        """K4 has C(4,3) = 4 triangles."""
        adj = np.ones((4, 4), dtype=bool)
        np.fill_diagonal(adj, False)
        triangles = find_triangles(adj)
        assert len(triangles) == 4

    def test_path_graph_no_triangles(self):
        """Path graph 0-1-2-3 has no triangles."""
        adj = np.zeros((4, 4), dtype=bool)
        adj[0, 1] = adj[1, 0] = True
        adj[1, 2] = adj[2, 1] = True
        adj[2, 3] = adj[3, 2] = True
        triangles = find_triangles(adj)
        assert len(triangles) == 0

    def test_single_triangle(self):
        """Triangle 0-1-2."""
        adj = np.zeros((3, 3), dtype=bool)
        adj[0, 1] = adj[1, 0] = True
        adj[1, 2] = adj[2, 1] = True
        adj[0, 2] = adj[2, 0] = True
        triangles = find_triangles(adj)
        assert len(triangles) == 1
        assert triangles[0] == (0, 1, 2)


class TestTriangleIndex:
    def test_index_completeness(self):
        triangles = [(0, 1, 2)]
        index = build_triangle_index(3, triangles)
        assert len(index[0]) == 1
        assert len(index[1]) == 1
        assert len(index[2]) == 1


class TestHigherOrderDynamics:
    """Triadic coupling should produce different dynamics than pairwise only."""

    def test_valid_result(self, engine, complete_corr_4):
        result = engine.run(complete_corr_4, seed=42)
        assert isinstance(result, HigherOrderKuramotoResult)
        assert result.phases.shape == (201, 4)
        assert result.order_parameter.shape == (201,)
        assert result.time.shape == (201,)

    def test_R_bounded(self, engine, complete_corr_4):
        result = engine.run(complete_corr_4, seed=42)
        assert np.all(result.order_parameter >= 0)
        assert np.all(result.order_parameter <= 1)

    def test_finite_outputs(self, engine, complete_corr_4):
        result = engine.run(complete_corr_4, seed=42)
        assert np.all(np.isfinite(result.phases))
        assert np.all(np.isfinite(result.order_parameter))
        assert np.all(np.isfinite(result.triadic_contribution))

    def test_triangles_detected(self, engine, complete_corr_4):
        result = engine.run(complete_corr_4, seed=42)
        assert result.n_triangles == 4  # K4 has 4 triangles

    def test_triadic_contribution_nonzero(self, engine, complete_corr_4):
        """With triangles, σ₂ term should contribute."""
        result = engine.run(complete_corr_4, seed=42)
        assert np.max(result.triadic_contribution) > 0

    def test_no_triadic_for_tree(self, engine):
        """Tree graph (no triangles) → triadic contribution = 0."""
        # Path correlation: only adjacent pairs correlated
        n = 4
        corr = np.eye(n)
        corr[0, 1] = corr[1, 0] = 0.5
        corr[1, 2] = corr[2, 1] = 0.5
        corr[2, 3] = corr[3, 2] = 0.5

        result = engine.run(corr, seed=42)
        assert result.n_triangles == 0
        assert np.all(result.triadic_contribution == 0)


class TestPairwiseVsHigherOrder:
    """Higher-order should detect clusters pairwise misses."""

    def test_sigma2_affects_dynamics(self):
        corr = np.full((4, 4), 0.8)
        np.fill_diagonal(corr, 1.0)

        e_pair = HigherOrderKuramotoEngine(sigma1=1.0, sigma2=0.0, steps=200)
        e_both = HigherOrderKuramotoEngine(sigma1=1.0, sigma2=1.0, steps=200)

        r_pair = e_pair.run(corr, seed=42)
        r_both = e_both.run(corr, seed=42)

        # Dynamics should differ
        assert not np.allclose(r_pair.order_parameter, r_both.order_parameter, atol=0.01)


class TestFromPrices:
    def test_run_from_prices(self, engine):
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.normal(0, 1, (80, 5)), axis=0)
        result = engine.run_from_prices(prices, window=30, seed=42)
        assert isinstance(result, HigherOrderKuramotoResult)
        assert np.all(np.isfinite(result.order_parameter))


class TestDeterminism:
    def test_deterministic(self, engine, complete_corr_4):
        r1 = engine.run(complete_corr_4, seed=42)
        r2 = engine.run(complete_corr_4, seed=42)
        np.testing.assert_array_equal(r1.phases, r2.phases)


class TestInputValidation:
    def test_bad_dt(self):
        with pytest.raises(ValueError):
            HigherOrderKuramotoEngine(dt=0)

    def test_bad_steps(self):
        with pytest.raises(ValueError):
            HigherOrderKuramotoEngine(steps=0)
