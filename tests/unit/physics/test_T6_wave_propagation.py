# SPDX-License-Identifier: MIT
"""T6 — Graph Diffusion Engine tests.

Fokker-Planck on graph Laplacian. NOT Maxwell literally.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.physics.wave_propagation import GraphDiffusionEngine


@pytest.fixture
def engine() -> GraphDiffusionEngine:
    return GraphDiffusionEngine(D_0=1.0)


@pytest.fixture
def simple_adjacency():
    """3-node path graph: 0—1—2."""
    return np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])


class TestDiffusionUniformForZeroCurvature:
    """With zero curvature, engine reduces to standard graph diffusion."""

    def test_zero_curvature_equals_no_curvature(self, engine, simple_adjacency):
        L_no_curv = engine.build_laplacian(simple_adjacency, curvature=None)
        L_zero_curv = engine.build_laplacian(
            simple_adjacency,
            curvature=np.zeros_like(simple_adjacency),
        )
        np.testing.assert_allclose(L_no_curv, L_zero_curv, atol=1e-12)


class TestDiffusionFasterOnPositiveCurvature:
    """Positive curvature → stronger coupling → faster diffusion."""

    def test_positive_curvature_speeds_diffusion(self, engine, simple_adjacency):
        L_zero = engine.build_laplacian(simple_adjacency)
        curv_pos = np.full_like(simple_adjacency, 1.0)
        np.fill_diagonal(curv_pos, 0.0)
        L_pos = engine.build_laplacian(simple_adjacency, curvature=curv_pos)

        # With positive curvature, weights increase → larger Laplacian eigenvalues
        eig_zero = np.sort(np.linalg.eigvalsh(L_zero))
        eig_pos = np.sort(np.linalg.eigvalsh(L_pos))
        # Fiedler value (2nd smallest) should be larger for positive curvature
        assert eig_pos[1] > eig_zero[1], "Positive curvature should increase algebraic connectivity"


class TestLaplacianEigenvaluesNonNegative:
    """Graph Laplacian of undirected graph has non-negative eigenvalues."""

    def test_eigenvalues_non_negative(self, engine, simple_adjacency):
        L = engine.build_laplacian(simple_adjacency)
        eigenvalues = engine.laplacian_eigenvalues(L)
        assert np.all(eigenvalues >= -1e-10), f"Negative eigenvalue found: {eigenvalues}"

    def test_smallest_eigenvalue_zero(self, engine, simple_adjacency):
        L = engine.build_laplacian(simple_adjacency)
        eigenvalues = engine.laplacian_eigenvalues(L)
        assert abs(eigenvalues[0]) < 1e-10, "Smallest eigenvalue should be ≈ 0"


class TestProbabilityConservation:
    """Σρ_i = 1 at all times."""

    def test_sum_preserved(self, engine, simple_adjacency):
        L = engine.build_laplacian(simple_adjacency)
        rho_0 = np.array([1.0, 0.0, 0.0])  # all mass at node 0

        for t in [0.1, 0.5, 1.0, 5.0, 10.0]:
            rho_t = engine.propagate(rho_0, L, t)
            assert abs(rho_t.sum() - 1.0) < 1e-10, f"Probability not conserved at t={t}"
            assert np.all(rho_t >= -1e-10), f"Negative density at t={t}"

    def test_converges_to_uniform(self, engine, simple_adjacency):
        """Long-time diffusion → uniform distribution."""
        L = engine.build_laplacian(simple_adjacency)
        rho_0 = np.array([1.0, 0.0, 0.0])
        rho_t = engine.propagate(rho_0, L, t=100.0)
        expected = np.ones(3) / 3
        np.testing.assert_allclose(rho_t, expected, atol=1e-6)


class TestVolatilityFront:
    def test_initial_front(self, engine):
        rho = np.array([0.5, 0.3, 0.2])
        front = engine.volatility_front(rho, threshold=0.25)
        assert 0 in front
        assert 1 in front
        assert 2 not in front

    def test_with_asset_names(self, engine):
        rho = np.array([0.6, 0.1, 0.3])
        front = engine.volatility_front(rho, threshold=0.25, asset_names=["BTC", "ETH", "SOL"])
        assert "BTC" in front
        assert "SOL" in front
        assert "ETH" not in front


class TestZeroPropagation:
    def test_t_zero_returns_input(self, engine, simple_adjacency):
        L = engine.build_laplacian(simple_adjacency)
        rho_0 = np.array([0.5, 0.3, 0.2])
        rho_t = engine.propagate(rho_0, L, t=0.0)
        np.testing.assert_allclose(rho_t, rho_0)

    def test_negative_time_raises(self, engine, simple_adjacency):
        L = engine.build_laplacian(simple_adjacency)
        with pytest.raises(ValueError, match="≥ 0"):
            engine.propagate(np.array([1.0, 0.0, 0.0]), L, t=-1.0)


class TestInputValidation:
    def test_D0_positive(self):
        with pytest.raises(ValueError):
            GraphDiffusionEngine(D_0=0.0)

    def test_non_square_adjacency(self, engine):
        with pytest.raises(ValueError, match="square"):
            engine.build_laplacian(np.ones((2, 3)))
