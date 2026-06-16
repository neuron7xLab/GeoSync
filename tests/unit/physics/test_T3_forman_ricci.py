# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""T3 — Forman-Ricci curvature tests."""

import numpy as np
import pytest

from core.physics.forman_ricci import (
    DualTrackRicciMonitor,
    FormanRicciCurvature,
    FormanRicciResult,
)


@pytest.fixture
def frc() -> FormanRicciCurvature:
    return FormanRicciCurvature(threshold=0.3)


@pytest.fixture
def complete_graph_corr():
    """4-node complete graph (all corr = 0.8)."""
    n = 4
    corr = np.full((n, n), 0.8)
    np.fill_diagonal(corr, 1.0)
    return corr


@pytest.fixture
def star_graph_corr():
    """Star graph: node 0 connected to all, others disconnected."""
    n = 5
    corr = np.eye(n)
    for i in range(1, n):
        corr[0, i] = corr[i, 0] = 0.6
    return corr


class TestFormanCurvatureFormula:
    """κ_F(i,j) = 4 - d_i - d_j + 3·T_ij."""

    def test_complete_graph_positive_curvature(self, frc, complete_graph_corr):
        """Complete K4: every edge in 2 triangles, degree=3.
        κ = 4 - 3 - 3 + 3·2 = 4 > 0."""
        result = frc.compute_from_correlation(complete_graph_corr)
        assert all(v > 0 for v in result.edge_curvatures.values())
        assert result.kappa_min > 0

    def test_star_graph_negative_curvature(self, frc, star_graph_corr):
        """Star: no triangles, hub degree=4, leaf degree=1.
        κ = 4 - 4 - 1 + 0 = -1."""
        result = frc.compute_from_correlation(star_graph_corr)
        assert result.kappa_min < 0

    def test_single_edge_curvature(self, frc):
        """2 nodes: d_i=d_j=1, T=0. κ = 4-1-1+0 = 2."""
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = frc.compute_from_correlation(corr)
        assert len(result.edge_curvatures) == 1
        assert abs(list(result.edge_curvatures.values())[0] - 2.0) < 1e-10


class TestHerdingDetection:
    """κ_min → 0 = herding."""

    def test_herding_index(self, frc, complete_graph_corr):
        result = frc.compute_from_correlation(complete_graph_corr)
        assert result.herding_index > 0, "Complete graph should show herding"

    def test_fragmented_low_herding(self, frc, star_graph_corr):
        result = frc.compute_from_correlation(star_graph_corr)
        assert result.herding_index < 1.0


class TestDualTrackMonitor:
    def test_update_and_margin(self):
        monitor = DualTrackRicciMonitor(forman_threshold=0.3, correlation_window=10)
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.normal(0, 1, (30, 5)), axis=0)

        result = monitor.update(prices)
        assert isinstance(result, FormanRicciResult)
        margin = monitor.margin_multiplier()
        assert margin >= 1.0, "Margin multiplier should be ≥ 1"

    def test_herding_increases_margin(self):
        monitor = DualTrackRicciMonitor(margin_sensitivity=3.0)
        # Simulate herding result
        herding = FormanRicciResult(
            edge_curvatures={(0, 1): 1.0},
            kappa_min=0.5,
            kappa_mean=0.5,
            kappa_max=0.5,
            herding_index=1.0,
        )
        normal = FormanRicciResult(
            edge_curvatures={(0, 1): -3.0},
            kappa_min=-3.0,
            kappa_mean=-3.0,
            kappa_max=-3.0,
            herding_index=0.0,
        )
        assert monitor.margin_multiplier(herding) > monitor.margin_multiplier(normal)

    def test_is_herding(self):
        monitor = DualTrackRicciMonitor()
        assert monitor.is_herding(
            FormanRicciResult(
                edge_curvatures={},
                kappa_min=0.0,
                kappa_mean=0.0,
                kappa_max=0.0,
                herding_index=0.0,
            )
        )
        assert not monitor.is_herding(
            FormanRicciResult(
                edge_curvatures={},
                kappa_min=-5.0,
                kappa_mean=-5.0,
                kappa_max=-5.0,
                herding_index=0.0,
            )
        )

    def test_fragility_trend(self):
        monitor = DualTrackRicciMonitor(forman_threshold=0.3, correlation_window=10)
        rng = np.random.default_rng(7)
        for t in range(15):
            prices = 100 + np.cumsum(rng.normal(0, 1, (20, 4)), axis=0)
            monitor.update(prices)
        trend = monitor.fragility_trend(lookback=10)
        assert np.isfinite(trend)


class TestComputeFromPrices:
    def test_basic_computation(self, frc):
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.normal(0, 1, (60, 5)), axis=0)
        result = frc.compute_from_prices(prices, window=20)
        assert isinstance(result, FormanRicciResult)
        assert np.isfinite(result.kappa_mean)


class TestInputValidation:
    def test_threshold_bounds(self):
        with pytest.raises(ValueError):
            FormanRicciCurvature(threshold=0.0)
        with pytest.raises(ValueError):
            FormanRicciCurvature(threshold=1.0)

    def test_non_square_corr(self, frc):
        with pytest.raises(ValueError):
            frc.compute_from_correlation(np.ones((3, 4)))

    def test_insufficient_prices(self, frc):
        with pytest.raises(ValueError):
            frc.compute_from_prices(np.ones((1, 5)))


class TestRepresentationHonesty:
    """Metamorphic locks for the declared unweighted/unsigned semantics.

    These tests do not change behaviour; they pin the *actual* semantics of
    the binarized ``|ρ| > threshold`` graph so the module cannot silently
    drift to claiming weight- or sign-sensitivity it does not have. The
    provenance fields are asserted to declare exactly the information the
    projection discards.
    """

    def test_provenance_fields_declare_unweighted_unsigned(self) -> None:
        frc = FormanRicciCurvature(threshold=0.3)
        corr = np.full((4, 4), 0.8)
        np.fill_diagonal(corr, 1.0)
        result = frc.compute_from_correlation(corr)
        assert result.weighted is False
        assert result.signed is False
        assert result.representation == "graph"
        assert "edge_weight_magnitude" in result.information_loss
        assert "correlation_sign" in result.information_loss

    def test_weight_insensitivity_is_declared_not_hidden(self) -> None:
        # Two graphs with IDENTICAL above-threshold topology but different
        # magnitudes must produce identical curvature — the honest signature
        # of an unweighted method. The result declares weighted=False so this
        # invariance is documented, not a masked "weighted" claim.
        frc = FormanRicciCurvature(threshold=0.3)
        n = 4
        weak = np.full((n, n), 0.4)
        np.fill_diagonal(weak, 1.0)
        strong = np.full((n, n), 0.95)
        np.fill_diagonal(strong, 1.0)
        r_weak = frc.compute_from_correlation(weak)
        r_strong = frc.compute_from_correlation(strong)
        assert r_weak.edge_curvatures == r_strong.edge_curvatures
        assert r_weak.weighted is False and r_strong.weighted is False

    def test_sign_erasure_is_recorded(self) -> None:
        # ρ = +0.8 and ρ = −0.8 collapse to the same edge under |ρ|, so the
        # anti-correlation case yields identical output. This is real
        # information loss and must be declared via information_loss / signed.
        frc = FormanRicciCurvature(threshold=0.3)
        n = 4
        pos = np.full((n, n), 0.8)
        np.fill_diagonal(pos, 1.0)
        neg = -np.full((n, n), 0.8)
        np.fill_diagonal(neg, 1.0)
        r_pos = frc.compute_from_correlation(pos)
        r_neg = frc.compute_from_correlation(neg)
        assert r_pos.edge_curvatures == r_neg.edge_curvatures
        assert r_pos.signed is False
        assert "correlation_sign" in r_pos.information_loss

    def test_empty_graph_result_also_declares_provenance(self) -> None:
        # A below-threshold (edgeless) correlation matrix still returns a
        # provenance-bearing result, not a bare zero with no declared loss.
        frc = FormanRicciCurvature(threshold=0.9)
        corr = np.eye(3) + 0.1 * (np.ones((3, 3)) - np.eye(3))
        result = frc.compute_from_correlation(corr)
        assert result.edge_curvatures == {}
        assert result.weighted is False
        assert result.information_loss == ("edge_weight_magnitude", "correlation_sign")
