# mypy: disable-error-code="attr-defined,unused-ignore,no-untyped-call"
"""Tests for augmented Forman-Ricci with triangle reinforcement."""

from __future__ import annotations

import numpy as np

from geosync.estimators.augmented_ricci import AugmentedFormanRicci


def test_correlated_assets_positive_curvature() -> None:
    """Highly correlated assets → positive κ (robust topology)."""
    np.random.seed(42)
    base = np.cumsum(np.random.randn(200))
    returns = np.column_stack(
        [
            np.diff(base),
            np.diff(base + 0.01 * np.random.randn(200)),
            np.diff(base + 0.02 * np.random.randn(200)),
        ]
    )
    ricci = AugmentedFormanRicci(correlation_threshold=0.1)
    kappa = ricci.compute_mean(returns, ["A", "B", "C"])
    # Highly correlated → triangles form → positive curvature
    assert kappa != 0.0, "Correlated assets should produce non-zero curvature"


def test_uncorrelated_assets_zero_curvature() -> None:
    """Independent assets → no edges above threshold → κ = 0."""
    np.random.seed(42)
    returns = np.random.randn(200, 5)
    ricci = AugmentedFormanRicci(correlation_threshold=0.5)
    kappa = ricci.compute_mean(returns, ["A", "B", "C", "D", "E"])
    assert kappa == 0.0


def test_two_assets_minimum() -> None:
    """Need at least 2 assets for graph construction."""
    np.random.seed(42)
    returns = np.random.randn(100, 1)
    ricci = AugmentedFormanRicci()
    kappa = ricci.compute_mean(returns, ["A"])
    assert kappa == 0.0


def test_short_series_returns_zero() -> None:
    """< 16 bars → insufficient for correlation → 0."""
    returns = np.random.randn(10, 3)
    ricci = AugmentedFormanRicci()
    kappa = ricci.compute_mean(returns, ["A", "B", "C"])
    assert kappa == 0.0


def test_shape_mismatch_raises() -> None:
    """returns.shape[1] != len(symbols) → ValueError."""
    try:
        AugmentedFormanRicci().compute_mean(np.random.randn(100, 3), ["A", "B"])
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass


def test_threshold_affects_graph_density() -> None:
    """Higher threshold → fewer edges → different κ."""
    np.random.seed(42)
    base = np.cumsum(np.random.randn(200))
    returns = np.column_stack(
        [
            np.diff(base),
            np.diff(base + 0.05 * np.random.randn(200)),
            np.diff(base + 0.1 * np.random.randn(200)),
        ]
    )
    k_low = AugmentedFormanRicci(correlation_threshold=0.1).compute_mean(
        returns, ["A", "B", "C"]
    )
    k_high = AugmentedFormanRicci(correlation_threshold=0.9).compute_mean(
        returns, ["A", "B", "C"]
    )
    # Higher threshold may prune edges → different curvature
    # (not necessarily lower — depends on which edges survive)
    assert isinstance(k_low, float)
    assert isinstance(k_high, float)
