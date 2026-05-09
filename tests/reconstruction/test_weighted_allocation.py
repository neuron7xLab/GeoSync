# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Tests for weighted_allocation.py + GATE_4 (reproducibility)."""

from __future__ import annotations

import numpy as np
import pytest

from research.reconstruction.weighted_allocation import (
    allocate_weights,
    sample_adjacency_bernoulli,
)


def _marginals(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s = rng.lognormal(mean=10.0, sigma=1.0, size=n)
    s_in = s.copy()
    rng.shuffle(s_in)
    # Force conservation of mass.
    s_in *= s.sum() / s_in.sum()
    return s, s_in


def test_sample_adjacency_diagonal_zero() -> None:
    rng = np.random.default_rng(42)
    p = np.full((30, 30), 0.5)
    a = sample_adjacency_bernoulli(p, rng=rng)
    assert np.all(np.diag(a) == 0)


def test_sample_adjacency_uint8_dtype() -> None:
    rng = np.random.default_rng(42)
    p = np.full((10, 10), 0.5)
    a = sample_adjacency_bernoulli(p, rng=rng)
    assert a.dtype == np.uint8


def test_sample_adjacency_in_zero_one() -> None:
    rng = np.random.default_rng(42)
    p = np.random.default_rng(0).uniform(size=(20, 20))
    a = sample_adjacency_bernoulli(p, rng=rng)
    assert set(np.unique(a).tolist()).issubset({0, 1})


def test_sample_adjacency_bit_exact_reproducibility() -> None:
    """GATE_4: identical seeds + p ⇒ identical A."""
    p = np.random.default_rng(0).uniform(size=(40, 40))
    a1 = sample_adjacency_bernoulli(p, rng=np.random.default_rng(99))
    a2 = sample_adjacency_bernoulli(p, rng=np.random.default_rng(99))
    np.testing.assert_array_equal(a1, a2)


def test_sample_adjacency_rejects_p_out_of_unit() -> None:
    rng = np.random.default_rng(42)
    p_bad = np.array([[0.5, 1.5], [0.0, 0.0]])
    with pytest.raises(ValueError):
        sample_adjacency_bernoulli(p_bad, rng=rng)


def test_sample_adjacency_rejects_non_square() -> None:
    rng = np.random.default_rng(42)
    p_bad = np.full((4, 5), 0.5)
    with pytest.raises(ValueError):
        sample_adjacency_bernoulli(p_bad, rng=rng)


def test_sample_adjacency_orphan_repair_default_on() -> None:
    """Orphan rows/cols get a forced edge by default."""
    rng = np.random.default_rng(42)
    p = np.full((30, 30), 1e-12)  # essentially zero edge probability
    p[0, 1] = 1.0  # one strong edge
    a = sample_adjacency_bernoulli(p, rng=rng)
    # Every row and every col should have ≥ 1 edge
    assert np.all(a.sum(axis=1) >= 1)
    assert np.all(a.sum(axis=0) >= 1)


def test_sample_adjacency_orphan_repair_can_be_disabled() -> None:
    rng = np.random.default_rng(42)
    p = np.zeros((30, 30))
    a = sample_adjacency_bernoulli(p, rng=rng, guarantee_row_col_support=False)
    assert a.sum() == 0


def test_allocate_weights_recovers_marginals_after_ipf() -> None:
    """IPF projects W onto the marginal slice — row/col sums match exactly.

    Uses p=0.30 to guarantee transportation feasibility on heavy-tailed
    marginals; sparser supports may leave residual error that Gate 5
    metrics will catch, which is the X-10R contract.
    """
    n = 60
    s_out, s_in = _marginals(n, seed=21)
    rng = np.random.default_rng(7)
    p = np.full((n, n), 0.30)
    a = sample_adjacency_bernoulli(p, rng=rng)
    w = allocate_weights(a, s_out, s_in)
    np.testing.assert_allclose(w.sum(axis=1), s_out, rtol=0, atol=1e-6 * s_out.max())
    np.testing.assert_allclose(w.sum(axis=0), s_in, rtol=0, atol=1e-6 * s_in.max())


def test_allocate_weights_diagonal_zero() -> None:
    n = 40
    s_out, s_in = _marginals(n, seed=22)
    a = np.ones((n, n), dtype=np.uint8)
    np.fill_diagonal(a, 0)
    w = allocate_weights(a, s_out, s_in)
    assert np.all(np.diag(w) == 0.0)


def test_allocate_weights_total_mass_conserved() -> None:
    n = 50
    s_out, s_in = _marginals(n, seed=23)
    rng = np.random.default_rng(7)
    p = np.full((n, n), 0.08)
    a = sample_adjacency_bernoulli(p, rng=rng)
    w = allocate_weights(a, s_out, s_in)
    assert abs(w.sum() - s_in.sum()) / s_in.sum() < 1e-6


def test_allocate_weights_rejects_negative_marginals() -> None:
    n = 10
    a = np.ones((n, n), dtype=np.uint8)
    np.fill_diagonal(a, 0)
    s = np.ones(n)
    s_bad = -np.ones(n)
    with pytest.raises(ValueError):
        allocate_weights(a, s_bad, s)


def test_allocate_weights_rejects_zero_total_mass() -> None:
    n = 10
    a = np.ones((n, n), dtype=np.uint8)
    np.fill_diagonal(a, 0)
    s = np.zeros(n)
    with pytest.raises(ValueError):
        allocate_weights(a, s, s)


def test_allocate_weights_rejects_non_square_adjacency() -> None:
    s = np.ones(4)
    with pytest.raises(ValueError):
        allocate_weights(np.zeros((4, 5), dtype=np.uint8), s, s)


def test_allocate_weights_rejects_infeasible_support() -> None:
    """A row with positive marginal but zero edges → ValueError."""
    n = 10
    a = np.zeros((n, n), dtype=np.uint8)
    a[1, 0] = 1
    a[0, 1] = 1  # only nodes 0 and 1 have edges
    s_out = np.ones(n) * 100.0  # all nodes have positive marginal
    s_in = s_out.copy()
    with pytest.raises(ValueError, match="infeasible"):
        allocate_weights(a, s_out, s_in)
