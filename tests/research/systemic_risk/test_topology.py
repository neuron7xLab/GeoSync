# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Invariant tests for :mod:`research.systemic_risk.topology`."""

from __future__ import annotations

import numpy as np
import pytest

from research.systemic_risk.topology import (
    InterbankTopology,
    barabasi_albert_null,
    from_exposure_matrix,
)


class TestInterbankTopologyInvariants:
    def test_inv_top1_asymmetric_adjacency_rejected(self) -> None:
        a = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.int8)
        with pytest.raises(ValueError, match="INV-TOP1"):
            InterbankTopology(adjacency=a, node_labels=("a", "b", "c"), source_label="t")

    def test_inv_top1_nonbinary_rejected(self) -> None:
        a = np.array([[0, 2], [2, 0]], dtype=np.int8)
        with pytest.raises(ValueError, match="INV-TOP1"):
            InterbankTopology(adjacency=a, node_labels=("a", "b"), source_label="t")

    def test_inv_top1_diagonal_must_be_zero(self) -> None:
        a = np.array([[1, 1], [1, 1]], dtype=np.int8)
        with pytest.raises(ValueError, match="INV-TOP1"):
            InterbankTopology(adjacency=a, node_labels=("a", "b"), source_label="t")

    def test_inv_top3_label_length_mismatch_rejected(self) -> None:
        a = np.zeros((3, 3), dtype=np.int8)
        with pytest.raises(ValueError, match="INV-TOP3"):
            InterbankTopology(adjacency=a, node_labels=("a", "b"), source_label="t")

    def test_inv_top2_negative_weights_rejected(self) -> None:
        a = np.array([[0, 1], [1, 0]], dtype=np.int8)
        w = np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="INV-TOP2"):
            InterbankTopology(adjacency=a, weights=w, node_labels=("a", "b"), source_label="t")

    def test_arrays_are_immutable_after_construction(self) -> None:
        a = np.array([[0, 1], [1, 0]], dtype=np.int8)
        topo = InterbankTopology(adjacency=a, node_labels=("a", "b"), source_label="t")
        with pytest.raises(ValueError):
            topo.adjacency[0, 1] = 0


class TestFromExposureMatrix:
    def test_symmetrises_and_binarises(self) -> None:
        e = np.array([[0.0, 1.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
        topo = from_exposure_matrix(e, node_labels=("a", "b", "c"))
        # (1+3)/2 = 2 → adjacency edge a-b; rest zero.
        assert topo.adjacency[0, 1] == 1
        assert topo.adjacency[1, 0] == 1
        assert topo.adjacency[2, 0] == 0
        assert topo.adjacency[2, 1] == 0

    def test_threshold_excludes_small_exposures(self) -> None:
        e = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64)
        topo = from_exposure_matrix(e, node_labels=("a", "b"), threshold=1.0)
        assert topo.adjacency[0, 1] == 0

    def test_negative_exposure_rejected(self) -> None:
        e = np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError):
            from_exposure_matrix(e, node_labels=("a", "b"))


class TestBarabasiAlbertNull:
    @pytest.mark.parametrize("m", [2, 3, 4])
    def test_topology_invariants_hold(self, m: int) -> None:
        topo = barabasi_albert_null(n_nodes=50, m=m, seed=42)
        assert topo.adjacency.shape == (50, 50)
        assert np.array_equal(topo.adjacency, topo.adjacency.T)
        assert np.all(np.diag(topo.adjacency) == 0)
        assert np.all((topo.adjacency == 0) | (topo.adjacency == 1))

    def test_seed_determinism(self) -> None:
        a = barabasi_albert_null(n_nodes=30, m=3, seed=7)
        b = barabasi_albert_null(n_nodes=30, m=3, seed=7)
        assert np.array_equal(a.adjacency, b.adjacency)

    def test_different_seeds_differ(self) -> None:
        a = barabasi_albert_null(n_nodes=50, m=3, seed=1)
        b = barabasi_albert_null(n_nodes=50, m=3, seed=2)
        assert not np.array_equal(a.adjacency, b.adjacency)

    def test_mean_degree_close_to_2m(self) -> None:
        # Statistical: ensemble of 50 BA graphs, mean degree → 2m as N → ∞.
        # At N=200, m=3 the empirical mean is 2m within ~10% (Albert-Barabási 2002).
        m = 3
        n = 200
        means = []
        for seed in range(50):
            topo = barabasi_albert_null(n_nodes=n, m=m, seed=seed)
            means.append(float(topo.degree.mean()))
        ensemble_mean = float(np.mean(means))
        assert abs(ensemble_mean - 2 * m) < 0.5, (
            f"BA mean-degree drift: ensemble_mean={ensemble_mean:.3f}, "
            f"expected 2m={2 * m} ± 0.5 at N={n}, m={m}, n_seeds=50"
        )

    def test_invalid_m_rejected(self) -> None:
        with pytest.raises(ValueError):
            barabasi_albert_null(n_nodes=10, m=0, seed=0)
        with pytest.raises(ValueError):
            barabasi_albert_null(n_nodes=3, m=5, seed=0)
