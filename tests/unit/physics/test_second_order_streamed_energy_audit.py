# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Public-contract tests for streamed second-order swing-energy audit."""

from __future__ import annotations

import numpy as np
import pytest

from core.kuramoto.config import KuramotoConfig
from core.kuramoto.second_order import SecondOrderKuramotoEngine


def _dense_default_complete_graph_energy(
    theta: np.ndarray,
    vel: np.ndarray,
    *,
    coupling_strength: float,
) -> np.ndarray:
    """Dense reference for the default symmetric complete graph energy."""
    n = theta.shape[1]
    mass = np.ones(n, dtype=np.float64)
    omega = np.zeros(n, dtype=np.float64)
    adj = np.full((n, n), coupling_strength / n, dtype=np.float64)
    np.fill_diagonal(adj, 0.0)

    delta = theta[:, None, :] - theta[:, :, None]
    energy = 0.5 * (mass * vel * vel).sum(axis=1)
    energy += -(omega * theta).sum(axis=1)
    energy += -0.5 * np.einsum("ij,tij->t", adj, np.cos(delta))
    return energy


def test_streamed_audit_drift_matches_dense_reference() -> None:
    """The optimized audit must preserve the original swing-energy drift."""
    n = 6
    cfg = KuramotoConfig(
        N=n,
        K=2.0,
        omega=np.zeros(n),
        theta0=np.linspace(0.0, 1.0, n),
        dt=0.005,
        steps=400,
        seed=0,
    )
    engine = SecondOrderKuramotoEngine(
        cfg, mass=1.0, damping=0.0, velocity0=0.1 * np.ones(n)
    )
    result = engine.run()

    dense_energy = _dense_default_complete_graph_energy(
        result.phases,
        result.velocities,
        coupling_strength=cfg.K,
    )
    dense_scale = float(np.abs(dense_energy).max()) + 1e-12
    dense_drift = float((dense_energy.max() - dense_energy.min()) / dense_scale)

    report = engine.audit()
    assert report.energy_like_drift == pytest.approx(dense_drift, rel=1e-12, abs=1e-12)
    assert report.audit_passed is True


def test_streamed_audit_metadata_declares_memory_contract() -> None:
    """The audit declares the evaluator and its bounded pairwise working memory."""
    n = 4
    cfg = KuramotoConfig(
        N=n,
        K=1.0,
        omega=np.zeros(n),
        theta0=np.linspace(0.0, 0.5, n),
        dt=0.005,
        steps=50,
        seed=0,
    )
    report = SecondOrderKuramotoEngine(cfg, mass=1.0, damping=0.0).audit()

    assert report.solver_metadata["energy_evaluator"] == "streamed_pairwise_potential"
    assert report.solver_metadata["energy_peak_memory"] == "O(N^2)"
