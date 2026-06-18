# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from core.kuramoto.kuramoto_ricci_engine import (
    phase_transition_boundary,
    ricci_to_adjacency,
)


def test_ricci_to_adjacency_is_bounded_nonnegative_symmetric_zero_diagonal() -> None:
    kappa = jnp.asarray(
        [
            [0.8, -0.4, 0.2],
            [0.1, 0.7, -0.3],
            [0.5, 0.4, -0.9],
        ]
    )
    adjacency = ricci_to_adjacency(kappa)
    assert adjacency.shape == (3, 3)
    assert bool(jnp.all(adjacency >= 0.0))
    assert bool(jnp.allclose(adjacency, adjacency.T))
    assert bool(jnp.allclose(jnp.diag(adjacency), 0.0))
    assert float(jnp.max(adjacency)) <= 1.0


def test_curvature_input_must_be_mapped_before_boundary_calculation() -> None:
    kappa = jnp.asarray([[0.0, -0.2], [-0.2, 0.0]])
    with pytest.raises(ValueError, match="non-negative"):
        phase_transition_boundary(K=1.0, lorentzian_half_width=0.4, A=kappa)


def test_ricci_matched_comparison_keeps_bridge_experimental() -> None:
    kappa = jnp.asarray(
        [
            [0.0, 0.9, -0.4, 0.2],
            [0.9, 0.0, 0.1, -0.3],
            [-0.4, 0.1, 0.0, 0.7],
            [0.2, -0.3, 0.7, 0.0],
        ],
        dtype=jnp.float32,
    )
    ricci_adjacency = ricci_to_adjacency(kappa)
    weights = np.sort(np.asarray(ricci_adjacency[np.triu_indices(4, k=1)]))
    null_adjacency = np.zeros((4, 4), dtype=np.float32)
    edge_order = [(0, 2), (0, 3), (1, 2), (1, 3), (0, 1), (2, 3)]
    for (i, j), weight in zip(edge_order, weights, strict=True):
        null_adjacency[i, j] = null_adjacency[j, i] = float(weight)

    ricci_report = phase_transition_boundary(
        K=1.0,
        lorentzian_half_width=0.4,
        A=ricci_adjacency,
    )
    null_report = phase_transition_boundary(
        K=1.0,
        lorentzian_half_width=0.4,
        A=jnp.asarray(null_adjacency),
    )

    assert ricci_report.lambda_max_A >= 0.0
    assert null_report.lambda_max_A >= 0.0
    assert ricci_report.K_c > 0.0 or ricci_report.K_c == float("inf")
    assert null_report.K_c > 0.0 or null_report.K_c == float("inf")


def test_ricci_bridge_summary_records_slice_one_carriers() -> None:
    path = Path("artifacts/physics_validation/ricci_bridge_summary.json")
    summary = json.loads(path.read_text(encoding="utf-8"))
    ledger = summary["provenance_ledger"]
    metric = ledger["bridge_completeness_metric"]

    assert "source_inputs" in ledger
    assert "graph_rule" in ledger
    assert "curvature_estimator_name" in ledger
    assert "curvature_estimator_assumptions" in ledger
    assert "curvature_to_weight_transform" in ledger
    assert "bridge_valid_domain" in ledger
    assert "bridge_excluded_domain" in ledger
    assert "ricci_test_link" in ledger
    assert "bridge_completeness_metric" in ledger
    assert summary["bridge_non_core_flag"] is True
    assert metric["required_fields"] == 11
    assert metric["present_fields"] == 11
    assert metric["ratio"] == 1.0
    assert metric["score_credit_allowed"] is False
    assert summary["issue_1228_tasks_completed"] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
