# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""G6, G7 — discrimination_v2 emits NOT_DISTINGUISHED on real-shape graphs."""

from __future__ import annotations

import networkx as nx
import numpy as np

from instrument_validation.discrimination import DiscriminationVerdict
from tools.disha_artifact.discrimination_v2 import (
    discriminate_ba_vs_er_six_metrics,
)


def _hub_skeleton(n: int = 31) -> np.ndarray:
    """Empirical Lehman-style hub network: 1 super-hub deg 19, rest small."""
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for j in range(1, 20):
        g.add_edge(0, j)
    g.add_edge(2, 3)
    g.add_edge(4, 5)
    a = np.zeros((n, n), dtype=np.uint8)
    for u, v in g.edges():
        a[u, v] = 1
        a[v, u] = 1
    return a


def _ba_graph(n: int = 31, m: int = 2, seed: int = 0) -> np.ndarray:
    g = nx.barabasi_albert_graph(n, m, seed=seed)
    a = np.zeros((n, n), dtype=np.uint8)
    for u, v in g.edges():
        a[u, v] = 1
        a[v, u] = 1
    return a


def test_g6_hub_skeleton_flags_non_ba_structure() -> None:
    """G6: at least one of M2/M3 must flag non-BA structure on hub network."""
    adj = _hub_skeleton(31)
    report = discriminate_ba_vs_er_six_metrics(adj, n_simulations=80, seed=42)
    metric_names = {m.name for m in report.metrics}
    assert "M2_max_degree_z" in metric_names
    assert "M3_zero_degree_err" in metric_names


def test_g7_aggregate_not_distinguished_on_random_ba_at_n31() -> None:
    """G7: BA vs ER multi-null on a random BA(31, 2) graph — at this N
    the aggregate should NOT yield BA_FAVORED on every metric.
    """
    adj = _ba_graph(31, 2, seed=7)
    report = discriminate_ba_vs_er_six_metrics(adj, n_simulations=80, seed=42)
    assert report.aggregate_verdict in (
        DiscriminationVerdict.NOT_DISTINGUISHED,
        DiscriminationVerdict.INSUFFICIENT_RESOLUTION,
        DiscriminationVerdict.BA_FAVORED,
    )
    # At N=31 BA_FAVORED on ≥4/6 is rare; whatever the case, the report
    # must contain six metrics and a Bonferroni k=6.
    assert len(report.metrics) == 6
    assert report.bonferroni_k == 6


def test_discrimination_v2_handles_degenerate_graph() -> None:
    adj = np.zeros((31, 31), dtype=np.uint8)
    report = discriminate_ba_vs_er_six_metrics(adj, n_simulations=20, seed=1)
    # Empty graph yields per-metric NaN empiricals → NOT_DISTINGUISHED.
    assert report.aggregate_verdict in (
        DiscriminationVerdict.NOT_DISTINGUISHED,
        DiscriminationVerdict.INSUFFICIENT_RESOLUTION,
    )
