# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Mathematical witnesses for the Ricci-geometry laws in the physics catalog.

These are *witnesses* in the repository's sense: each binds a pytest function to
a first-principles physical law via ``@law(...)`` and proves the law holds in the
real code, deriving every tolerance from the law (no magic literals). They raise
``physics_contracts.law`` coverage by binding two previously-unwitnessed
catalog laws:

* ``ricci.ollivier_bounds``   — κ ∈ [−1, 1] (INV-RC1 universal upper bound;
  INV-RC3 full bound on price graphs).
* ``ricci.flow_monotonicity`` — the Perelman-like F-functional Σ κ²w is
  non-increasing under a discrete Ricci flow step (INV-RC-FLOW family).

Nothing here is a market claim. These are geometry facts about the code.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from core.indicators.ricci import build_price_graph, ricci_curvature_edge
from core.kuramoto.ricci_flow import RicciFlowConfig, discrete_ricci_flow_step
from physics_contracts.law import law

# Strict definitional bounds carry only a numerical-noise tolerance.
CURVATURE_EPS = 1e-9
# Catalog ``ricci.flow_monotonicity`` tolerance: F(t+1) ≤ F(t)·(1 + 1e-9).
FLOW_MONO_TOLERANCE = 1e-9


def _random_connected_weighted_graph(seed: int, n: int) -> nx.Graph:
    """A connected small-world graph with strictly positive edge weights."""
    rng = np.random.default_rng(seed)
    g: nx.Graph = nx.connected_watts_strogatz_graph(n, 3, 0.4, seed=seed)
    for u, v in g.edges():
        g[u][v]["weight"] = float(rng.uniform(0.1, 2.0))
    return g


def _weight_matrix(g: nx.Graph) -> np.ndarray:
    n = g.number_of_nodes()
    w = np.zeros((n, n), dtype=np.float64)
    for u, v, data in g.edges(data=True):
        weight = float(data["weight"])
        w[u, v] = weight
        w[v, u] = weight
    return w


def _curvature_dict(g: nx.Graph) -> dict[tuple[int, int], float]:
    return {(int(u), int(v)): ricci_curvature_edge(g, int(u), int(v)) for u, v in g.edges()}


def _f_functional(curvature: dict[tuple[int, int], float], w: np.ndarray) -> float:
    """Perelman-like functional F = Σ_e κ_e² · w_e."""
    return float(sum(k * k * w[i, j] for (i, j), k in curvature.items()))


@law("ricci.ollivier_bounds", n_random_graphs=60, n_price_graphs=40)
def test_ollivier_ricci_curvature_is_bounded() -> None:
    """INV-RC1 / INV-RC3: Ollivier–Ricci curvature obeys its definitional bound.

    κ(x,y) = 1 − W₁(mₓ,m_y)/d(x,y). The upper bound κ ≤ 1 holds for every edge of
    every connected weighted graph (INV-RC1). The full bound κ ∈ [−1, 1] holds on
    graphs produced by ``build_price_graph`` (INV-RC3); the lower bound is not
    universal because the implementation's 1-D positional embedding can drive κ
    below −1 on non-price topologies, so it is asserted only where it is owed.
    """
    n_random_graphs = 60
    worst_upper = -np.inf
    for seed in range(n_random_graphs):
        g = _random_connected_weighted_graph(seed, n=int(5 + (seed % 7)))
        for u, v in g.edges():
            kappa = ricci_curvature_edge(g, int(u), int(v))
            worst_upper = max(worst_upper, kappa)
    assert worst_upper <= 1.0 + CURVATURE_EPS, (
        "INV-RC1 violated: observed max κ = "
        f"{worst_upper:.6f} must be ≤ 1 (definitional upper bound, tolerance "
        f"{CURVATURE_EPS:.0e}) over n_random_graphs={n_random_graphs} connected "
        "graphs with degree=3 small-world topology (seed=0..N); "
        "κ = 1 − W₁/d cannot exceed 1"
    )

    n_price_graphs = 40
    worst_low = np.inf
    worst_high = -np.inf
    rng = np.random.default_rng(seed=99)
    for trial in range(n_price_graphs):
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 1e-3, size=64)))
        gp = build_price_graph(prices, delta=0.005)
        for u, v in gp.edges():
            kappa = ricci_curvature_edge(gp, int(u), int(v))
            worst_low = min(worst_low, kappa)
            worst_high = max(worst_high, kappa)
    assert worst_low >= -1.0 - CURVATURE_EPS, (
        "INV-RC3 violated: observed min κ = "
        f"{worst_low:.6f} must be ≥ −1 on price graphs (tolerance "
        f"{CURVATURE_EPS:.0e}) over n_price_graphs={n_price_graphs} build_price_graph "
        "outputs with delta=0.005 and seed=99"
    )
    assert worst_high <= 1.0 + CURVATURE_EPS, (
        "INV-RC3 violated: observed max κ = "
        f"{worst_high:.6f} must be ≤ 1 on price graphs (tolerance "
        f"{CURVATURE_EPS:.0e}) over n_price_graphs={n_price_graphs} build_price_graph "
        "outputs with delta=0.005 and seed=99"
    )


@law("ricci.flow_monotonicity", n_random_graphs=45, etas=(0.02, 0.05, 0.1))
def test_ricci_flow_F_functional_is_non_increasing() -> None:
    """INV-RC-FLOW: the Perelman-like F-functional Σκ²w never grows in a step.

    A single explicit-Euler discrete Ricci flow step w ← w − η·κ·w is dissipative
    on F = Σ_e κ_e²·w_e. The witness recomputes the curvature after the step and
    requires F(t+1) ≤ F(t)·(1 + tolerance) with the catalog's numerical slack,
    over a deterministic suite of connected weighted graphs and step sizes.
    """
    n_random_graphs = 45
    etas = (0.02, 0.05, 0.1)
    worst_ratio = 0.0
    checked = 0
    for eta in etas:
        for seed in range(n_random_graphs):
            g = _random_connected_weighted_graph(seed + 1000, n=int(5 + (seed % 5)))
            w0 = _weight_matrix(g)
            curvature_before = _curvature_dict(g)
            f_before = _f_functional(curvature_before, w0)
            if f_before <= CURVATURE_EPS:
                continue
            w1 = discrete_ricci_flow_step(w0, curvature_before, RicciFlowConfig(eta=float(eta)))
            g1 = nx.from_numpy_array(w1)
            g1.remove_edges_from(nx.selfloop_edges(g1))
            if g1.number_of_edges() == 0:
                continue
            curvature_after = _curvature_dict(g1)
            f_after = _f_functional(curvature_after, w1)
            worst_ratio = max(worst_ratio, f_after / f_before)
            checked += 1
    assert checked > 0, (
        "INV-RC-FLOW witness degenerate: observed checked=0 cases must be > 0 "
        f"with etas={etas} and seed=1000..N; the flow fixture produced no "
        "valid F(t) — expected at least one connected weighted graph"
    )
    assert worst_ratio <= 1.0 + FLOW_MONO_TOLERANCE, (
        "ricci.flow_monotonicity / INV-RC-FLOW violated: observed worst "
        f"F(t+1)/F(t) = {worst_ratio:.6f} must be ≤ 1 (tolerance "
        f"{FLOW_MONO_TOLERANCE:.0e}) over checked={checked} steps "
        f"with etas={etas}; the F-functional must be non-increasing under the flow"
    )
