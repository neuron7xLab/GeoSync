# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
r"""Discrete Ricci flow on a weighted edge graph + neckpinch surgery.

Status
------
EXPERIMENTAL. Pure NumPy + ``networkx`` for bridge detection. Opt-in. The
existing :class:`KuramotoRicciFlowEngine` retains byte-identical behavior
unless the explicit constructor flags ``enable_discrete_flow`` and/or
``enable_neckpinch_surgery`` are set.

Mathematical core
-----------------
We follow the discrete Ricci flow on edge weights of an undirected
weighted graph (after Hamilton 1982 / Chow & Luo 2003 / Ni et al. 2019,
arXiv:2510.15942 for the financial-network application):

.. math::

    w_{ij}^{(n+1)} = w_{ij}^{(n)} - \eta \kappa_{ij}^{(n)} \, w_{ij}^{(n)}

After the flow step we optionally perform **neckpinch surgery**: edges
whose weights have collapsed below ``eps_weight`` or whose curvature
falls in the singular tail (``κ ≤ -1 + eps_neck`` for Ollivier-Ricci, or
the most-negative ``max_surgery_fraction`` of edges otherwise) are
removed — unless they are bridges and ``preserve_connectedness=True``,
in which case they are clamped to ``eps_weight`` instead.

Invariants
----------
- ``w_{ij}`` is finite, non-negative, symmetric, zero on the diagonal
  (``INV-RC-FLOW``).
- Surgery never disconnects the graph when ``preserve_connectedness=True``.
- Surgery removes at most ``max_surgery_fraction`` of remaining edges per
  step.
- Optional total-edge-mass preservation: the post-flow weight matrix is
  rescaled so that its sum equals the pre-flow sum (when
  ``preserve_total_edge_mass=True``).

No-alpha disclaimer
-------------------
This module is an experimental geometric primitive. No claim of trading
edge or out-of-sample performance is made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

import networkx as nx
import numpy as np
from numpy.typing import NDArray

__all__ = [
    "RicciFlowConfig",
    "NeckpinchEvent",
    "RicciFlowStepResult",
    "discrete_ricci_flow_step",
    "detect_neckpinch_candidates",
    "apply_neckpinch_surgery",
    "ricci_flow_with_surgery",
]


_FLOAT_TOL: Final[float] = 1e-12

NeckpinchAction = Literal["removed", "clamped", "skipped_bridge"]


@dataclass(frozen=True, slots=True)
class RicciFlowConfig:
    """Hyperparameters for one Ricci-flow + surgery step.

    Attributes
    ----------
    eta:
        Flow step size in ``(0, 1)``. Default 0.05.
    eps_weight:
        Lower clamp / removal threshold for edge weights. Default 1e-8.
    eps_neck:
        Curvature tolerance defining the Ollivier-Ricci neckpinch tail
        ``κ ≤ -1 + eps_neck``. Default 1e-3.
    preserve_total_edge_mass:
        When True, rescale the post-flow matrix to keep ``Σ w_{ij}``
        equal to the pre-flow value.
    preserve_connectedness:
        When True, bridge edges are clamped (not removed) so that the
        graph stays connected.
    allow_disconnect:
        Reserved escape hatch — must be False unless the caller
        explicitly opts into disconnection.
    max_surgery_fraction:
        Upper bound on the fraction of currently active edges removed
        per single surgery call. Default 0.05 (5 %).
    """

    eta: float = 0.05
    eps_weight: float = 1e-8
    eps_neck: float = 1e-3
    preserve_total_edge_mass: bool = True
    preserve_connectedness: bool = True
    allow_disconnect: bool = False
    max_surgery_fraction: float = 0.05

    def __post_init__(self) -> None:
        if not (0.0 < self.eta < 1.0):
            raise ValueError("eta must be in (0, 1).")
        if self.eps_weight <= 0.0:
            raise ValueError("eps_weight must be > 0.")
        if not (0.0 < self.eps_neck < 1.0):
            raise ValueError("eps_neck must be in (0, 1).")
        if not (0.0 <= self.max_surgery_fraction <= 1.0):
            raise ValueError("max_surgery_fraction must be in [0, 1].")
        if self.allow_disconnect and self.preserve_connectedness:
            raise ValueError(
                "allow_disconnect=True is incompatible with preserve_connectedness=True."
            )


@dataclass(frozen=True, slots=True)
class NeckpinchEvent:
    """One surgery decision for a single edge."""

    edge: tuple[int, int]
    old_weight: float
    new_weight: float
    curvature: float
    action: NeckpinchAction


@dataclass(frozen=True, slots=True)
class RicciFlowStepResult:
    """Result of a single Ricci flow + surgery step."""

    weights_before: NDArray[np.float64]
    weights_after: NDArray[np.float64]
    curvature: dict[tuple[int, int], float]
    surgery_events: tuple[NeckpinchEvent, ...]
    total_edge_mass_before: float
    total_edge_mass_after: float
    surgery_event_count: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "surgery_event_count", len(self.surgery_events))

    @property
    def neckpinch_edges(self) -> tuple[tuple[int, int], ...]:
        """Edges that participated in any surgery event."""
        return tuple(e.edge for e in self.surgery_events)


# ── Validation helpers ─────────────────────────────────────────────────────


def _validate_weights(weights: NDArray[np.float64]) -> None:
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("weights must be a square matrix.")
    if not np.isfinite(weights).all():
        raise ValueError("weights contain non-finite values.")
    if (weights < -_FLOAT_TOL).any():
        raise ValueError("weights must be non-negative.")
    if not np.allclose(weights, weights.T, atol=1e-10):
        raise ValueError("weights must be symmetric.")
    if not np.allclose(np.diag(weights), 0.0, atol=1e-12):
        raise ValueError("weights must have zero diagonal.")


def _curvature_to_dense(curvature: dict[tuple[int, int], float], n: int) -> NDArray[np.float64]:
    """Symmetrise a curvature dict into a dense ``(n, n)`` matrix.

    Missing edges receive curvature 0 (curvature only matters where weights
    are non-zero).
    """
    K = np.zeros((n, n), dtype=np.float64)
    for (i, j), kappa in curvature.items():
        if not np.isfinite(kappa):
            raise ValueError(f"non-finite curvature at edge ({i},{j}).")
        if i == j:
            continue
        K[i, j] = float(kappa)
        K[j, i] = float(kappa)
    return K


# ── Flow ───────────────────────────────────────────────────────────────────


def discrete_ricci_flow_step(
    weights: NDArray[np.float64],
    curvature: dict[tuple[int, int], float],
    cfg: RicciFlowConfig,
) -> NDArray[np.float64]:
    """Single explicit-Euler discrete Ricci flow step.

    .. math::

        w_{ij}^{n+1} = \\max\bigl(0,\\ w_{ij}^{n} - \\eta\\,\\kappa_{ij}\\,w_{ij}^{n}\bigr)

    Optionally rescales to preserve ``Σ w_{ij}`` when
    ``cfg.preserve_total_edge_mass`` is True. Diagonal is zeroed and the
    matrix is re-symmetrised.
    """
    _validate_weights(weights)
    n = weights.shape[0]
    K = _curvature_to_dense(curvature, n)

    new_w = weights - cfg.eta * K * weights
    new_w = np.maximum(new_w, 0.0)  # INV-RC-FLOW: weights non-negative.
    new_w = 0.5 * (new_w + new_w.T)  # exact symmetrisation
    np.fill_diagonal(new_w, 0.0)

    if cfg.preserve_total_edge_mass:
        total_before = float(weights.sum())
        total_after = float(new_w.sum())
        if total_after > _FLOAT_TOL and total_before > _FLOAT_TOL:
            new_w = new_w * (total_before / total_after)
            np.fill_diagonal(new_w, 0.0)

    if not np.isfinite(new_w).all():
        raise ValueError("Ricci flow step produced non-finite weights.")
    return new_w.astype(np.float64, copy=False)


# ── Neckpinch detection ────────────────────────────────────────────────────


def detect_neckpinch_candidates(
    weights: NDArray[np.float64],
    curvature: dict[tuple[int, int], float],
    cfg: RicciFlowConfig,
) -> list[tuple[int, int]]:
    """Return candidate neckpinch edges in deterministic lexicographic order.

    A directed edge ``(i, j)`` with ``i < j`` is a candidate if either:

    1. its weight is at or below ``cfg.eps_weight``, **and** the edge
       currently exists (weight > 0), or
    2. its Ollivier-Ricci curvature is in the singular tail
       ``κ ≤ -1 + eps_neck``.

    The candidate list is sorted lexicographically by ``(i, j)``.
    """
    _validate_weights(weights)
    n = weights.shape[0]
    candidates: set[tuple[int, int]] = set()

    iu, ju = np.triu_indices(n, k=1)
    weight_pairs = zip(iu.tolist(), ju.tolist(), weights[iu, ju].tolist(), strict=True)
    for i, j, w in weight_pairs:
        if 0.0 < w <= cfg.eps_weight:
            candidates.add((i, j))

    for (i, j), kappa in curvature.items():
        if i >= j:
            continue
        if not np.isfinite(kappa):
            continue
        if weights[i, j] <= 0.0:
            continue
        if kappa <= -1.0 + cfg.eps_neck:
            candidates.add((int(i), int(j)))

    return sorted(candidates)


# ── Surgery ────────────────────────────────────────────────────────────────


def _is_bridge(graph: nx.Graph, edge: tuple[int, int]) -> bool:
    if not graph.has_edge(*edge):
        return False
    return edge in set(nx.bridges(graph)) or (edge[1], edge[0]) in set(nx.bridges(graph))


def apply_neckpinch_surgery(
    weights: NDArray[np.float64],
    curvature: dict[tuple[int, int], float],
    cfg: RicciFlowConfig,
) -> tuple[NDArray[np.float64], tuple[NeckpinchEvent, ...]]:
    """Apply neckpinch surgery; return (new_weights, ordered events).

    The number of edges removed is capped at
    ``floor(max_surgery_fraction * n_active_edges)``; remaining
    candidates are clamped to ``cfg.eps_weight`` (``"clamped"``) or
    skipped (``"skipped_bridge"``).
    """
    _validate_weights(weights)
    new_w = weights.astype(np.float64, copy=True)
    n = new_w.shape[0]

    iu, ju = np.triu_indices(n, k=1)
    active_edges = [
        (int(i), int(j))
        for i, j, w in zip(iu.tolist(), ju.tolist(), new_w[iu, ju].tolist(), strict=True)
        if w > 0.0
    ]
    n_active = len(active_edges)
    max_remove = int(np.floor(cfg.max_surgery_fraction * n_active))

    candidates = detect_neckpinch_candidates(new_w, curvature, cfg)

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i, j in active_edges:
        graph.add_edge(i, j, weight=float(new_w[i, j]))

    events: list[NeckpinchEvent] = []
    removed_count = 0

    for edge in candidates:
        i, j = edge
        old = float(new_w[i, j])
        kappa = float(curvature.get(edge, curvature.get((j, i), 0.0)))

        is_bridge = cfg.preserve_connectedness and _is_bridge(graph, edge)

        if is_bridge:
            new_val = max(cfg.eps_weight, _FLOAT_TOL)
            new_w[i, j] = new_val
            new_w[j, i] = new_val
            graph[i][j]["weight"] = new_val
            events.append(
                NeckpinchEvent(
                    edge=edge,
                    old_weight=old,
                    new_weight=new_val,
                    curvature=kappa,
                    action="skipped_bridge",
                )
            )
            continue

        if removed_count < max_remove:
            new_w[i, j] = 0.0
            new_w[j, i] = 0.0
            if graph.has_edge(i, j):
                graph.remove_edge(i, j)
            events.append(
                NeckpinchEvent(
                    edge=edge,
                    old_weight=old,
                    new_weight=0.0,
                    curvature=kappa,
                    action="removed",
                )
            )
            removed_count += 1
        else:
            new_w[i, j] = cfg.eps_weight
            new_w[j, i] = cfg.eps_weight
            if graph.has_edge(i, j):
                graph[i][j]["weight"] = cfg.eps_weight
            events.append(
                NeckpinchEvent(
                    edge=edge,
                    old_weight=old,
                    new_weight=cfg.eps_weight,
                    curvature=kappa,
                    action="clamped",
                )
            )

    np.fill_diagonal(new_w, 0.0)
    new_w = 0.5 * (new_w + new_w.T)

    if cfg.preserve_connectedness and not cfg.allow_disconnect and n_active > 0:
        check_graph = nx.Graph()
        check_graph.add_nodes_from(range(n))
        for i, j in zip(iu.tolist(), ju.tolist(), strict=True):
            if new_w[i, j] > 0.0:
                check_graph.add_edge(int(i), int(j))
        # Connectedness over nodes that had at least one active edge before.
        active_nodes = {a for edge in active_edges for a in edge}
        if active_nodes:
            sub = check_graph.subgraph(active_nodes)
            if not nx.is_connected(sub):
                raise RuntimeError(
                    "Surgery disconnected the active subgraph despite preserve_connectedness=True."
                )

    return new_w.astype(np.float64, copy=False), tuple(events)


# ── Combined step ──────────────────────────────────────────────────────────


def ricci_flow_with_surgery(
    weights: NDArray[np.float64],
    curvature: dict[tuple[int, int], float],
    cfg: RicciFlowConfig,
) -> RicciFlowStepResult:
    """Run one flow step then surgery; return a typed result bundle."""
    _validate_weights(weights)
    weights_before = weights.astype(np.float64, copy=True)

    flowed = discrete_ricci_flow_step(weights_before, curvature, cfg)
    after, events = apply_neckpinch_surgery(flowed, curvature, cfg)

    if not np.isfinite(after).all():
        raise ValueError("Ricci flow + surgery produced non-finite weights.")
    if (after < -_FLOAT_TOL).any():
        raise ValueError("Ricci flow + surgery produced negative weights.")
    if not np.allclose(after, after.T, atol=1e-10):
        raise ValueError("Ricci flow + surgery lost symmetry.")
    if not np.allclose(np.diag(after), 0.0, atol=1e-12):
        raise ValueError("Ricci flow + surgery introduced self-loops.")

    return RicciFlowStepResult(
        weights_before=weights_before,
        weights_after=after,
        curvature=dict(curvature),
        surgery_events=events,
        total_edge_mass_before=float(weights_before.sum()),
        total_edge_mass_after=float(after.sum()),
    )
