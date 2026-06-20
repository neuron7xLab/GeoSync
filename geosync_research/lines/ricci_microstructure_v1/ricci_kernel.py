"""Ollivier-Ricci curvature kernel for L2 microstructure graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class RicciKernelConfig:
    alpha: float = 0.5
    n_min_edges: int = 2


def _neighbors(graph: nx.Graph, node: Any) -> list[Any]:
    return sorted(graph.neighbors(node), key=str)


def _measure(graph: nx.Graph, node: Any, alpha: float) -> dict[Any, float]:
    neighbors = _neighbors(graph, node)
    if not neighbors:
        return {node: 1.0}
    mass = {node: float(alpha)}
    share = (1.0 - alpha) / len(neighbors)
    for neighbor in neighbors:
        mass[neighbor] = mass.get(neighbor, 0.0) + share
    return mass


def _wasserstein_distance(graph: nx.Graph, left: dict[Any, float], right: dict[Any, float]) -> float:
    left_nodes = list(left)
    right_nodes = list(right)
    c = np.array(
        [
            nx.shortest_path_length(graph, source=src, target=dst)
            for src in left_nodes
            for dst in right_nodes
        ],
        dtype=float,
    )
    a_eq: list[list[float]] = []
    b_eq: list[float] = []
    width = len(left_nodes) * len(right_nodes)
    for i, src in enumerate(left_nodes):
        row = [0.0] * width
        for j in range(len(right_nodes)):
            row[i * len(right_nodes) + j] = 1.0
        a_eq.append(row)
        b_eq.append(float(left[src]))
    for j, dst in enumerate(right_nodes):
        row = [0.0] * width
        for i in range(len(left_nodes)):
            row[i * len(right_nodes) + j] = 1.0
        a_eq.append(row)
        b_eq.append(float(right[dst]))
    result = linprog(c, A_eq=np.array(a_eq), b_eq=np.array(b_eq), bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(f"Wasserstein solver failed: {result.message}")
    return float(result.fun)


def _compute_local_ollivier(graph: nx.DiGraph, *, alpha: float) -> nx.DiGraph:
    undirected = graph.to_undirected()
    for src, dst in sorted(graph.edges(), key=lambda item: (str(item[0]), str(item[1]))):
        distance = nx.shortest_path_length(undirected, source=src, target=dst)
        if distance <= 0:
            raise RuntimeError(f"invalid graph distance for edge {(src, dst)!r}")
        left = _measure(undirected, src, alpha)
        right = _measure(undirected, dst, alpha)
        w1 = _wasserstein_distance(undirected, left, right)
        graph[src][dst]["ricciCurvature"] = float(1.0 - w1 / float(distance))
    return graph


def compute_ricci_curvature(
    graph: nx.DiGraph, *, config: RicciKernelConfig | None = None
) -> nx.DiGraph:
    cfg = config or RicciKernelConfig()
    if graph.number_of_edges() < cfg.n_min_edges:
        raise RuntimeError("graph_edges < N_min")
    for _, _, data in graph.edges(data=True):
        value = float(data.get("edge_weight", data.get("weight", 1.0)))
        if value != value:
            raise ValueError("NaN edge weight")

    try:
        from GraphRicciCurvature.OllivierRicci import OllivierRicci  # type: ignore
    except Exception:
        result = _compute_local_ollivier(graph.copy(), alpha=cfg.alpha)
    else:
        orc = OllivierRicci(graph.copy(), alpha=cfg.alpha, verbose="ERROR", weight="weight")
        orc.compute_ricci_curvature()
        result = orc.G

    if result.number_of_edges() == 0:
        raise RuntimeError("empty curvature output")
    missing = [edge for edge in result.edges if "ricciCurvature" not in result[edge[0]][edge[1]]]
    if missing:
        raise RuntimeError(f"missing ricciCurvature on {len(missing)} edges")
    return result


def edge_curvature_records(graph: nx.DiGraph) -> list[dict[str, Any]]:
    records = []
    for src, dst, data in sorted(graph.edges(data=True), key=lambda item: (str(item[0]), str(item[1]))):
        curvature = float(data["ricciCurvature"])
        if curvature != curvature:
            raise RuntimeError("NaN ricciCurvature")
        records.append(
            {
                "source": str(src),
                "target": str(dst),
                "edge_weight": float(data.get("edge_weight", data.get("weight", 1.0))),
                "ricciCurvature": curvature,
            }
        )
    if not records:
        raise RuntimeError("empty curvature output")
    return records
