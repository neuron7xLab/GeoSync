"""Build deterministic L2 order-book graphs for Ricci inference."""

from __future__ import annotations

from typing import Any

import networkx as nx
import pandas as pd

from .nobi import nobi


def build_l2_graph(row: pd.Series | dict[str, Any], *, depth: int, weight_mode: str = "nobi") -> nx.DiGraph:
    if depth < 1:
        raise ValueError("depth must be >= 1")
    record = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    imbalance = nobi(record, depth)
    if weight_mode != "nobi":
        raise ValueError(f"unsupported edge weight mode: {weight_mode}")

    graph = nx.DiGraph()
    for idx in range(1, depth + 1):
        bid_node = f"bid_{idx}"
        ask_node = f"ask_{idx}"
        graph.add_node(
            bid_node,
            side="bid",
            level=idx,
            price=float(record[f"bid_px_{idx}"]),
            size=float(record[f"bid_sz_{idx}"]),
        )
        graph.add_node(
            ask_node,
            side="ask",
            level=idx,
            price=float(record[f"ask_px_{idx}"]),
            size=float(record[f"ask_sz_{idx}"]),
        )

    distance_weight = abs(imbalance) + 1.0e-12
    for idx in range(1, depth):
        for src, dst in ((f"bid_{idx}", f"bid_{idx + 1}"), (f"ask_{idx}", f"ask_{idx + 1}")):
            graph.add_edge(src, dst, edge_weight=imbalance, weight=distance_weight)
            graph.add_edge(dst, src, edge_weight=imbalance, weight=distance_weight)
    for idx in range(1, depth + 1):
        graph.add_edge(f"bid_{idx}", f"ask_{idx}", edge_weight=imbalance, weight=distance_weight)
        graph.add_edge(f"ask_{idx}", f"bid_{idx}", edge_weight=imbalance, weight=distance_weight)

    for _, _, data in graph.edges(data=True):
        if data["edge_weight"] != data["edge_weight"]:
            raise ValueError("NaN edge weight")
    return graph
