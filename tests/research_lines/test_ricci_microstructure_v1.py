from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest

from geosync_research.lines.ricci_microstructure_v1.artifact import validate_artifact_payload
from geosync_research.lines.ricci_microstructure_v1.graph_builder import build_l2_graph
from geosync_research.lines.ricci_microstructure_v1.ingest import DataFrameSchemaError, validate_l2_frame
from geosync_research.lines.ricci_microstructure_v1.ricci_kernel import (
    RicciKernelConfig,
    compute_ricci_curvature,
    edge_curvature_records,
)
from geosync_research.lines.ricci_microstructure_v1.synthetic_controls import effect_size_class


def frame(depth: int = 5) -> pd.DataFrame:
    rows = []
    for t in range(3):
        row = {"timestamp": f"2026-01-01T00:00:0{t}Z", "price": 100.0 + t}
        for i in range(1, depth + 1):
            row[f"bid_px_{i}"] = 100.0 - i * 0.01
            row[f"ask_px_{i}"] = 100.0 + i * 0.01
            row[f"bid_sz_{i}"] = 10.0 + i + t
            row[f"ask_sz_{i}"] = 9.0 + i
        rows.append(row)
    return pd.DataFrame(rows)


def test_l2_schema_rejects_missing_column() -> None:
    bad = frame().drop(columns=["bid_sz_3"])
    with pytest.raises(DataFrameSchemaError):
        validate_l2_frame(bad, depth=5)


def test_l2_schema_rejects_negative_size() -> None:
    bad = frame()
    bad.loc[0, "ask_sz_2"] = -1
    with pytest.raises(DataFrameSchemaError):
        validate_l2_frame(bad, depth=5)


def test_l2_schema_rejects_non_monotonic_timestamp() -> None:
    bad = frame()
    bad.loc[2, "timestamp"] = "2025-01-01T00:00:00Z"
    with pytest.raises(DataFrameSchemaError):
        validate_l2_frame(bad, depth=5)


def test_graph_builder_and_ricci_emit_curvature() -> None:
    graph = build_l2_graph(frame().iloc[0], depth=5)
    curved = compute_ricci_curvature(graph, config=RicciKernelConfig(alpha=0.5, n_min_edges=2))
    records = edge_curvature_records(curved)
    assert records
    assert all(isinstance(record["ricciCurvature"], float) for record in records)


def test_too_few_edges_fails_closed() -> None:
    graph = nx.DiGraph()
    graph.add_edge("a", "b", edge_weight=1.0, weight=1.0)
    with pytest.raises(RuntimeError, match="graph_edges < N_min"):
        compute_ricci_curvature(graph, config=RicciKernelConfig(n_min_edges=2))


def test_synthetic_cycle_curvature_near_zero_and_complete_positive() -> None:
    cycle = nx.DiGraph()
    for u, v in nx.cycle_graph(6).edges():
        cycle.add_edge(u, v, edge_weight=1.0, weight=1.0)
        cycle.add_edge(v, u, edge_weight=1.0, weight=1.0)
    cycle_curved = compute_ricci_curvature(cycle, config=RicciKernelConfig(alpha=0.5, n_min_edges=2))
    cycle_mean = sum(r["ricciCurvature"] for r in edge_curvature_records(cycle_curved)) / cycle_curved.number_of_edges()
    assert abs(cycle_mean) < 1e-9

    complete = nx.complete_graph(5).to_directed()
    for src, dst in complete.edges:
        complete[src][dst]["edge_weight"] = 1.0
        complete[src][dst]["weight"] = 1.0
    complete_curved = compute_ricci_curvature(complete, config=RicciKernelConfig(alpha=0.5, n_min_edges=2))
    complete_mean = sum(r["ricciCurvature"] for r in edge_curvature_records(complete_curved)) / complete_curved.number_of_edges()
    assert complete_mean > 0.0


def test_effect_size_classes() -> None:
    assert effect_size_class(0.0) == "negligible"
    assert effect_size_class(0.2) == "small"
    assert effect_size_class(0.4) == "medium"
    assert effect_size_class(0.6) == "large"


def test_artifact_schema_requires_fields() -> None:
    errors = validate_artifact_payload({})
    assert errors


def test_artifact_schema_forbids_market_mythology() -> None:
    payload = {
        "run_id": "x",
        "schema_version": "ricci_microstructure_artifact.v1",
        "config_hash": "a" * 64,
        "data_sha256": "b" * 64,
        "git_sha": "c" * 40,
        "mean_curvature": 0.0,
        "p_value": 1.0,
        "cliffs_delta": 0.0,
        "effect_size_class": "negligible",
        "null_model_type": "iaaft",
        "n_surrogates": 1000,
        "falsification_status": "ALPHA_FOUND",
        "replay_command": "geosync-research run --line ricci_microstructure_v1 --config c --data d --out o",
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "claim_tier": "T3/NOVEL/EXPLORATORY/FALSIFIABLE",
    }
    errors = validate_artifact_payload(payload)
    assert errors
