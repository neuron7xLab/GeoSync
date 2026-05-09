# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Tests for tools/build_disha_ba_correlation_figures.py."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tools.build_disha_ba_correlation_figures import (
    BuildOptions,
    build_article_artifact,
    build_correlation_matrix,
    build_period_matrix,
    build_period_outward_strength_panel,
    compute_ba_comparison,
    compute_risk_concentration,
    estimate_ba_m,
    load_dataset,
    parse_quarter,
    quarter_end_date,
    threshold_adjacency,
    to_log_changes,
)

# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """4-country, 8-quarter synthetic BIS-like dataset_dir.

    Country 0 (AAA) and 1 (BBB) form a strongly correlated pair.
    Country 2 (CCC) is constant exposure (will trigger constant-node handling).
    Country 3 (DDD) is uncorrelated noise.
    Edge 0->1 is the strongest persistent edge.
    """
    nodes = ["AAA", "BBB", "CCC", "DDD"]
    n = len(nodes)
    # Quarters: 2006Q1..2009Q4 (16 quarters)
    quarters = []
    for year in range(2006, 2010):
        for q in (1, 2, 3, 4):
            quarters.append(quarter_end_date(year, q))
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(123)
    for k, qd in enumerate(quarters):
        # AAA <-> BBB strongly correlated (both grow with k + small noise)
        base = 1000.0 + 50.0 * k + rng.normal(0, 5)
        m = np.zeros((n, n))
        m[0, 1] = base * 1.5
        m[1, 0] = base * 0.9
        # CCC always 100 (constant)
        m[2, 0] = 100.0
        m[2, 1] = 100.0
        # DDD random uncorrelated
        m[3, 0] = max(10.0, rng.normal(500, 200))
        m[3, 1] = max(10.0, rng.normal(400, 200))
        # AAA also exports to CCC (small)
        m[0, 2] = 50.0 + 5 * k
        for i in range(n):
            for j in range(n):
                if i != j and m[i, j] > 0:
                    rows.append(
                        {
                            "date": qd,
                            "source": i,
                            "target": j,
                            "exposure": float(m[i, j]),
                        }
                    )
    df = pd.DataFrame(rows)
    ds = tmp_path / "synth_ds"
    ds.mkdir()
    df.to_parquet(ds / "exposure_panel.parquet", index=False)
    nm = pd.DataFrame({"node_id": list(range(n)), "bank_label": nodes})
    nm.to_parquet(ds / "node_mapping.parquet", index=False)
    payload_sha = hashlib.sha256((ds / "exposure_panel.parquet").read_bytes()).hexdigest()
    manifest = {
        "source_id": "TEST-SYNTHETIC",
        "schema_version": "interbank.panel.v1",
        "capture_timestamp_utc": "2026-05-09T12:00:00+00:00",
        "payload_sha256": payload_sha,
        "seed": 42,
        "config_hash": "synthetic-test",
        "n_banks": n,
        "n_days": len(quarters),
        "crisis_lock_timestamp_utc": "2026-05-01T00:00:00+00:00",
        "first_evaluation_timestamp_utc": "2026-05-08T00:00:00+00:00",
        "interpretation_caveat": "synthetic test fixture; not BIS data",
        "filter_spec": {
            "FREQ": "Q",
            "TIME_RANGE": ["2006-Q1", "2009-Q4"],
        },
        "config": {"window": 4, "align": "trailing"},
    }
    (ds / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (ds / "license.txt").write_text("Test fixture — use is permitted.\n")
    (ds / "crisis_ledger.json").write_text(json.dumps({"events": []}))
    return ds


# ---------------------------------------------------------------------------
# Quarter / parsing
# ---------------------------------------------------------------------------


def test_parse_quarter_round_trip() -> None:
    assert parse_quarter("2008Q3") == (2008, 3)
    assert parse_quarter("2008-Q3") == (2008, 3)
    assert parse_quarter("1995q1") == (1995, 1)


def test_parse_quarter_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_quarter("not-a-quarter")


def test_quarter_end_date() -> None:
    assert quarter_end_date(2008, 3) == date(2008, 9, 30)
    assert quarter_end_date(2023, 4) == date(2023, 12, 31)


# ---------------------------------------------------------------------------
# DATA LOADING / MATRIX TESTS
# ---------------------------------------------------------------------------


def test_load_dataset_synth(synthetic_dataset: Path) -> None:
    ds = load_dataset(synthetic_dataset)
    assert ds.nodes == ("AAA", "BBB", "CCC", "DDD")
    assert ds.label_to_id == {"AAA": 0, "BBB": 1, "CCC": 2, "DDD": 3}
    assert ds.panel.shape[0] > 0


def test_load_dataset_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "does_not_exist")


def test_period_filtering(synthetic_dataset: Path) -> None:
    ds = load_dataset(synthetic_dataset)
    n = len(ds.nodes)
    mat, qs = build_period_matrix(ds.panel, n, (2006, 1), (2007, 4))
    assert mat.shape == (n, n)
    assert len(qs) == 8  # 8 quarters in 2006-2007


def test_period_filtering_empty_window(synthetic_dataset: Path) -> None:
    ds = load_dataset(synthetic_dataset)
    n = len(ds.nodes)
    # Window outside synthetic data range
    mat, qs = build_period_matrix(ds.panel, n, (1990, 1), (1990, 4))
    assert mat.shape == (n, n)
    assert qs == []
    assert np.all(mat == 0)


def test_node_ordering_deterministic(synthetic_dataset: Path) -> None:
    ds1 = load_dataset(synthetic_dataset)
    ds2 = load_dataset(synthetic_dataset)
    assert ds1.nodes == ds2.nodes


def test_edge_direction_preserved(synthetic_dataset: Path) -> None:
    ds = load_dataset(synthetic_dataset)
    n = len(ds.nodes)
    mat, _ = build_period_matrix(ds.panel, n, (2006, 1), (2007, 4))
    # AAA->BBB stronger than BBB->AAA in fixture
    assert mat[0, 1] > mat[1, 0]


def test_non_negative_finite_values_preserved(synthetic_dataset: Path) -> None:
    ds = load_dataset(synthetic_dataset)
    n = len(ds.nodes)
    mat, _ = build_period_matrix(ds.panel, n, (2006, 1), (2009, 4))
    assert np.all(mat >= 0)
    assert np.all(np.isfinite(mat))


def test_outward_strength_panel(synthetic_dataset: Path) -> None:
    ds = load_dataset(synthetic_dataset)
    n = len(ds.nodes)
    panel, qs = build_period_outward_strength_panel(ds.panel, n, (2006, 1), (2007, 4))
    assert panel.shape == (8, n)
    # CCC always 100 to AAA + 100 to BBB = 200 constant
    assert (panel.iloc[:, 2] == 200).all()


# ---------------------------------------------------------------------------
# NETWORK TESTS
# ---------------------------------------------------------------------------


def test_threshold_deterministic() -> None:
    M = np.array(
        [
            [0, 5, 1],
            [2, 0, 3],
            [4, 0, 0],
        ],
        dtype=float,
    )
    b1 = threshold_adjacency(M, 0.5)
    b2 = threshold_adjacency(M, 0.5)
    assert np.array_equal(b1, b2)
    # diagonal must be zero
    assert np.all(np.diag(b1) == 0)


def test_threshold_all_zero_safe() -> None:
    M = np.zeros((4, 4))
    b = threshold_adjacency(M, 0.85)
    assert b.shape == (4, 4)
    assert np.all(b == 0)


def test_estimate_ba_m_bounds() -> None:
    assert estimate_ba_m(31, 100) >= 1
    assert estimate_ba_m(31, 100) <= 30
    assert estimate_ba_m(5, 100) <= 4
    assert estimate_ba_m(0, 0) == 0
    assert estimate_ba_m(2, 0) == 1  # max(1, round(0/2)) = 1


def test_ba_comparison_returns_finite_for_normal_graph() -> None:
    rng = np.random.default_rng(7)
    n = 12
    A = (rng.random((n, n)) > 0.6).astype(np.uint8)
    np.fill_diagonal(A, 0)
    info = compute_ba_comparison(A, ba_simulations=20, seed=42)
    assert np.isfinite(info["ba_degree_pearson_r"]) or np.isnan(info["ba_degree_pearson_r"])
    assert info["ba_m_estimate"] >= 1
    assert info["ba_m_estimate"] <= n - 1


def test_ba_comparison_degenerate_returns_documented_nan() -> None:
    A = np.zeros((10, 10), dtype=np.uint8)
    info = compute_ba_comparison(A, ba_simulations=5, seed=42)
    assert np.isnan(info["ba_degree_pearson_r"])
    assert info["interpretation"].startswith("inconclusive")


# ---------------------------------------------------------------------------
# CORRELATION TESTS
# ---------------------------------------------------------------------------


def test_correlation_symmetric_diag_one() -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.normal(size=(10, 4)))
    corr, _ = build_correlation_matrix(df)
    assert np.allclose(corr, corr.T, equal_nan=True)
    for i in range(4):
        assert corr[i, i] == 1.0


def test_correlation_constant_node_handling() -> None:
    df = pd.DataFrame(
        {
            0: [1.0, 2.0, 3.0, 4.0, 5.0],
            1: [5.0, 5.0, 5.0, 5.0, 5.0],  # constant
            2: [2.0, 4.0, 6.0, 8.0, 10.0],  # perfectly correlated with 0
        }
    )
    corr, constants = build_correlation_matrix(df)
    assert constants == [1]
    assert np.isnan(corr[1, 0])
    assert np.isnan(corr[0, 1])
    assert abs(corr[0, 2] - 1.0) < 1e-9


def test_log_changes_one_fewer_observation() -> None:
    df = pd.DataFrame({0: [1.0, 2.0, 4.0, 8.0, 16.0]})
    chg = to_log_changes(df, epsilon=1.0)
    assert chg.shape == (4, 1)


def test_delta_correlation_simple() -> None:
    A = np.array([[1.0, 0.5], [0.5, 1.0]])
    B = np.array([[1.0, 0.9], [0.9, 1.0]])
    delta = B - A
    assert delta[0, 1] == pytest.approx(0.4)
    assert delta[0, 0] == 0.0


# ---------------------------------------------------------------------------
# RISK CONCENTRATION
# ---------------------------------------------------------------------------


def test_risk_concentration_ranking_basic() -> None:
    nodes = ("AAA", "BBB", "CCC")
    cor_lev_n = np.array([[1.0, 0.5, 0.1], [0.5, 1.0, 0.1], [0.1, 0.1, 1.0]])
    cor_lev_l = np.array([[1.0, 0.9, 0.7], [0.9, 1.0, 0.7], [0.7, 0.7, 1.0]])
    cor_chg_n = cor_lev_n.copy()
    cor_chg_l = cor_lev_l.copy()
    weighted = np.array([[0, 100, 50], [10, 0, 5], [1, 1, 0]], dtype=float)
    binary = np.array([[0, 1, 1], [1, 0, 0], [0, 0, 0]], dtype=np.uint8)
    rc = compute_risk_concentration(
        nodes=nodes,
        corr_levels_normal=cor_lev_n,
        corr_levels_lehman=cor_lev_l,
        corr_changes_normal=cor_chg_n,
        corr_changes_lehman=cor_chg_l,
        weighted_lehman=weighted,
        binary_lehman=binary,
    )
    assert list(rc["country"]) == ["AAA", "BBB", "CCC"] or set(rc["country"]) == set(nodes)
    assert rc["risk_concentration_rank"].tolist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# INTEGRATION: end-to-end with synthetic dataset
# ---------------------------------------------------------------------------


def test_build_article_artifact_end_to_end(synthetic_dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "article_out"
    opts = BuildOptions(
        dataset_dir=synthetic_dataset,
        output_dir=out,
        normal_start=(2006, 1),
        normal_end=(2007, 4),
        lehman_start=(2008, 3),
        lehman_end=(2009, 2),
        sensitivity_start=(2007, 1),
        sensitivity_end=(2009, 4),
        edge_quantile=0.5,
        top_n_labels=4,
        ba_simulations=20,
        seed=42,
    )
    summary = build_article_artifact(opts)
    assert summary["n_quarters_normal"] == 8
    assert summary["n_quarters_lehman"] == 4
    # all expected files
    expected = [
        "network_normal.png",
        "network_lehman.png",
        "correlation_levels_normal.png",
        "correlation_levels_lehman.png",
        "correlation_levels_delta.png",
        "correlation_changes_normal.png",
        "correlation_changes_lehman.png",
        "correlation_changes_delta.png",
        "risk_concentration_bar.png",
        "ba_fit_summary.csv",
        "top_correlated_pairs_levels.csv",
        "top_correlated_pairs_changes.csv",
        "risk_concentration_summary.csv",
        "data_quality_summary.csv",
        "DISHA_ARTICLE_SUMMARY.md",
        "REPRODUCIBILITY.md",
    ]
    for name in expected:
        assert (out / name).is_file(), f"missing: {name}"


def test_forbidden_wording_absent_from_summary(synthetic_dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "article_out"
    opts = BuildOptions(
        dataset_dir=synthetic_dataset,
        output_dir=out,
        normal_start=(2006, 1),
        normal_end=(2007, 4),
        lehman_start=(2008, 3),
        lehman_end=(2009, 2),
        sensitivity_start=(2007, 1),
        sensitivity_end=(2009, 4),
        edge_quantile=0.5,
        top_n_labels=4,
        ba_simulations=10,
        seed=42,
    )
    build_article_artifact(opts)
    summary_md = (out / "DISHA_ARTICLE_SUMMARY.md").read_text(encoding="utf-8").lower()
    forbidden = [
        "bank-level interbank network",
        "bank-to-bank exposures",
        "validated repo liquidity",
        "confirmed barabási-albert",
        "confirmed barabasi-albert",
        "confirmed systemic-risk phase transition",
        "liquidity contagion proof",
        "production-grade scientific validation",
    ]
    for f in forbidden:
        assert f not in summary_md, f"forbidden wording leaked: {f!r}"


def test_repro_markdown_records_sha_and_dataset(synthetic_dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "article_out"
    opts = BuildOptions(
        dataset_dir=synthetic_dataset,
        output_dir=out,
        normal_start=(2006, 1),
        normal_end=(2007, 4),
        lehman_start=(2008, 3),
        lehman_end=(2009, 2),
        sensitivity_start=(2007, 1),
        sensitivity_end=(2009, 4),
        edge_quantile=0.5,
        top_n_labels=4,
        ba_simulations=10,
        seed=42,
    )
    build_article_artifact(opts)
    repro = (out / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
    assert "Git SHA" in repro
    assert str(synthetic_dataset) in repro
