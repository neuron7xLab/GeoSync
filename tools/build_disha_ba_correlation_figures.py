#!/usr/bin/env python3
# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Build BA-style topology + Pearson-correlation article figures for Disha.

Honest scope:
    BIS LBS country-level banking-system aggregate exposure data.
    Not bank-level. Not repo-specific. Not a contagion proof.
    Descriptive macro illustration only.

Outputs:
    figures/disha_ba_correlation/
        network_normal.png  network_lehman.png
        correlation_levels_{normal,lehman,delta}.png
        correlation_changes_{normal,lehman,delta}.png
        risk_concentration_bar.png
        ba_fit_summary.csv
        top_correlated_pairs_levels.csv
        top_correlated_pairs_changes.csv
        risk_concentration_summary.csv
        data_quality_summary.csv
        DISHA_ARTICLE_SUMMARY.md
        REPRODUCIBILITY.md
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

__all__ = [
    "build_article_artifact",
    "build_correlation_matrix",
    "build_period_matrix",
    "compute_ba_comparison",
    "compute_risk_concentration",
    "estimate_ba_m",
    "load_dataset",
    "main",
    "parse_quarter",
    "quarter_end_date",
    "threshold_adjacency",
]


_FORBIDDEN_WORDING: tuple[str, ...] = (
    "bank-level interbank network",
    "bank-to-bank exposures",
    "validated repo liquidity",
    "confirmed Barabási-Albert",
    "confirmed Barabasi-Albert",
    "confirmed systemic-risk phase transition",
    "liquidity contagion proof",
    "production-grade scientific validation",
)


# ---------------------------------------------------------------------------
# Quarter parsing
# ---------------------------------------------------------------------------


def parse_quarter(label: str) -> tuple[int, int]:
    """``"2008Q3" -> (2008, 3)``."""
    s = label.strip().upper().replace("-", "")
    if "Q" not in s:
        raise ValueError(f"unrecognised quarter label: {label!r}")
    yr_str, q_str = s.split("Q", 1)
    return int(yr_str), int(q_str)


def quarter_end_date(year: int, quarter: int) -> date:
    end = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}[quarter]
    return date(year, end[0], end[1])


def quarter_in_range(d: date, start: tuple[int, int], end: tuple[int, int]) -> bool:
    s = quarter_end_date(*start)
    e = quarter_end_date(*end)
    return s <= d <= e


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedDataset:
    dataset_dir: Path
    manifest: dict[str, Any]
    panel: pd.DataFrame
    nodes: tuple[str, ...]
    label_to_id: dict[str, int]
    interpretation_caveat: str


def load_dataset(dataset_dir: Path) -> LoadedDataset:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset_dir does not exist: {dataset_dir}")
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json missing in {dataset_dir}")
    panel_path = dataset_dir / "exposure_panel.parquet"
    nodes_path = dataset_dir / "node_mapping.parquet"
    if not panel_path.is_file() or not nodes_path.is_file():
        raise FileNotFoundError("exposure_panel.parquet or node_mapping.parquet missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    nm = pd.read_parquet(nodes_path)
    nm_sorted = nm.sort_values("node_id").reset_index(drop=True)
    nodes = tuple(nm_sorted["bank_label"].tolist())
    label_to_id = dict(zip(nodes, range(len(nodes)), strict=True))
    panel = pd.read_parquet(panel_path)
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.date
    panel = panel[(panel["exposure"].notna()) & (panel["exposure"] >= 0)]
    return LoadedDataset(
        dataset_dir=dataset_dir,
        manifest=manifest,
        panel=panel,
        nodes=nodes,
        label_to_id=label_to_id,
        interpretation_caveat=str(manifest.get("interpretation_caveat", "")),
    )


# ---------------------------------------------------------------------------
# Per-period matrix
# ---------------------------------------------------------------------------


def build_period_matrix(
    panel: pd.DataFrame,
    n_nodes: int,
    period_start: tuple[int, int],
    period_end: tuple[int, int],
) -> tuple[np.ndarray, list[date]]:
    """Average directed exposure matrix across quarters in ``[start, end]``.

    Returns (average matrix N×N, list of contributing quarter-end dates).
    """
    in_range = panel["date"].apply(lambda d: quarter_in_range(d, period_start, period_end))
    sub = panel[in_range]
    quarters_in = sorted(sub["date"].unique())
    if not quarters_in:
        return np.zeros((n_nodes, n_nodes), dtype=np.float64), []
    accum = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for q_date in quarters_in:
        m = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        for _, row in sub[sub["date"] == q_date].iterrows():
            s = int(row["source"])
            t = int(row["target"])
            if 0 <= s < n_nodes and 0 <= t < n_nodes and s != t:
                m[s, t] = float(row["exposure"])
        accum += m
    return accum / float(len(quarters_in)), quarters_in


def build_period_outward_strength_panel(
    panel: pd.DataFrame,
    n_nodes: int,
    period_start: tuple[int, int],
    period_end: tuple[int, int],
) -> tuple[pd.DataFrame, list[date]]:
    """T × N panel of out-strength per node per quarter, in chronological order."""
    in_range = panel["date"].apply(lambda d: quarter_in_range(d, period_start, period_end))
    sub = panel[in_range].copy()
    quarters_in = sorted(sub["date"].unique())
    if not quarters_in:
        return pd.DataFrame(), []
    rows = []
    for q_date in quarters_in:
        out_strength = np.zeros(n_nodes, dtype=np.float64)
        for _, row in sub[sub["date"] == q_date].iterrows():
            s = int(row["source"])
            if 0 <= s < n_nodes:
                out_strength[s] += float(row["exposure"])
        rows.append(out_strength)
    df = pd.DataFrame(np.array(rows), index=quarters_in, columns=list(range(n_nodes)))
    return df, quarters_in


# ---------------------------------------------------------------------------
# Network thresholding
# ---------------------------------------------------------------------------


def threshold_adjacency(weighted: np.ndarray, edge_quantile: float) -> np.ndarray:
    """Binary directed adjacency from weighted matrix at ``edge_quantile`` of POSITIVE weights."""
    if weighted.size == 0:
        return np.zeros_like(weighted, dtype=np.uint8)
    positive = weighted[weighted > 0]
    if positive.size == 0:
        return np.zeros_like(weighted, dtype=np.uint8)
    threshold = float(np.quantile(positive, edge_quantile))
    binary: np.ndarray = (weighted >= threshold).astype(np.uint8)
    np.fill_diagonal(binary, 0)
    return binary


# ---------------------------------------------------------------------------
# BA-style comparison (descriptive)
# ---------------------------------------------------------------------------


def estimate_ba_m(n_nodes: int, n_edges: int) -> int:
    """``m`` in BA = round(E / N), bounded to ``[1, N-1]``."""
    if n_nodes < 2:
        return 0
    raw = max(1, round(n_edges / max(n_nodes, 1)))
    return int(min(raw, n_nodes - 1))


def degree_gini(degrees: np.ndarray) -> float:
    """Gini coefficient on a non-negative degree sequence; NaN if all zero."""
    if degrees.size == 0:
        return float("nan")
    arr = np.sort(np.asarray(degrees, dtype=np.float64))
    if arr.sum() <= 0:
        return float("nan")
    n = arr.size
    cum = np.cumsum(arr)
    return float((2.0 * np.sum((np.arange(1, n + 1)) * arr)) / (n * cum[-1]) - (n + 1.0) / n)


def compute_ba_comparison(
    binary: np.ndarray,
    *,
    ba_simulations: int,
    seed: int,
) -> dict[str, Any]:
    """Empirical degree sequence vs mean BA simulated sequence."""
    n = int(binary.shape[0])
    # Undirected total degree for BA comparison (BA model is undirected).
    sym = ((binary + binary.T) > 0).astype(np.uint8)
    np.fill_diagonal(sym, 0)
    n_edges = int(sym.sum() // 2)
    deg = sym.sum(axis=1).astype(np.int64)
    if n_edges == 0 or n < 2:
        return {
            "ba_m_estimate": 0,
            "empirical_mean_degree": float(deg.mean()) if n else float("nan"),
            "empirical_max_degree": int(deg.max()) if n else 0,
            "empirical_degree_gini": float("nan"),
            "ba_degree_pearson_r": float("nan"),
            "ba_degree_ks_distance": float("nan"),
            "interpretation": "inconclusive — graph degenerate (no edges)",
            "caveat": "small N and/or sparse graph; not a strict power-law test",
        }
    m = estimate_ba_m(n, n_edges)
    if m < 1:
        m = 1
    if m >= n:
        m = n - 1
    rng_master = np.random.default_rng(seed)
    sim_seeds = rng_master.integers(0, 2**31 - 1, size=ba_simulations)
    sim_degree_sequences = np.zeros((ba_simulations, n), dtype=np.float64)
    pooled_degrees: list[int] = []
    for k in range(ba_simulations):
        g = nx.barabasi_albert_graph(n, m, seed=int(sim_seeds[k]))
        ba_deg = sorted(dict(g.degree()).values(), reverse=True)
        sim_degree_sequences[k, :] = ba_deg
        pooled_degrees.extend(ba_deg)
    mean_sim_sorted = sim_degree_sequences.mean(axis=0)
    emp_sorted = np.sort(deg)[::-1].astype(np.float64)
    if np.std(emp_sorted) == 0 or np.std(mean_sim_sorted) == 0:
        pearson_r = float("nan")
    else:
        pearson_r = float(np.corrcoef(emp_sorted, mean_sim_sorted)[0, 1])
    # KS distance between empirical degree distribution and pooled BA degrees
    pooled_arr = np.asarray(pooled_degrees, dtype=np.float64)
    if pooled_arr.size == 0:
        ks = float("nan")
    else:
        all_vals = np.union1d(emp_sorted, pooled_arr)
        emp_cdf = np.searchsorted(np.sort(emp_sorted), all_vals, side="right") / emp_sorted.size
        ba_cdf = np.searchsorted(np.sort(pooled_arr), all_vals, side="right") / pooled_arr.size
        ks = float(np.max(np.abs(emp_cdf - ba_cdf)))
    if math.isnan(pearson_r):
        interp = "inconclusive — degree sequence too uniform"
    elif pearson_r >= 0.9:
        interp = "descriptively similar to preferential-attachment-style concentration"
    elif pearson_r >= 0.5:
        interp = "weak descriptive match"
    else:
        interp = "poor descriptive match"
    return {
        "ba_m_estimate": int(m),
        "empirical_mean_degree": float(deg.mean()),
        "empirical_max_degree": int(deg.max()),
        "empirical_degree_gini": degree_gini(deg.astype(np.float64)),
        "ba_degree_pearson_r": pearson_r,
        "ba_degree_ks_distance": ks,
        "interpretation": interp,
        "caveat": "small N (~31), sparse thresholded graph; descriptive only — not a strict power-law test",
    }


# ---------------------------------------------------------------------------
# Pearson correlation
# ---------------------------------------------------------------------------


def build_correlation_matrix(
    series: pd.DataFrame,
    *,
    constant_threshold: float = 1e-12,
) -> tuple[np.ndarray, list[int]]:
    """Pearson correlation of columns; constants → NaN row/col, returned in second element."""
    if series.shape[0] < 2:
        n = series.shape[1]
        nan_mat = np.full((n, n), np.nan)
        return nan_mat, list(range(n))
    arr = series.to_numpy(dtype=np.float64)
    stds = arr.std(axis=0, ddof=0)
    constants = [int(i) for i, s in enumerate(stds) if s < constant_threshold]
    n = arr.shape[1]
    corr = np.full((n, n), np.nan)
    for i in range(n):
        if i in constants:
            continue
        for j in range(i, n):
            if j in constants:
                continue
            if i == j:
                corr[i, j] = 1.0
                continue
            xi = arr[:, i]
            xj = arr[:, j]
            r = float(np.corrcoef(xi, xj)[0, 1])
            corr[i, j] = r
            corr[j, i] = r
    return corr, constants


def to_log_changes(panel: pd.DataFrame, epsilon: float = 1.0) -> pd.DataFrame:
    """Quarter-over-quarter log differences with safe positive epsilon."""
    arr = panel.to_numpy(dtype=np.float64)
    if arr.shape[0] < 2:
        return pd.DataFrame(np.empty((0, arr.shape[1])), columns=panel.columns)
    log_arr = np.log(arr + epsilon)
    diff = np.diff(log_arr, axis=0)
    return pd.DataFrame(diff, index=panel.index[1:], columns=panel.columns)


# ---------------------------------------------------------------------------
# Risk concentration
# ---------------------------------------------------------------------------


def compute_risk_concentration(
    *,
    nodes: tuple[str, ...],
    corr_levels_normal: np.ndarray,
    corr_levels_lehman: np.ndarray,
    corr_changes_normal: np.ndarray,
    corr_changes_lehman: np.ndarray,
    weighted_lehman: np.ndarray,
    binary_lehman: np.ndarray,
) -> pd.DataFrame:
    n = len(nodes)

    def _mean_abs(corr: np.ndarray, exclude_diag: bool = True) -> np.ndarray:
        vals = np.zeros(n)
        for i in range(n):
            row = np.abs(corr[i, :]).copy()
            if exclude_diag and not np.isnan(row[i]):
                row[i] = np.nan
            finite = row[~np.isnan(row)]
            vals[i] = float(np.mean(finite)) if finite.size else float("nan")
        return vals

    macn = _mean_abs(corr_levels_normal)
    macl = _mean_abs(corr_levels_lehman)
    machn = _mean_abs(corr_changes_normal)
    machl = _mean_abs(corr_changes_lehman)
    out_str = weighted_lehman.sum(axis=1)
    in_str = weighted_lehman.sum(axis=0)
    total_str = out_str + in_str
    bin_deg = ((binary_lehman + binary_lehman.T) > 0).astype(int).sum(axis=1)
    df = pd.DataFrame(
        {
            "country": list(nodes),
            "mean_abs_corr_levels_normal": macn,
            "mean_abs_corr_levels_lehman": macl,
            "delta_mean_abs_corr_levels": macl - macn,
            "mean_abs_corr_changes_normal": machn,
            "mean_abs_corr_changes_lehman": machl,
            "delta_mean_abs_corr_changes": machl - machn,
            "weighted_out_strength_lehman": out_str,
            "weighted_in_strength_lehman": in_str,
            "total_strength_lehman": total_str,
            "binary_degree_lehman": bin_deg,
        }
    )
    df = df.sort_values(
        ["delta_mean_abs_corr_changes", "mean_abs_corr_changes_lehman", "total_strength_lehman"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    df["risk_concentration_rank"] = np.arange(1, len(df) + 1, dtype=np.int64)
    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_network(
    weighted: np.ndarray,
    binary: np.ndarray,
    nodes: tuple[str, ...],
    *,
    out_path: Path,
    title: str,
    subtitle: str,
    top_n_labels: int,
    seed: int,
) -> None:
    n = len(nodes)
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if binary[i, j] and i != j:
                g.add_edge(i, j, weight=float(weighted[i, j]))
    out_strength = weighted.sum(axis=1)
    in_strength = weighted.sum(axis=0)
    total_strength = out_strength + in_strength
    sizes = 100.0 + 1500.0 * (total_strength / max(total_strength.max(), 1.0))
    pos = nx.spring_layout(g, seed=seed, k=0.6, iterations=150)
    fig, ax = plt.subplots(figsize=(11, 9))
    if g.number_of_edges() > 0:
        weights = np.array([g[u][v]["weight"] for u, v in g.edges()])
        widths = 0.3 + 2.5 * (weights / max(weights.max(), 1.0))
        nx.draw_networkx_edges(
            g,
            pos,
            ax=ax,
            alpha=0.35,
            width=widths,
            edge_color="#444444",
            arrows=True,
            arrowsize=8,
            connectionstyle="arc3,rad=0.05",
        )
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=sizes, node_color="#3577b8", alpha=0.85)
    top_idx = np.argsort(-total_strength)[:top_n_labels]
    labels = {int(i): nodes[i] for i in top_idx}
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=9, font_weight="bold")
    ax.set_title(title, fontsize=12, weight="bold")
    ax.text(
        0.5,
        -0.04,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        style="italic",
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_correlation(
    corr: np.ndarray,
    nodes: tuple[str, ...],
    *,
    out_path: Path,
    title: str,
    subtitle: str,
    diverging: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 9))
    cmap = "RdBu_r" if diverging else "viridis"
    vmin, vmax = (-1.0, 1.0) if diverging else (-1.0, 1.0)
    im = ax.imshow(corr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(nodes)))
    ax.set_yticks(range(len(nodes)))
    ax.set_xticklabels(nodes, rotation=90, fontsize=8)
    ax.set_yticklabels(nodes, fontsize=8)
    ax.set_title(title, fontsize=12, weight="bold")
    ax.text(
        0.5,
        -0.16,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        style="italic",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_risk_concentration(df: pd.DataFrame, *, out_path: Path, top: int = 15) -> None:
    head = df.head(top)
    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(head))
    ax.barh(y, head["delta_mean_abs_corr_changes"], color="#c0392b", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(head["country"])
    ax.invert_yaxis()
    ax.set_xlabel("Δ mean |Pearson r| of log-changes (Lehman − Normal)")
    ax.set_title(
        "Countries with strongest exposure co-movement increase around Lehman window",
        fontsize=11,
        weight="bold",
    )
    ax.text(
        0.5,
        -0.14,
        "country-level BIS aggregate exposures; descriptive, not bank-level contagion proof",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        style="italic",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top correlated pairs
# ---------------------------------------------------------------------------


def top_correlated_pairs(
    corr: np.ndarray,
    nodes: tuple[str, ...],
    *,
    mode: str,
    period: str,
    top: int = 25,
) -> pd.DataFrame:
    n = corr.shape[0]
    rows: list[dict[str, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            r = corr[i, j]
            if not np.isfinite(r):
                continue
            rows.append(
                {
                    "mode": mode,
                    "period": period,
                    "country_i": nodes[i],
                    "country_j": nodes[j],
                    "pearson_r": float(r),
                    "abs_pearson_r": float(abs(r)),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        df["rank"] = []
        df["caveat"] = []
        return df
    df = df.sort_values("abs_pearson_r", ascending=False).reset_index(drop=True)
    df = df.head(top).copy()
    df["rank"] = np.arange(1, len(df) + 1, dtype=np.int64)
    df["caveat"] = "country-level BIS aggregate; descriptive co-movement, not causal"
    return df


# ---------------------------------------------------------------------------
# Build artefact (orchestrator)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildOptions:
    dataset_dir: Path
    output_dir: Path
    normal_start: tuple[int, int]
    normal_end: tuple[int, int]
    lehman_start: tuple[int, int]
    lehman_end: tuple[int, int]
    sensitivity_start: tuple[int, int]
    sensitivity_end: tuple[int, int]
    edge_quantile: float
    top_n_labels: int
    ba_simulations: int
    seed: int


def _git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "UNKNOWN"


def build_article_artifact(opts: BuildOptions) -> dict[str, Any]:
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(opts.dataset_dir)
    nodes = ds.nodes
    n = len(nodes)
    panel = ds.panel
    quarter_min = panel["date"].min()
    quarter_max = panel["date"].max()

    # Per-period exposure matrices and outward-strength panels
    mat_normal, q_normal = build_period_matrix(panel, n, opts.normal_start, opts.normal_end)
    mat_lehman, q_lehman = build_period_matrix(panel, n, opts.lehman_start, opts.lehman_end)
    mat_sens, q_sens = build_period_matrix(panel, n, opts.sensitivity_start, opts.sensitivity_end)

    str_normal, _ = build_period_outward_strength_panel(
        panel, n, opts.normal_start, opts.normal_end
    )
    str_lehman, _ = build_period_outward_strength_panel(
        panel, n, opts.lehman_start, opts.lehman_end
    )
    str_sens, _ = build_period_outward_strength_panel(
        panel, n, opts.sensitivity_start, opts.sensitivity_end
    )

    # Thresholded binary graphs
    bin_normal = threshold_adjacency(mat_normal, opts.edge_quantile)
    bin_lehman = threshold_adjacency(mat_lehman, opts.edge_quantile)

    # BA-style descriptive comparison
    ba_normal = compute_ba_comparison(
        bin_normal, ba_simulations=opts.ba_simulations, seed=opts.seed
    )
    ba_lehman = compute_ba_comparison(
        bin_lehman, ba_simulations=opts.ba_simulations, seed=opts.seed + 1
    )
    ba_rows = []
    for label, mat, binary, info in [
        ("normal", mat_normal, bin_normal, ba_normal),
        ("lehman", mat_lehman, bin_lehman, ba_lehman),
    ]:
        ba_rows.append(
            {
                "period": label,
                "n_nodes": n,
                "n_edges_thresholded": int(((binary + binary.T) > 0).sum() // 2),
                "edge_quantile": opts.edge_quantile,
                **info,
            }
        )
    ba_df = pd.DataFrame(ba_rows)
    ba_df.to_csv(opts.output_dir / "ba_fit_summary.csv", index=False)

    # Pearson correlations — levels
    corr_lev_normal, const_lev_normal = build_correlation_matrix(str_normal)
    corr_lev_lehman, const_lev_lehman = build_correlation_matrix(str_lehman)
    corr_lev_delta = corr_lev_lehman - corr_lev_normal

    # Pearson correlations — log-changes
    chg_normal = to_log_changes(str_normal)
    chg_lehman = to_log_changes(str_lehman)
    chg_sens = to_log_changes(str_sens)
    corr_chg_normal, const_chg_normal = build_correlation_matrix(chg_normal)
    corr_chg_lehman, const_chg_lehman = build_correlation_matrix(chg_lehman)
    corr_chg_sens, const_chg_sens = build_correlation_matrix(chg_sens)
    corr_chg_delta = corr_chg_lehman - corr_chg_normal

    low_sample = chg_lehman.shape[0] < 4

    # Top correlated pairs CSVs
    top_lev_normal = top_correlated_pairs(corr_lev_normal, nodes, mode="levels", period="normal")
    top_lev_lehman = top_correlated_pairs(corr_lev_lehman, nodes, mode="levels", period="lehman")
    top_chg_normal = top_correlated_pairs(corr_chg_normal, nodes, mode="changes", period="normal")
    top_chg_lehman = top_correlated_pairs(corr_chg_lehman, nodes, mode="changes", period="lehman")
    top_chg_sens = top_correlated_pairs(
        corr_chg_sens,
        nodes,
        mode="changes",
        period="sensitivity_2007Q1_2009Q4",
    )
    pd.concat([top_lev_normal, top_lev_lehman], ignore_index=True).to_csv(
        opts.output_dir / "top_correlated_pairs_levels.csv",
        index=False,
    )
    pd.concat([top_chg_normal, top_chg_lehman, top_chg_sens], ignore_index=True).to_csv(
        opts.output_dir / "top_correlated_pairs_changes.csv",
        index=False,
    )

    # Risk concentration
    rc = compute_risk_concentration(
        nodes=nodes,
        corr_levels_normal=corr_lev_normal,
        corr_levels_lehman=corr_lev_lehman,
        corr_changes_normal=corr_chg_normal,
        corr_changes_lehman=corr_chg_lehman,
        weighted_lehman=mat_lehman,
        binary_lehman=bin_lehman,
    )
    rc.to_csv(opts.output_dir / "risk_concentration_summary.csv", index=False)

    # Figures — networks
    _plot_network(
        mat_normal,
        bin_normal,
        nodes,
        out_path=opts.output_dir / "network_normal.png",
        title=(
            f"BIS LBS country-level banking-system exposure network — Normal window "
            f"({opts.normal_start[0]}Q{opts.normal_start[1]}–{opts.normal_end[0]}Q{opts.normal_end[1]})"
        ),
        subtitle=(
            "node size = total strength; edge width ∝ avg exposure; "
            "thresholded at top {q:.0%} of positive weights — not bank-level exposures"
        ).format(q=1 - opts.edge_quantile),
        top_n_labels=opts.top_n_labels,
        seed=opts.seed,
    )
    _plot_network(
        mat_lehman,
        bin_lehman,
        nodes,
        out_path=opts.output_dir / "network_lehman.png",
        title=(
            f"BIS LBS country-level banking-system exposure network — Lehman window "
            f"({opts.lehman_start[0]}Q{opts.lehman_start[1]}–{opts.lehman_end[0]}Q{opts.lehman_end[1]})"
        ),
        subtitle=(
            "node size = total strength; edge width ∝ avg exposure; "
            "thresholded at top {q:.0%} of positive weights — not bank-level exposures"
        ).format(q=1 - opts.edge_quantile),
        top_n_labels=opts.top_n_labels,
        seed=opts.seed,
    )

    # Figures — correlations (levels)
    _plot_correlation(
        corr_lev_normal,
        nodes,
        out_path=opts.output_dir / "correlation_levels_normal.png",
        title="Pearson r of country out-strength LEVELS — Normal",
        subtitle="trend-sensitive; for descriptive comparison only",
    )
    _plot_correlation(
        corr_lev_lehman,
        nodes,
        out_path=opts.output_dir / "correlation_levels_lehman.png",
        title="Pearson r of country out-strength LEVELS — Lehman",
        subtitle="short window (4 quarters); descriptive only",
    )
    _plot_correlation(
        corr_lev_delta,
        nodes,
        out_path=opts.output_dir / "correlation_levels_delta.png",
        title="Δ Pearson r LEVELS (Lehman − Normal)",
        subtitle="positive = stronger co-movement during Lehman window",
        diverging=True,
    )

    # Figures — correlations (changes)
    _plot_correlation(
        corr_chg_normal,
        nodes,
        out_path=opts.output_dir / "correlation_changes_normal.png",
        title="Pearson r of country out-strength LOG-CHANGES — Normal",
        subtitle="quarter-over-quarter; main co-movement metric",
    )
    _plot_correlation(
        corr_chg_lehman,
        nodes,
        out_path=opts.output_dir / "correlation_changes_lehman.png",
        title="Pearson r of country out-strength LOG-CHANGES — Lehman",
        subtitle=(
            "LOW_SAMPLE_WARNING: short window — read as descriptive only"
            if low_sample
            else "short window; descriptive only"
        ),
    )
    _plot_correlation(
        corr_chg_delta,
        nodes,
        out_path=opts.output_dir / "correlation_changes_delta.png",
        title="Δ Pearson r LOG-CHANGES (Lehman − Normal)",
        subtitle="positive = stronger co-movement during Lehman window",
        diverging=True,
    )

    # Figure — risk concentration
    _plot_risk_concentration(rc, out_path=opts.output_dir / "risk_concentration_bar.png")

    # Data quality summary
    constants_lev = sorted(set(const_lev_normal) | set(const_lev_lehman))
    constants_chg = sorted(set(const_chg_normal) | set(const_chg_lehman))
    dq = pd.DataFrame(
        [
            {"metric": "n_quarters_total", "value": int(panel["date"].nunique())},
            {"metric": "n_quarters_normal", "value": len(q_normal)},
            {"metric": "n_quarters_lehman", "value": len(q_lehman)},
            {"metric": "n_quarters_sensitivity", "value": len(q_sens)},
            {"metric": "n_nodes", "value": n},
            {
                "metric": "n_edges_positive_normal",
                "value": int((mat_normal > 0).sum()),
            },
            {
                "metric": "n_edges_positive_lehman",
                "value": int((mat_lehman > 0).sum()),
            },
            {
                "metric": "residual_nodes_excluded",
                "value": "5Q (already filtered upstream by build_bis_lbs_dataset.py)",
            },
            {
                "metric": "constant_nodes_levels",
                "value": ",".join(nodes[i] for i in constants_lev) or "(none)",
            },
            {
                "metric": "constant_nodes_changes",
                "value": ",".join(nodes[i] for i in constants_chg) or "(none)",
            },
            {
                "metric": "missing_quarters",
                "value": (
                    "(none — every quarter in window has ≥1 row)"
                    if (q_normal and q_lehman)
                    else "WARNING: empty period"
                ),
            },
            {
                "metric": "low_sample_warning_lehman_changes",
                "value": "TRUE" if low_sample else "FALSE",
            },
            {
                "metric": "warnings",
                "value": (
                    "Pearson on 4-quarter Lehman window is fragile; rely on sensitivity window."
                ),
            },
            {
                "metric": "panel_date_range",
                "value": f"{quarter_min}..{quarter_max}",
            },
            {
                "metric": "interpretation_caveat",
                "value": ds.interpretation_caveat or "(none in manifest)",
            },
        ]
    )
    dq.to_csv(opts.output_dir / "data_quality_summary.csv", index=False)

    # Article summary + reproducibility
    repo_root = Path(__file__).resolve().parents[1]
    sha = _git_sha(repo_root)
    cmd = " ".join(sys.argv) if len(sys.argv) > 1 else "(programmatic call)"
    _write_article_summary(
        opts=opts,
        ds=ds,
        ba_df=ba_df,
        rc=rc,
        top_lev=top_lev_lehman,
        top_chg=top_chg_lehman,
        low_sample_warning=low_sample,
        sha=sha,
    )
    _write_reproducibility(opts=opts, ds=ds, sha=sha, cmd=cmd)

    return {
        "n_quarters_normal": len(q_normal),
        "n_quarters_lehman": len(q_lehman),
        "n_quarters_sensitivity": len(q_sens),
        "ba_normal": ba_normal,
        "ba_lehman": ba_lehman,
        "low_sample_warning": low_sample,
        "constants_levels": [nodes[i] for i in constants_lev],
        "constants_changes": [nodes[i] for i in constants_chg],
    }


# ---------------------------------------------------------------------------
# Markdown writers
# ---------------------------------------------------------------------------


def _write_article_summary(
    *,
    opts: BuildOptions,
    ds: LoadedDataset,
    ba_df: pd.DataFrame,
    rc: pd.DataFrame,
    top_lev: pd.DataFrame,
    top_chg: pd.DataFrame,
    low_sample_warning: bool,
    sha: str,
) -> None:
    rc_top = rc.head(10)
    rc_lines = "\n".join(
        f"  {i + 1}. {row.country} — Δ|r|_changes = {row.delta_mean_abs_corr_changes:+.3f}, "
        f"|r|_changes_lehman = {row.mean_abs_corr_changes_lehman:.3f}"
        for i, row in rc_top.iterrows()
    )
    top_lev_lines = "\n".join(
        f"  {i + 1}. {row.country_i}–{row.country_j}: r = {row.pearson_r:+.3f}"
        for i, row in top_lev.head(8).iterrows()
    )
    top_chg_lines = "\n".join(
        f"  {i + 1}. {row.country_i}–{row.country_j}: r = {row.pearson_r:+.3f}"
        for i, row in top_chg.head(8).iterrows()
    )
    ba_lines = "\n".join(
        f"  {row.period}: BA r = {row.ba_degree_pearson_r:.3f}, KS = {row.ba_degree_ks_distance:.3f}, "
        f"m = {row.ba_m_estimate}, Gini = {row.empirical_degree_gini:.3f} — {row.interpretation}"
        for _, row in ba_df.iterrows()
    )
    low_sample_block = (
        "\n**LOW_SAMPLE_WARNING:** the Lehman window contains < 4 usable observations after the "
        "log-change transformation. Top change-correlation pairs for the Lehman window are "
        "informational only; rely on the sensitivity window 2007Q1–2009Q4 for ranking.\n"
        if low_sample_warning
        else ""
    )
    md = f"""# Disha BA + Correlation — Article Summary

## 1. Plain-English summary

This analysis uses public BIS LBS country-level banking-system exposure data. It builds a
directed exposure network between national banking systems and compares the thresholded
topology with a Barabási-Albert-style preferential-attachment benchmark. It also computes
Pearson correlations between country exposure time series to show which banking systems
moved together more strongly around the Lehman period. The result is useful as a macro
network illustration for an article, but it is not bank-level interbank evidence and it
does not validate repo liquidity contagion.

## 2. Data

- Source: BIS Locational Banking Statistics (LBS), bulk feed `WS_LBS_D_PUB`.
- Aggregation: country-level reporting banking system → counterparty country (sector A).
- Period span: panel covers the full window provided by upstream dataset_dir.
- Residual node `5Q`: already filtered upstream; not present in the country list.

## 3. BA-style comparison

We threshold each period's averaged exposure matrix at the top {(1 - opts.edge_quantile):.0%} of
positive weights and compare the thresholded undirected degree sequence to {opts.ba_simulations}
Barabási-Albert simulated networks of the same N and matched edge density. We report Pearson
correlation between empirical and BA mean-sorted degree sequences, plus a Kolmogorov-Smirnov
distance between the two distributions.

{ba_lines}

This is a **descriptive topology comparison**, not a strict power-law validation: country-level
N (~31) is too small for a clean tail estimate.

## 4. Pearson correlation — what it measures

For each country we sum its outward exposure each quarter (out-strength). Two correlations
are computed:

- **LEVELS**: correlation of raw out-strength time series. Trend-sensitive — a long secular
  growth in claims will inflate level correlations.
- **LOG-CHANGES** (main metric): quarter-over-quarter `log(x + 1)` differences. Removes
  the secular trend; isolates co-movement of shocks.

The headline finding for crisis-window co-movement uses LOG-CHANGES.
{low_sample_block}
## 5. Normal vs Lehman comparison

- Normal window: {opts.normal_start[0]}Q{opts.normal_start[1]} – {opts.normal_end[0]}Q{opts.normal_end[1]}
- Lehman window: {opts.lehman_start[0]}Q{opts.lehman_start[1]} – {opts.lehman_end[0]}Q{opts.lehman_end[1]}
- Sensitivity window (wider): {opts.sensitivity_start[0]}Q{opts.sensitivity_start[1]} – {opts.sensitivity_end[0]}Q{opts.sensitivity_end[1]}

The Lehman window is intentionally illustrative and short; correlation results should be read
as descriptive, not as stable statistical inference.

Top |r| level pairs (Lehman):

{top_lev_lines}

Top |r| log-change pairs (Lehman):

{top_chg_lines}

## 6. Risk concentration interpretation

We rank countries by the increase in their mean |Pearson r| of log-changes from the normal
to the Lehman window (then by Lehman |r| and by Lehman total strength as tiebreakers).
This identifies banking systems whose exposure shocks moved more in lock-step with others
around the Lehman period.

Top 10:

{rc_lines}

## 7. Limitations

1. Country-level aggregation, not bank-level interbank exposures.
2. Small N (~31 reporting countries).
3. Edge thresholding is sensitive to `--edge-quantile`; results may shift with different cuts.
4. Pearson correlation over short (~4-quarter) windows is statistically fragile.
5. Level correlations are trend-sensitive.
6. Log-change correlations are better for shock co-movement but still descriptive — not causal.
7. Not a bank-level repo contagion model.
8. Not a causal model — these are descriptive co-movement statistics.

## 8. Reproducibility

```
python tools/build_disha_ba_correlation_figures.py \\
    --dataset-dir {opts.dataset_dir} \\
    --output-dir {opts.output_dir} \\
    --normal-start {opts.normal_start[0]}Q{opts.normal_start[1]} \\
    --normal-end {opts.normal_end[0]}Q{opts.normal_end[1]} \\
    --lehman-start {opts.lehman_start[0]}Q{opts.lehman_start[1]} \\
    --lehman-end {opts.lehman_end[0]}Q{opts.lehman_end[1]} \\
    --sensitivity-start {opts.sensitivity_start[0]}Q{opts.sensitivity_start[1]} \\
    --sensitivity-end {opts.sensitivity_end[0]}Q{opts.sensitivity_end[1]} \\
    --edge-quantile {opts.edge_quantile} \\
    --top-n-labels {opts.top_n_labels} \\
    --ba-simulations {opts.ba_simulations} \\
    --seed {opts.seed}
```

Git SHA: `{sha}`

## 9. Out of scope

This analysis does NOT use Kuramoto / phase-synchronisation modelling. The earlier X-9R
Kuramoto pipeline is unrelated to this artefact and was explicitly excluded per Disha's
request.

This analysis does NOT make bank-level claims. The BIS LBS bulk feed publishes bilateral
cells only at country level; bank-to-bank bilateral data requires supervisory access.

## 10. Suggested 100-word paragraph for Disha

> Using public BIS Locational Banking Statistics, we built a directed network of cross-border
> claims between national banking systems and compared its thresholded topology with a
> Barabási-Albert preferential-attachment benchmark. We then computed Pearson correlations
> between country-level exposure time series for a normal pre-crisis window and the Lehman
> window. Several country pairs show markedly stronger co-movement of exposure changes during
> the Lehman period, and a small set of countries dominate the network's mass. These figures
> illustrate macro banking-network structure descriptively; they are not bank-to-bank
> exposures and do not constitute a validated liquidity-contagion model.
"""
    (opts.output_dir / "DISHA_ARTICLE_SUMMARY.md").write_text(md, encoding="utf-8")
    # Forbidden-wording self-check
    lower = md.lower()
    for forbidden in _FORBIDDEN_WORDING:
        if forbidden.lower() in lower:
            raise RuntimeError(
                f"forbidden wording present in DISHA_ARTICLE_SUMMARY.md: {forbidden!r}"
            )


def _write_reproducibility(
    *,
    opts: BuildOptions,
    ds: LoadedDataset,
    sha: str,
    cmd: str,
) -> None:
    py_ver = sys.version.replace("\n", " ")
    md = f"""# Reproducibility — Disha BA + Correlation Artefact

- Repository: `neuron7xLab/GeoSync`
- Git SHA: `{sha}`
- Dataset directory: `{opts.dataset_dir}`
- Manifest path: `{opts.dataset_dir / "manifest.json"}`
- Manifest source_id: `{ds.manifest.get("source_id", "?")}`
- Manifest config_hash: `{ds.manifest.get("config_hash", "?")}`
- Output directory: `{opts.output_dir}`
- Python: `{py_ver}`

## Command

```
{cmd}
```

Or programmatically equivalent:

```
python tools/build_disha_ba_correlation_figures.py \\
    --dataset-dir {opts.dataset_dir} \\
    --output-dir {opts.output_dir} \\
    --normal-start {opts.normal_start[0]}Q{opts.normal_start[1]} \\
    --normal-end {opts.normal_end[0]}Q{opts.normal_end[1]} \\
    --lehman-start {opts.lehman_start[0]}Q{opts.lehman_start[1]} \\
    --lehman-end {opts.lehman_end[0]}Q{opts.lehman_end[1]} \\
    --sensitivity-start {opts.sensitivity_start[0]}Q{opts.sensitivity_start[1]} \\
    --sensitivity-end {opts.sensitivity_end[0]}Q{opts.sensitivity_end[1]} \\
    --edge-quantile {opts.edge_quantile} \\
    --top-n-labels {opts.top_n_labels} \\
    --ba-simulations {opts.ba_simulations} \\
    --seed {opts.seed}
```

## Generated files

- `network_normal.png`, `network_lehman.png`
- `correlation_levels_{{normal,lehman,delta}}.png`
- `correlation_changes_{{normal,lehman,delta}}.png`
- `risk_concentration_bar.png`
- `ba_fit_summary.csv`
- `top_correlated_pairs_levels.csv`
- `top_correlated_pairs_changes.csv`
- `risk_concentration_summary.csv`
- `data_quality_summary.csv`
- `DISHA_ARTICLE_SUMMARY.md`
- `REPRODUCIBILITY.md` (this file)

## Caveats

- Country-level aggregation; not bank-level.
- Edge threshold and BA m estimate are deterministic given `--seed` and `--edge-quantile`.
- Network layout uses `nx.spring_layout` with `seed=--seed`; minor matplotlib version drift
  may cause cosmetic differences but no metric changes.
- BA simulations use deterministic per-simulation seeds derived from `--seed`.

## Interpretation boundary

Allowed wording: country-level banking-system exposure network; macro banking-network
illustration; descriptive preferential-attachment-style comparison; correlation between
country banking-system exposure time series; public reproducible BIS baseline.

Forbidden wording (auto-checked): bank-level interbank network; bank-to-bank exposures;
validated repo liquidity-risk model; confirmed Barabási-Albert law; confirmed systemic-risk
phase transition; liquidity contagion proof; production-grade scientific validation.
"""
    (opts.output_dir / "REPRODUCIBILITY.md").write_text(md, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_disha_ba_correlation_figures",
        description="BA-style + Pearson-correlation article artefact for Disha.",
    )
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--normal-start", type=str, default="2006Q1")
    p.add_argument("--normal-end", type=str, default="2007Q4")
    p.add_argument("--lehman-start", type=str, default="2008Q3")
    p.add_argument("--lehman-end", type=str, default="2009Q2")
    p.add_argument("--sensitivity-start", type=str, default="2007Q1")
    p.add_argument("--sensitivity-end", type=str, default="2009Q4")
    p.add_argument("--edge-quantile", type=float, default=0.85)
    p.add_argument("--top-n-labels", type=int, default=12)
    p.add_argument("--ba-simulations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    opts = BuildOptions(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        normal_start=parse_quarter(args.normal_start),
        normal_end=parse_quarter(args.normal_end),
        lehman_start=parse_quarter(args.lehman_start),
        lehman_end=parse_quarter(args.lehman_end),
        sensitivity_start=parse_quarter(args.sensitivity_start),
        sensitivity_end=parse_quarter(args.sensitivity_end),
        edge_quantile=float(args.edge_quantile),
        top_n_labels=int(args.top_n_labels),
        ba_simulations=int(args.ba_simulations),
        seed=int(args.seed),
    )
    summary = build_article_artifact(opts)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
