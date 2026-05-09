# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Discrimination layer — multi-metric Bonferroni aggregation."""

from __future__ import annotations

import numpy as np

from instrument_validation.discrimination import (
    DiscriminationVerdict,
    discriminate,
    mde_at_n31,
    metric_ks_distance,
    metric_max_degree_zscore,
    metric_zero_degree_count_error,
)


def test_mde_constant() -> None:
    assert mde_at_n31() == 0.05


def test_metric_ks_distance_basic() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    assert metric_ks_distance(a, b) == 0.0
    c = np.array([10.0, 20.0])
    assert metric_ks_distance(a, c) > 0.5


def test_metric_max_degree_zscore() -> None:
    sim = np.array([5, 6, 7, 8, 5, 6, 7, 8], dtype=np.float64)
    z_in = metric_max_degree_zscore(7.0, sim)
    z_far = metric_max_degree_zscore(20.0, sim)
    assert abs(z_in) < abs(z_far)


def test_metric_zero_degree_count_error() -> None:
    assert metric_zero_degree_count_error(0, np.array([0.0, 0.0])) == 0.0
    assert metric_zero_degree_count_error(7, np.array([0.0, 0.0, 0.0])) == 7.0


def test_discriminate_not_distinguished_when_pools_overlap() -> None:
    rng = np.random.default_rng(0)
    metrics = {
        f"M{i}": {
            "empirical": 0.5,
            "ba_pool": rng.normal(0.5, 0.1, 200),
            "er_pool": rng.normal(0.5, 0.1, 200),
        }
        for i in range(6)
    }
    report = discriminate(metrics)
    assert report.aggregate_verdict in (
        DiscriminationVerdict.NOT_DISTINGUISHED,
        DiscriminationVerdict.INSUFFICIENT_RESOLUTION,
    )


def test_discriminate_ba_favored_when_metric_outside_er_inside_ba() -> None:
    """Synthetic case where 5 of 6 metrics put empirical outside ER 95% CI
    but inside BA 95% CI — aggregate should be BA_FAVORED.
    """
    rng = np.random.default_rng(1)
    metrics: dict[str, dict[str, np.ndarray | float]] = {}
    for i in range(5):
        metrics[f"M{i}"] = {
            "empirical": 5.0,
            "ba_pool": rng.normal(5.0, 0.5, 500),
            "er_pool": rng.normal(0.0, 0.5, 500),
        }
    metrics["M5"] = {
        "empirical": 2.5,
        "ba_pool": rng.normal(2.5, 1.0, 500),
        "er_pool": rng.normal(2.5, 1.0, 500),
    }
    report = discriminate(metrics)
    assert report.n_metrics_favor_ba >= 4
    assert report.aggregate_verdict is DiscriminationVerdict.BA_FAVORED
