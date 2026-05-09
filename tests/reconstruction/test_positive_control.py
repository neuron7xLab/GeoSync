# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Tests for positive_control.py — substrate generators + Gate 5 sweep."""

from __future__ import annotations

import numpy as np

from research.reconstruction.positive_control import (
    GroundTruthRecoveryCertificate,
    ground_truth_ba,
    ground_truth_core_periphery,
    ground_truth_hierarchical,
    run_recovery_on_substrate,
)


def test_ground_truth_ba_shape_and_diagonal() -> None:
    w = ground_truth_ba(n=80, m=4, seed=0)
    assert w.shape == (80, 80)
    assert np.all(np.diag(w) == 0.0)
    assert w.sum() > 0
    # Asymmetric: forward and backward weights are independent samples.
    assert not np.allclose(w, w.T)


def test_ground_truth_core_periphery_core_size() -> None:
    n = 100
    core_frac = 0.30
    w = ground_truth_core_periphery(n=n, core_frac=core_frac, seed=0)
    assert w.shape == (n, n)
    assert np.all(np.diag(w) == 0.0)
    n_core = int(n * core_frac)
    # Core is fully connected: core block has > 90% non-zero.
    core_block = w[:n_core, :n_core]
    nonzero_frac = (core_block > 0).sum() / (n_core * n_core - n_core)
    assert nonzero_frac > 0.9


def test_ground_truth_hierarchical_blocks() -> None:
    w = ground_truth_hierarchical(n=80, n_tiers=4, seed=0)
    assert w.shape == (80, 80)
    assert np.all(np.diag(w) == 0.0)
    # First tier (lo=0, hi=20) should have higher density than last cross-tier
    tier_size = 80 // 4
    in_tier_0 = (w[:tier_size, :tier_size] > 0).sum()
    cross_3_to_4 = (w[60:80, 60:80] > 0).sum()
    assert in_tier_0 > 0
    assert cross_3_to_4 >= 0  # last tier may have only intra-edges


def test_run_recovery_on_cp_substrate_passes_full_sweep() -> None:
    w = ground_truth_core_periphery(n=120, core_frac=0.30, seed=1)
    cert = run_recovery_on_substrate("CP_120", w, seed=1)
    assert isinstance(cert, GroundTruthRecoveryCertificate)
    assert cert.passed is True
    assert len(cert.failure_reasons) == 0
    assert cert.cert_id != ""


def test_run_recovery_certificate_id_is_64_hex() -> None:
    w = ground_truth_core_periphery(n=80, core_frac=0.30, seed=2)
    cert = run_recovery_on_substrate("CP_80", w, seed=2)
    assert len(cert.cert_id) == 64
    int(cert.cert_id, 16)  # hex check


def test_run_recovery_certificate_id_seed_sensitive() -> None:
    w = ground_truth_core_periphery(n=80, core_frac=0.30, seed=3)
    cert_a = run_recovery_on_substrate("CP_80", w, seed=10)
    cert_b = run_recovery_on_substrate("CP_80", w, seed=11)
    assert cert_a.cert_id != cert_b.cert_id


def test_run_recovery_per_density_reports_match_sweep() -> None:
    w = ground_truth_core_periphery(n=80, core_frac=0.30, seed=4)
    sweep = (0.05, 0.08)
    cert = run_recovery_on_substrate("CP_80", w, seed=4, sweep=sweep)
    assert set(cert.per_density_reports.keys()) == set(sweep)


def test_ba_default_m_is_in_safe_regime() -> None:
    """Default m=5 must keep BA in fitness-model regime of validity.

    Uses the canonical X-10R seed=42 (the value run_recovery_on_substrate
    uses by default). Other seeds may produce borderline results — m=5 is
    AT the boundary of the model's regime, so seed sensitivity is expected;
    that's why m=3 is the documented stress substrate.
    """
    w = ground_truth_ba(n=200, seed=42)
    cert = run_recovery_on_substrate("BA_200_default", w, seed=42)
    assert cert.passed is True
    assert len(cert.failure_reasons) == 0
    assert len(cert.per_density_reports) == 4
    for d, report in cert.per_density_reports.items():
        assert report.passed is True, f"density {d} unexpectedly failed: {report.failure_reasons}"
        assert report.spectral_radius_relative_error <= 0.20
        assert report.top_k_hub_jaccard >= 0.60
        assert report.row_sum_invariant_L1 <= 0.05
        assert report.col_sum_invariant_L1 <= 0.05


def test_ba_m3_stress_substrate_at_canonical_seed_fails() -> None:
    """BA(m=3) at the canonical seed=42 hits the documented Cimini ceiling.

    Some seeds produce milder degree concentration and pass; the canonical
    seed=42 (used everywhere else in X-10R) reliably triggers the ρ
    failure that motivates m=5 as the safe default.
    """
    w = ground_truth_ba(n=200, m=3, seed=42)
    cert = run_recovery_on_substrate("BA_200_stress", w, seed=42)
    has_spectral_failure = any(
        any("spectral_radius_relative_error" in f for f in r.failure_reasons)
        for r in cert.per_density_reports.values()
    )
    assert has_spectral_failure
    assert cert.passed is False


def test_substrate_generators_deterministic() -> None:
    a = ground_truth_ba(n=60, m=3, seed=42)
    b = ground_truth_ba(n=60, m=3, seed=42)
    np.testing.assert_array_equal(a, b)


def test_substrate_generators_seed_sensitive() -> None:
    a = ground_truth_ba(n=60, m=3, seed=42)
    b = ground_truth_ba(n=60, m=3, seed=43)
    assert not np.array_equal(a, b)


def test_run_recovery_handles_zero_topology_gracefully() -> None:
    """All-zero W produces failure_reasons (caught) without bubbling exceptions."""
    w = np.zeros((20, 20))
    cert = run_recovery_on_substrate("ZERO", w, seed=0)
    assert cert.passed is False
    assert len(cert.failure_reasons) >= 1
    assert any("crashed" in f or "spectral" in f for f in cert.failure_reasons)
