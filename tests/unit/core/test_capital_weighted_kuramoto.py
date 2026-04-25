# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Tests for capital-weighted (β) Kuramoto coupling primitives.

INV-KBETA — capital-weighted coupling preserves finiteness, non-negativity,
symmetry, zero diagonal, and is invariant under uniform depth scaling.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from core.kuramoto.capital_weighted import (
    CapitalWeightedCouplingConfig,
    L2DepthSnapshot,
    build_capital_weighted_adjacency,
    compute_capital_ratio,
    compute_k_beta,
    compute_l2_depth_mass,
    estimate_scalar_beta,
)


def _baseline_adj(n: int, off: float = 0.5) -> NDArray[np.float64]:
    adj = np.full((n, n), off, dtype=np.float64)
    np.fill_diagonal(adj, 0.0)
    return adj


def _snapshot(n: int, levels: int = 3, *, seed: int = 0, ts: int = 1_000) -> L2DepthSnapshot:
    rng = np.random.default_rng(seed)
    bid = rng.uniform(0.0, 5.0, size=(n, levels)).astype(np.float64)
    ask = rng.uniform(0.0, 5.0, size=(n, levels)).astype(np.float64)
    mid = rng.uniform(50.0, 150.0, size=n).astype(np.float64)
    return L2DepthSnapshot(timestamp_ns=ts, bid_sizes=bid, ask_sizes=ask, mid_prices=mid)


def test_kbeta_finite_bounded_symmetric_zero_diag() -> None:
    """INV-KBETA: K_ij is finite, non-negative, symmetric, zero diagonal."""
    n = 6
    base = _baseline_adj(n, off=0.4)
    snap = _snapshot(n, seed=11, ts=500)
    cfg = CapitalWeightedCouplingConfig()

    res = build_capital_weighted_adjacency(base, snap, signal_timestamp_ns=1_000, cfg=cfg)
    K = res.coupling

    violations: list[str] = []
    if not np.isfinite(K).all():
        violations.append("non-finite values")
    if (K < -1e-12).any():
        violations.append("negative values")
    if not np.allclose(K, K.T, atol=1e-10):
        violations.append("asymmetry")
    if not np.allclose(np.diag(K), 0.0, atol=1e-12):
        violations.append("non-zero diagonal")

    assert not violations, (
        f"INV-KBETA VIOLATED: K_ij must be finite, bounded, symmetric, "
        f"zero-diag. Observed violations={violations}; min={K.min():.6f}, "
        f"max={K.max():.6f}, with N={n}, beta={res.beta:.4f}, "
        f"used_fallback={res.used_fallback}."
    )


def test_beta_one_recovers_baseline_adjacency() -> None:
    """INV-KBETA: β=1 (forced via direct API) is the identity envelope."""
    n = 5
    base = _baseline_adj(n, off=0.3)
    cfg = CapitalWeightedCouplingConfig(K0=1.0, gamma=0.7, delta=1.5)
    r = np.linspace(0.5, 1.5, n).astype(np.float64)
    envelope = compute_k_beta(r, beta=1.0, cfg=cfg)

    expected = np.full(n, cfg.K0, dtype=np.float64)
    np.testing.assert_allclose(
        envelope,
        expected,
        atol=1e-12,
        err_msg=(
            "INV-KBETA VIOLATED: beta=1.0 should recover K0 envelope; "
            f"observed envelope={envelope}, expected={expected}, with N={n}, "
            f"K0={cfg.K0}, gamma={cfg.gamma}, delta={cfg.delta}."
        ),
    )

    # And the full adjacency builder with beta=1 should leave baseline
    # invariant up to renormalization (gamma=0 -> envelope ≡ K0 ≡ 1).
    cfg2 = CapitalWeightedCouplingConfig(K0=1.0, gamma=0.0, normalize=False)
    snap = _snapshot(n, seed=3, ts=10)
    res = build_capital_weighted_adjacency(base, snap, signal_timestamp_ns=20, cfg=cfg2)
    np.testing.assert_allclose(
        res.coupling,
        base,
        atol=1e-12,
        err_msg=(
            "INV-KBETA VIOLATED: with gamma=0 the envelope is constant K0=1 "
            "and must reproduce the baseline; observed max-abs-diff="
            f"{np.max(np.abs(res.coupling - base)):.3e}, with N={n}, "
            f"normalize=False, K0=1.0."
        ),
    )


def test_missing_l2_fallback_is_explicit() -> None:
    """INV-KBETA: missing snapshot returns baseline with explicit reason."""
    base = _baseline_adj(4, off=0.25)
    cfg = CapitalWeightedCouplingConfig()
    res = build_capital_weighted_adjacency(base, None, signal_timestamp_ns=0, cfg=cfg)

    assert res.used_fallback is True, "used_fallback flag must be True on None snapshot"
    assert (
        res.reason == "no_l2_snapshot"
    ), f"fallback reason must be 'no_l2_snapshot'; observed='{res.reason}'"
    np.testing.assert_array_equal(res.coupling, base)


def test_future_l2_snapshot_rejected() -> None:
    """INV-KBETA: snapshot.timestamp_ns > signal_timestamp_ns is rejected."""
    base = _baseline_adj(3)
    snap = _snapshot(3, seed=0, ts=2_000)
    cfg = CapitalWeightedCouplingConfig(fail_on_future_l2=True)
    with pytest.raises(ValueError, match="look-ahead"):
        build_capital_weighted_adjacency(base, snap, signal_timestamp_ns=1_000, cfg=cfg)


def test_depth_mass_non_negative_and_finite() -> None:
    """INV-KBETA: depth_mass is finite and non-negative across many random snapshots."""
    rng = np.random.default_rng(123)
    failures: list[str] = []
    for trial in range(20):
        n = int(rng.integers(2, 12))
        snap = _snapshot(n, levels=int(rng.integers(1, 6)), seed=int(rng.integers(0, 10_000)))
        m = compute_l2_depth_mass(snap)
        if not np.isfinite(m).all():
            failures.append(f"trial={trial} non-finite depth_mass")
        if (m < 0.0).any():
            failures.append(f"trial={trial} negative depth_mass min={m.min()}")
    assert not failures, (
        f"INV-KBETA VIOLATED: depth_mass must be finite & non-negative; "
        f"violations={failures} across n_trials=20."
    )


def test_scalar_beta_deterministic() -> None:
    """INV-KBETA: estimate_scalar_beta is a pure deterministic function."""
    snap = _snapshot(7, seed=42, ts=10)
    m = compute_l2_depth_mass(snap)
    b1 = estimate_scalar_beta(m, beta_min=0.25, beta_max=4.0)
    b2 = estimate_scalar_beta(m, beta_min=0.25, beta_max=4.0)
    assert b1 == b2, (
        f"INV-KBETA VIOLATED: estimate_scalar_beta must be deterministic; "
        f"observed b1={b1} vs b2={b2}, expected equality, params=N=7,seed=42."
    )
    assert 0.25 <= b1 <= 4.0, (
        f"INV-KBETA VIOLATED: beta out of bounds; observed beta={b1}, "
        f"expected [0.25, 4.0], with N=7, seed=42."
    )


def test_scale_invariance_under_uniform_depth_scaling() -> None:
    """INV-KBETA: K_ij is invariant under uniform multiplicative depth scaling.

    Multiplying every bid/ask size by c>0 leaves r_i and beta unchanged,
    therefore K_ij is unchanged (the median scales identically with the
    mass, and Gini is scale-free).
    """
    n = 5
    base = _baseline_adj(n, off=0.6)
    cfg = CapitalWeightedCouplingConfig(normalize=False)
    snap = _snapshot(n, seed=7, ts=10)
    res1 = build_capital_weighted_adjacency(base, snap, signal_timestamp_ns=20, cfg=cfg)

    failures: list[str] = []
    for c in (0.5, 2.0, 10.0, 1000.0):
        scaled = L2DepthSnapshot(
            timestamp_ns=snap.timestamp_ns,
            bid_sizes=(snap.bid_sizes * c).astype(np.float64),
            ask_sizes=(snap.ask_sizes * c).astype(np.float64),
            mid_prices=snap.mid_prices,
        )
        res2 = build_capital_weighted_adjacency(base, scaled, signal_timestamp_ns=20, cfg=cfg)
        diff = float(np.max(np.abs(res1.coupling - res2.coupling)))
        if diff > 1e-9:
            failures.append(f"c={c} max|ΔK|={diff:.3e}")
    assert not failures, (
        "INV-KBETA VIOLATED: K_ij must be invariant under uniform depth scaling; "
        f"violations={failures} expected diff<1e-9 across c∈{{0.5,2,10,1000}}, N=5."
    )


def test_no_self_coupling() -> None:
    """INV-KBETA: diagonal of K must be zero by construction."""
    n = 8
    base = _baseline_adj(n, off=0.42)
    snap = _snapshot(n, seed=99, ts=10)
    cfg = CapitalWeightedCouplingConfig()
    res = build_capital_weighted_adjacency(base, snap, signal_timestamp_ns=20, cfg=cfg)
    np.testing.assert_allclose(
        np.diag(res.coupling),
        0.0,
        atol=1e-12,
        err_msg=(
            "INV-KBETA VIOLATED: self-coupling must be zero; observed "
            f"max|K_ii|={np.max(np.abs(np.diag(res.coupling))):.3e}, expected 0, "
            f"with N={n}, seed=99."
        ),
    )


def test_negative_sizes_rejected() -> None:
    """Validation: negative bid/ask sizes raise."""
    bid = np.array([[-1.0, 1.0]], dtype=np.float64)
    ask = np.array([[1.0, 1.0]], dtype=np.float64)
    mid = np.array([100.0], dtype=np.float64)
    snap = L2DepthSnapshot(timestamp_ns=0, bid_sizes=bid, ask_sizes=ask, mid_prices=mid)
    with pytest.raises(ValueError, match="non-negative"):
        compute_l2_depth_mass(snap)


def test_non_positive_mid_rejected() -> None:
    """Validation: zero or negative mid prices raise."""
    bid = np.ones((1, 2), dtype=np.float64)
    ask = np.ones((1, 2), dtype=np.float64)
    mid = np.array([0.0], dtype=np.float64)
    snap = L2DepthSnapshot(timestamp_ns=0, bid_sizes=bid, ask_sizes=ask, mid_prices=mid)
    with pytest.raises(ValueError, match="strictly positive"):
        compute_l2_depth_mass(snap)


def test_capital_ratio_floor() -> None:
    """compute_capital_ratio uses the floor when median is degenerate."""
    m = np.zeros(5, dtype=np.float64)
    r = compute_capital_ratio(m, floor=1e-12)
    assert np.all(r == 0.0)
