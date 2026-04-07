# mypy: disable-error-code="attr-defined,unused-ignore,no-untyped-call,arg-type"
"""Tests for value functions: P&L attribution + dislocation detection."""

from __future__ import annotations

from geosync.neuroeconomics.dislocation_detector import DislocationDetector
from geosync.neuroeconomics.pnl_attribution import PnLAttributor

# === P&L Attribution ===


def test_attribution_tracks_regime_pnl() -> None:
    attr = PnLAttributor()
    attr.record(regime="METASTABLE", decision="TRADE", pnl=0.01, size=0.8)
    attr.record(regime="METASTABLE", decision="TRADE", pnl=-0.005, size=0.8)
    attr.record(
        regime="CRITICAL", decision="ABORT", pnl=0.0, size=0.0, hypothetical_pnl=-0.05
    )

    r = attr.report()
    assert r.by_regime["METASTABLE"].count == 2
    assert r.by_regime["METASTABLE"].total_pnl == 0.005
    assert r.total_trades == 2
    assert r.total_aborts == 1


def test_attribution_protection_value() -> None:
    """ABORT that avoided a loss = positive protection value."""
    attr = PnLAttributor()
    # 3 ABORTs avoiding losses
    for _ in range(3):
        attr.record(
            regime="CRITICAL",
            decision="ABORT",
            pnl=0.0,
            size=0.0,
            hypothetical_pnl=-0.1,
        )
    r = attr.report()
    assert r.protection_value == 0.3  # saved $0.30
    assert r.abort_avoided_pnl == 0.3


def test_attribution_observe_missed_pnl() -> None:
    """OBSERVE tracks what we missed (could be gain or loss)."""
    attr = PnLAttributor()
    attr.record(
        regime="METASTABLE",
        decision="OBSERVE",
        pnl=0.0,
        size=0.0,
        hypothetical_pnl=0.05,
    )
    attr.record(
        regime="METASTABLE",
        decision="OBSERVE",
        pnl=0.0,
        size=0.0,
        hypothetical_pnl=-0.02,
    )
    r = attr.report()
    assert abs(r.observe_missed_pnl - 0.03) < 1e-6


def test_attribution_sharpe_per_regime() -> None:
    import numpy as np

    attr = PnLAttributor()
    rng = np.random.RandomState(42)
    for _ in range(50):
        attr.record(
            regime="METASTABLE",
            decision="TRADE",
            pnl=0.002 + rng.normal(0, 0.001),  # positive mean, nonzero variance
            size=1.0,
        )
    r = attr.report()
    assert r.by_regime["METASTABLE"].sharpe > 0


def test_attribution_summary_dict() -> None:
    attr = PnLAttributor()
    attr.record(regime="COHERENT", decision="TRADE", pnl=0.02, size=0.9)
    d = attr.summary_dict()
    assert "total_pnl" in d
    assert "protection_value" in d
    assert "pnl_coherent" in d
    assert "sharpe_coherent" in d


# === Dislocation Detector ===


def test_dislocation_stable_topology() -> None:
    """Stable κ, γ, R → no dislocation."""
    dd = DislocationDetector()
    for _ in range(10):
        state = dd.update(kappa=0.3, gamma=1.0, order_r=0.5)
    assert state.dislocation_score < 0.3
    assert not state.is_pre_dislocation


def test_dislocation_detects_kappa_collapse() -> None:
    """Falling κ = topology fragmenting → pre-dislocation."""
    dd = DislocationDetector()
    for i in range(15):
        kappa = 0.5 - i * 0.05  # κ falling from 0.5 to -0.2
        state = dd.update(kappa=kappa, gamma=1.0, order_r=0.5)
    assert state.kappa_velocity < 0
    assert state.dislocation_score > 0.0


def test_dislocation_detects_herding_onset() -> None:
    """R accelerating = everyone running same direction."""
    dd = DislocationDetector()
    for i in range(15):
        # Quadratic R: acceleration > 0
        r = 0.3 + 0.002 * i * i  # R = 0.3 + 0.002*i² (accelerating)
        state = dd.update(kappa=0.0, gamma=1.0, order_r=min(1.0, r))
    assert state.r_acceleration > 0


def test_dislocation_lead_time_nonzero_on_crisis() -> None:
    """When κ is falling fast, lead_bars > 0."""
    dd = DislocationDetector()
    for i in range(10):
        state = dd.update(kappa=0.5 - i * 0.1, gamma=1.0 + i * 0.05, order_r=0.5)
    assert state.lead_bars >= 3


def test_dislocation_nan_safe() -> None:
    dd = DislocationDetector()
    for _ in range(10):
        state = dd.update(kappa=float("nan"), gamma=float("inf"), order_r=float("-inf"))
    assert state.dislocation_score >= 0.0
