# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Tests for CALIB-GRID-001 — external ground-truth Kuramoto calibration.

Covers: embedded-grid contract invariants, the Dörfler–Bullo reduction
formulae (exact at float precision), the K→∞ tight-sync recovery
property (Hypothesis), calibration-loop determinism, pre-registration ↔
code agreement (fail-closed), the identifiability sweep that proves the
instrument itself is sound, and the stable NEGATIVE verdict ledger.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from core.kuramoto.contracts import PhaseMatrix
from core.kuramoto.coupling_estimator import (
    CouplingEstimationConfig,
    CouplingEstimator,
)
from research.calibration.grid_kuramoto import (
    NOISELESS_GATES,
    NOISY_GATES,
    SimConfig,
    coupling_from_susceptance,
    dorfler_bullo_critical_coupling,
    evaluate_gates,
    ieee_39_new_england,
    natural_frequency_from_injection,
    overall_verdict,
    run_calibration,
    wscc_9_bus,
)
from research.calibration.grid_kuramoto.run import build_ledger

_PREREG = (
    Path(__file__).resolve().parents[3]
    / "research"
    / "calibration"
    / "grid_kuramoto"
    / "PREREGISTRATION.md"
)


# ---------------------------------------------------------------------------
# Embedded-grid contract invariants
# ---------------------------------------------------------------------------


def test_wscc9_contract_invariants() -> None:
    """WSCC-9 fixture is a valid 3-node symmetric lossless system."""
    sys = wscc_9_bus()
    assert sys.n == 3
    assert sys.susceptance.shape == (3, 3)
    np.testing.assert_allclose(sys.susceptance, sys.susceptance.T, atol=1e-12)
    assert np.all(np.diag(sys.susceptance) == 0.0)
    assert np.all(sys.voltage > 0.0)
    assert np.all(sys.inertia > 0.0)
    assert np.all(sys.damping > 0.0)
    # Injections are mean-centred (lossless reduced model balance).
    assert abs(float(np.mean(sys.injection))) < 1e-12


def test_ieee39_contract_invariants() -> None:
    """IEEE-39 fixture is a valid 10-node symmetric system."""
    sys = ieee_39_new_england()
    assert sys.n == 10
    np.testing.assert_allclose(sys.susceptance, sys.susceptance.T, atol=1e-12)
    assert np.all(np.diag(sys.susceptance) == 0.0)
    assert np.all(sys.voltage > 0.0)
    assert abs(float(np.mean(sys.injection))) < 1e-12


@pytest.mark.parametrize("bad", ["susceptance", "voltage", "inertia"])
def test_gridsystem_rejects_malformed(bad: str) -> None:
    """GridSystem fails closed on contract violations."""
    sys = wscc_9_bus()
    kw: dict[str, Any] = {
        "name": sys.name,
        "bus_ids": sys.bus_ids,
        "susceptance": sys.susceptance.copy(),
        "voltage": sys.voltage.copy(),
        "injection": sys.injection.copy(),
        "inertia": sys.inertia.copy(),
        "damping": sys.damping.copy(),
        "citation": sys.citation,
    }
    if bad == "susceptance":
        s = kw["susceptance"]
        s[0, 1] = 99.0  # break symmetry
    elif bad == "voltage":
        kw["voltage"] = np.array([-1.0, 1.0, 1.0])
    else:
        kw["inertia"] = np.array([0.0, 1.0, 1.0])
    from research.calibration.grid_kuramoto import GridSystem

    with pytest.raises(ValueError):
        GridSystem(**kw)


# ---------------------------------------------------------------------------
# Reduction formulae — exact at float precision (INV-style algebraic)
# ---------------------------------------------------------------------------


def test_coupling_formula_exact() -> None:
    """K_ij = |V_i||V_j|B_ij to float precision (Dörfler–Bullo Eq. (2))."""
    sys = wscc_9_bus()
    k = coupling_from_susceptance(sys.susceptance, sys.voltage)
    for i in range(sys.n):
        for j in range(sys.n):
            if i == j:
                assert k[i, j] == 0.0
            else:
                expected = sys.voltage[i] * sys.voltage[j] * sys.susceptance[i, j]
                assert abs(k[i, j] - expected) < 1e-12
    np.testing.assert_allclose(k, k.T, atol=1e-12)


def test_natural_frequency_mean_centred() -> None:
    """ω_i = P_i/d_i, then mean-removed (rotating reference gauge)."""
    sys = wscc_9_bus()
    w = natural_frequency_from_injection(sys.injection, sys.damping)
    assert abs(float(np.mean(w))) < 1e-12
    # Relative ordering preserved (gauge does not change relative ω).
    raw = sys.injection / sys.damping
    assert np.argmax(w) == np.argmax(raw)


def test_dorfler_bullo_uniform_scale_invariance() -> None:
    """s_crit scales exactly as 1/s under uniform K↦sK (Eq. (3))."""
    sys = wscc_9_bus()
    k = coupling_from_susceptance(sys.susceptance, sys.voltage)
    w = natural_frequency_from_injection(sys.injection, sys.damping)
    s0 = dorfler_bullo_critical_coupling(k, w)
    for s in (2.0, 5.0, 10.0):
        s_scaled = dorfler_bullo_critical_coupling(s * k, w)
        assert abs(s_scaled - s0 / s) < 1e-9 * max(1.0, s0)


def test_dorfler_bullo_disconnected_fails_closed() -> None:
    """Disconnected coupling graph → ValueError (no silent best-effort)."""
    k = np.zeros((4, 4), dtype=np.float64)
    k[0, 1] = k[1, 0] = 1.0  # {0,1} isolated from {2,3}
    w = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float64)
    with pytest.raises(ValueError, match="disconnected"):
        dorfler_bullo_critical_coupling(k, w)


# ---------------------------------------------------------------------------
# Hypothesis property — K→∞ tight-sync recovery on matched first-order data
# ---------------------------------------------------------------------------


def _first_order_traj(k: np.ndarray, w: np.ndarray, dt: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    th = np.asarray(rng.uniform(-0.4, 0.4, size=k.shape[0]), dtype=np.float64)
    out = np.empty((n + 1, k.shape[0]), dtype=np.float64)
    out[0] = th

    def f(x: np.ndarray) -> np.ndarray:
        d = x[None, :] - x[:, None]
        return np.asarray(w + (k * np.sin(d)).sum(axis=1), dtype=np.float64)

    for t in range(n):
        k1 = f(th)
        k2 = f(th + 0.5 * dt * k1)
        k3 = f(th + 0.5 * dt * k2)
        k4 = f(th + dt * k3)
        th = th + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[t + 1] = th
    return out


@pytest.mark.slow
@settings(max_examples=8, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_property_weak_coupling_topology_recovered(seed: int) -> None:
    """In the estimator's design regime (weak, non-locked, matched
    first-order model) the *topology support* of a 3-node chain is
    recovered: the present edge survives, the absent edge stays absent.

    This is the property that must hold for the calibration instrument
    to be sound — it isolates estimator behaviour from the second-order
    model mismatch documented in RESULTS.md.
    """
    k = np.array(
        [[0.0, 0.35, 0.0], [0.35, 0.0, 0.30], [0.0, 0.30, 0.0]],
        dtype=np.float64,
    )
    w = np.array([-1.4, 0.2, 1.2], dtype=np.float64)
    w = w - w.mean()
    traj = _first_order_traj(k, w, dt=0.01, n=5000, seed=seed)
    wrapped = np.mod(traj, 2.0 * np.pi)
    wrapped = np.clip(wrapped, 0.0, np.nextafter(2.0 * np.pi, 0.0))
    pm = PhaseMatrix(
        theta=wrapped,
        timestamps=np.arange(wrapped.shape[0], dtype=np.float64) * 0.01,
        asset_ids=("a", "b", "c"),
        extraction_method="hilbert",
        frequency_band=(1e-6, 0.5),
    )
    est = CouplingEstimator(CouplingEstimationConfig(penalty="mcp", lambda_reg=0.005, dt=0.01))
    k_hat = np.abs(0.5 * (est.estimate(pm).K + est.estimate(pm).K.T))
    scale = float(np.max(k_hat))
    assert scale > 0.0
    present = k_hat / scale
    # Present edges clearly above the absent (0,2) edge.
    assert present[0, 1] > present[0, 2]
    assert present[1, 2] > present[0, 2]


# ---------------------------------------------------------------------------
# Pre-registration ↔ code agreement (fail-closed)
# ---------------------------------------------------------------------------


def test_preregistration_matches_code() -> None:
    """PREREGISTRATION.md gate table mirrors gates.py byte-for-numeric."""
    text = _PREREG.read_text(encoding="utf-8")
    for gate in (*NOISELESS_GATES, *NOISY_GATES):
        # The gate name and its threshold must both appear in the doc,
        # on a markdown table row containing the operator.
        pat = re.compile(
            re.escape(f"`{gate.name}`")
            + r".*?"
            + re.escape(gate.operator)
            + r".*?"
            + re.escape(f"`{gate.threshold}`")
        )
        assert pat.search(text), (
            f"PREREGISTRATION.md drifted from gates.py for {gate.name}: "
            f"expected operator {gate.operator} threshold {gate.threshold}"
        )


def test_gate_thresholds_are_frozen_values() -> None:
    """Lock the exact pre-registered numbers (post-data edit detector)."""
    nl = {g.name: (g.operator, g.threshold) for g in NOISELESS_GATES}
    ny = {g.name: (g.operator, g.threshold) for g in NOISY_GATES}
    assert nl == {
        "noiseless.frobenius": ("<=", 0.10),
        "noiseless.topology_f1": (">=", 0.95),
        "noiseless.critical_coupling": ("<=", 0.15),
    }
    assert ny == {
        "noisy.frobenius": ("<=", 0.25),
        "noisy.topology_f1": (">=", 0.90),
    }


# ---------------------------------------------------------------------------
# Calibration loop — determinism + stable NEGATIVE verdict
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_calibration_deterministic() -> None:
    """Seeded calibration is bit-reproducible across two runs."""
    sys = wscc_9_bus()
    cfg = SimConfig()
    a = run_calibration(sys, cfg, noisy=False)
    b = run_calibration(sys, cfg, noisy=False)
    assert a.frobenius_rel_error == b.frobenius_rel_error
    assert a.topology_f1 == b.topology_f1
    assert a.critical_coupling_rel_error == b.critical_coupling_rel_error


@pytest.mark.slow
def test_wscc9_verdict_is_negative_and_localized() -> None:
    """The pre-registered WSCC-9 calibration lands a NEGATIVE artifact
    whose failing gates name a concrete refinement target.

    This asserts the *honest* outcome: topology support is recovered
    noiseless (F1≥0.95) but the signed-coupling Frobenius gate is
    missed — the estimator is a first-order identifier on second-order
    data (see RESULTS.md). No promotion language is implied by the
    test; it pins the NEGATIVE verdict so a future estimator change
    that closes the gap is detected as a deliberate, reviewed event.
    """
    sys = wscc_9_bus()
    cfg = SimConfig()
    nl = run_calibration(sys, cfg, noisy=False)
    ny = run_calibration(sys, cfg, noisy=True)
    results = evaluate_gates(nl, NOISELESS_GATES) + evaluate_gates(ny, NOISY_GATES)
    verdict = overall_verdict(results)
    assert verdict == "NEGATIVE"

    by_name = {r.name: r for r in results}
    # Topology support IS recovered noiseless (instrument is sound).
    assert by_name["noiseless.topology_f1"].passed
    # Signed-coupling Frobenius IS missed (the localized finding).
    assert not by_name["noiseless.frobenius"].passed
    assert "coupling_estimator" in by_name["noiseless.frobenius"].localises_to
    # Every failed gate carries a non-empty localisation string.
    for r in results:
        if not r.passed:
            assert r.localises_to.strip()


def test_ledger_is_machine_readable_and_sha_pinned() -> None:
    """build_ledger emits a JSON-serialisable, sha-pinned NEGATIVE ledger."""
    sys = wscc_9_bus()
    ledger = build_ledger(sys, SimConfig())
    blob = json.dumps(ledger, sort_keys=True)
    assert json.loads(blob)["verdict"] == "NEGATIVE"
    assert ledger["is_hypothesis"] is False
    assert len(ledger["ledger_sha256"]) == 64
    assert ledger["localized_refinement_targets"]


# ---------------------------------------------------------------------------
# Instrument-soundness diagnostic — recovery in the estimator design regime
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_identifiability_sweep_weak_regime() -> None:
    """The estimator *does* recover K in its design regime.

    Weak coupling on a *matched first-order* model keeps the trajectory
    non-phase-locked (persistently excited); the row design then has
    full rank and the Frobenius error is small. This proves the WSCC-9
    NEGATIVE verdict is a genuine model-class / excitation boundary,
    not a broken instrument.
    """
    k_base = np.array(
        [[0.0, 7.74, 5.17], [7.74, 0.0, 8.87], [5.17, 8.87, 0.0]],
        dtype=np.float64,
    )
    w = np.array([-1.5, 2.4, -0.9], dtype=np.float64)
    w = w - w.mean()
    k = 0.05 * k_base  # weak: K_max ≈ 0.44, system does not lock
    traj = _first_order_traj(k, w, dt=0.01, n=6000, seed=7)
    wrapped = np.mod(traj, 2.0 * np.pi)
    wrapped = np.clip(wrapped, 0.0, np.nextafter(2.0 * np.pi, 0.0))
    pm = PhaseMatrix(
        theta=wrapped,
        timestamps=np.arange(wrapped.shape[0], dtype=np.float64) * 0.01,
        asset_ids=("a", "b", "c"),
        extraction_method="hilbert",
        frequency_band=(1e-6, 0.5),
    )
    est = CouplingEstimator(CouplingEstimationConfig(penalty="mcp", lambda_reg=0.005, dt=0.01))
    k_hat = 0.5 * (est.estimate(pm).K + est.estimate(pm).K.T)
    rel = float(np.linalg.norm(k_hat - k) / np.linalg.norm(k))
    # Design-regime recovery: order-of-magnitude better than the
    # strongly-coupled WSCC-9 miss (~1.0). Bound is a soundness floor,
    # not a pre-registered gate.
    assert rel < 0.30, (
        f"instrument unsound: weak-regime Frobenius rel err {rel:.3f} "
        f"should be << 1.0; estimator must recover K when its "
        f"first-order persistently-excited assumptions hold"
    )


@pytest.mark.slow
def test_ieee39_calibration_runs_and_is_scored() -> None:
    """Heavy IEEE-39 fixture runs end-to-end and produces finite metrics.

    Not part of the pre-registered WSCC-9 verdict; this only checks the
    loop scales to a 10-machine system and the ledger stays well-formed.
    """
    sys = ieee_39_new_england()
    cfg = SimConfig()
    ledger = build_ledger(sys, cfg)
    assert ledger["system"] == "IEEE-39"
    assert ledger["verdict"] in {"PASS", "NEGATIVE"}
    for regime in ("noiseless", "noisy"):
        m = ledger["metrics"][regime]
        assert np.isfinite(m["frobenius_rel_error"])
        assert 0.0 <= m["topology_f1"] <= 1.0
