# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Architectural forcing function for the swing coupling-estimator stack.

``core.kuramoto.coupling_estimator`` was an open-loop universal sink:
every CALIB-GRID lineage bolted another symmetric-joint estimation path
(``estimate_swing_coupling`` R1, then ``estimate_swing_coupling_integral``
CALIB-GRID-002) onto the module with its global-design assembly inlined
into the public function — 521 → 1322 LOC, +154 % over five lineages,
monotonic, with no strategy/boundary capping growth. PR #759 extracted
the shared solve tail (``_solve_symmetric_joint``); this refactor
extracted the path-specific *design assembly* into registered
:class:`SwingDesignStrategy` objects and turned the public functions
into thin dispatchers.

This module is the missing module-scale negative-feedback term (it
mirrors the lineage-scale forcing functions added in #762). It is
strictly pure-additive — no production formula, no frozen artifact, no
gate/threshold/seed and no pre-existing test is touched. Each test fails
closed the moment a future symmetric-joint estimation path is added
*outside* the registry or perturbs a path's numerics:

* ``test_every_symmetric_joint_path_dispatches_through_registry`` — the
  set of public symmetric-joint swing entry points equals the set of
  registered strategies routed by the single dispatcher. A lineage that
  inlines a third ``estimate_swing_coupling_*`` symmetric-joint design
  instead of registering a strategy fails this test (the structural cap
  on the monotonic accretion).
* ``test_dispatcher_core_is_path_independent`` — the one dispatcher
  ``_dispatch_swing`` mentions no path-specific symbol; new paths cannot
  be wired by editing it.
* ``test_strategy_registry_is_frozen_and_fail_closed`` — strategies are
  frozen, the registry rejects a duplicate key (no silent shadowing of
  an audited path), and the public view is read-only.
* ``test_golden_vectors_bit_identical`` — every estimation path's
  ``K̂`` / ``ω̂`` / ``P̂`` / identifiability score / PE diagnostic and
  every prox operator are byte-for-byte the values captured on the
  parent sha (65666072), proving the extraction is algorithm-preserving.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path

import numpy as np

from core.kuramoto.contracts import PhaseMatrix
from core.kuramoto.coupling_estimator import (
    CouplingEstimationConfig,
    CouplingEstimator,
    SwingDesign,
    SwingDesignStrategy,
    complementary_pairs_stability,
    estimate_coupling,
    estimate_swing_coupling,
    estimate_swing_coupling_integral,
    mcp_prox,
    scad_prox,
    soft_threshold,
    swing_strategy_registry,
)

_MODULE = Path(__file__).resolve().parents[3] / "core" / "kuramoto" / "coupling_estimator.py"


# ---------------------------------------------------------------------------
# Structural cap: every symmetric-joint path goes through the registry
# ---------------------------------------------------------------------------


def test_every_symmetric_joint_path_dispatches_through_registry() -> None:
    """Symmetric-joint estimation paths == registered strategies.

    The two public symmetric-joint swing entry points (the differential
    ``estimate_swing_coupling`` with ``symmetric=True`` and the
    weak/integral ``estimate_swing_coupling_integral``) each route
    through ``_dispatch_swing`` with a registry key, and the registry
    holds exactly those keys. A future lineage that adds a third
    symmetric-joint design inline — instead of registering a strategy —
    breaks this equality and fails closed.
    """
    registry = swing_strategy_registry()
    assert set(registry) == {"differential_symmetric", "integral_weak_form"}, (
        "registered swing strategy key set drifted — a new symmetric-joint "
        "estimation path must register a SwingDesignStrategy, not inline its "
        "design assembly into the dispatcher/public function"
    )

    # encoding pinned: coupling_estimator.py carries β/θ/λ/κ glyphs; a
    # locale-default read crashes under an ASCII CI locale (UnicodeDecodeError).
    tree = ast.parse(_MODULE.read_text(encoding="utf-8"))
    dispatched_keys: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_dispatch_swing"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            dispatched_keys.add(node.args[0].value)
    assert dispatched_keys == set(registry), (
        f"dispatched keys {sorted(dispatched_keys)} != registered "
        f"{sorted(registry)} — a symmetric-joint path bypassed the registry "
        "or a registered strategy is unreachable (orphaned accretion)"
    )

    # Every registered strategy satisfies the frozen contract.
    for key, strat in registry.items():
        assert isinstance(strat, SwingDesignStrategy)
        assert strat.key == key


def test_dispatcher_core_is_path_independent() -> None:
    """The single dispatcher mentions no path-specific estimator symbol.

    ``_dispatch_swing`` selects a strategy by key and runs the shared
    tail; it must not name a Savitzky–Golay / weak-form / stencil symbol
    (that would mean a path was wired by editing the dispatcher rather
    than by registering a strategy).
    """
    import core.kuramoto.coupling_estimator as ce

    body = inspect.getsource(ce._dispatch_swing)
    forbidden = ("savgol", "_test_function_stencil", "bump_order", "trapezoid", "unwrap")
    leaked = [tok for tok in forbidden if tok in body]
    assert not leaked, (
        f"_dispatch_swing leaked path-specific symbols {leaked} — the "
        "dispatcher core must stay path-independent so new paths require "
        "strategy registration, not a dispatcher edit"
    )


def test_strategy_registry_is_frozen_and_fail_closed() -> None:
    """Strategies are frozen, the view is read-only, dup keys rejected."""
    import core.kuramoto.coupling_estimator as ce

    registry = swing_strategy_registry()
    strat = next(iter(registry.values()))
    # Frozen dataclass: attribute assignment is blocked.
    try:
        strat.key = "mutated"  # type: ignore[misc]
    except AttributeError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("swing strategy must be frozen (no attr set)")
    # Read-only public view.
    try:
        registry["x"] = strat  # type: ignore[index]
    except TypeError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("swing_strategy_registry() view must be read-only")
    # Duplicate-key registration is fail-closed (no silent shadow).
    existing = next(iter(registry.values()))
    try:
        ce._register_swing_strategy(existing)
    except ValueError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("duplicate strategy key must fail closed")

    # SwingDesign is a frozen contract carrying only the assembly output.
    sd = SwingDesign(
        design=np.zeros((2, 2), dtype=np.float64),
        target=np.zeros(2, dtype=np.float64),
        edges=[(0, 1)],
        n_edge=1,
    )
    try:
        sd.n_edge = 2  # type: ignore[misc]
    except AttributeError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("SwingDesign must be frozen")


# ---------------------------------------------------------------------------
# Algorithm-preserving extraction: golden vectors pinned to parent sha
# ---------------------------------------------------------------------------


def _h(a: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _first_order_pm() -> tuple[PhaseMatrix, np.ndarray]:
    rng = np.random.default_rng(42)
    n, t_len, dt = 6, 1500, 0.05
    k_true = np.zeros((n, n))
    k_true[0, 1] = 1.8
    k_true[1, 0] = 1.5
    k_true[2, 3] = -1.2
    k_true[4, 5] = 2.0
    omega = rng.uniform(0.3, 0.7, n)
    theta = np.zeros((t_len, n))
    theta[0] = rng.uniform(0.0, 2 * np.pi, n)
    for t in range(1, t_len):
        di = theta[t - 1, None, :] - theta[t - 1, :, None]
        theta[t] = (
            theta[t - 1]
            + dt * (omega + np.sum(k_true * np.sin(di), axis=1))
            + 0.05 * rng.standard_normal(n) * np.sqrt(dt)
        )
    theta = np.mod(theta, 2 * np.pi)
    pm = PhaseMatrix(
        theta=theta,
        timestamps=np.arange(t_len, dtype=np.float64) * dt,
        asset_ids=tuple(f"x{i}" for i in range(n)),
        extraction_method="hilbert",
        frequency_band=(0.01, 1.0),
    )
    return pm, theta


def _swing_pm() -> tuple[PhaseMatrix, np.ndarray, np.ndarray]:
    k = np.array([[0.0, 0.9, 0.5], [0.9, 0.0, 0.7], [0.5, 0.7, 0.0]])
    p = np.array([0.6, -0.1, -0.5]) - np.mean([0.6, -0.1, -0.5])
    m = np.array([0.4, 0.5, 0.3])
    d = np.array([0.25, 0.3, 0.2])
    rng = np.random.default_rng(11)
    th0 = rng.uniform(-0.8, 0.8, 3)
    th0 -= th0.mean()
    nn, dt, n = 3, 0.005, 4000
    thr = th0.astype(np.float64).copy()
    v = np.zeros(nn)
    traj = np.empty((n + 1, nn))
    traj[0] = thr

    def accel(x: np.ndarray, w: np.ndarray) -> np.ndarray:
        di = x[:, None] - x[None, :]
        return np.asarray((p - (k * np.sin(di)).sum(1) - d * w) / m, dtype=np.float64)

    for t in range(n):
        a1 = accel(thr, v)
        a2 = accel(thr + 0.5 * dt * v, v + 0.5 * dt * a1)
        a3 = accel(thr + 0.5 * dt * (v + 0.5 * dt * a1), v + 0.5 * dt * a2)
        a4 = accel(thr + dt * (v + 0.5 * dt * a2), v + dt * a3)
        thr = thr + dt * v + (dt * dt / 6.0) * (a1 + 2 * a2 + 2 * a3)
        v = v + (dt / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
        traj[t + 1] = thr
    w = np.mod(traj, 2 * np.pi)
    w = np.clip(w, 0.0, np.nextafter(2 * np.pi, 0.0))
    pm = PhaseMatrix(
        theta=w,
        timestamps=np.arange(w.shape[0], dtype=np.float64) * dt,
        asset_ids=("a", "b", "c"),
        extraction_method="hilbert",
        frequency_band=(1e-6, 0.5),
    )
    return pm, m, d


# Golden vectors captured on the parent sha 65666072 (pre-refactor) —
# every estimation path's raw-byte hash. A single ULP drift on any path
# flips one of these and fails the test (algorithm-preserving contract).
_GOLDEN: dict[str, object] = {
    # audited: numpy-array content sha256 / scalar diagnostics, NOT
    # credentials — the bit-identical golden vectors from sha 65666072.
    "first_order_mcp_K": "2ef43b77c3cb4e24b567563caddebd0a31c01155a72a283bb0b048a66266cdff",  # pragma: allowlist secret
    "first_order_mcp_sparsity": 0.9333333333333333,
    "first_order_stab_K": "4a1693424442e9ef473e2f0d6a4699a885a8316d54a284412cae70e0fdcd858e",  # pragma: allowlist secret
    "first_order_stab_scores": "f4c4a8f934857770e8649f2ee958d2af3a63a783ea4038e41108a423053d4297",  # pragma: allowlist secret
    "cps_K_median": "bd3e05110466b148c0eae3b58928949f54207036b9a30369f66cfb2cb0c66148",  # pragma: allowlist secret
    "cps_stability": "f4c4a8f934857770e8649f2ee958d2af3a63a783ea4038e41108a423053d4297",  # pragma: allowlist secret
    "mcp_prox": "0509a9a0bed614b2eb21a44091006cc979db732f34076fe63af2aaa0d992226f",  # pragma: allowlist secret
    "scad_prox": "9dff064a01a1963ac4ee4b0f71ee300deb65f81835151df3419e3043f10f935f",  # pragma: allowlist secret
    "soft_threshold": "93f77c0d636d04dd2cd1683ec9936f95bcdeec12e3526e4c2e5b66861f3fa7c6",  # pragma: allowlist secret
    "swing_sym_K": "f2f63bc4f07cedc6044017d641fb93e264f98e9b7959daf38b97a021a6e1ca76",  # pragma: allowlist secret
    "swing_sym_P": "3b9e7840e3f6056bbc94e043a9ca8fbe94132a83af040168151c73c0edbc7e87",  # pragma: allowlist secret
    "swing_sym_omega": "749e28c849ccc30f5858bd7b6f1087e24262ab450bd6f92b06bc6aeaf8800797",  # pragma: allowlist secret
    "swing_sym_msr": 0.04004743831202507,
    "swing_sym_ident_K": "f2f63bc4f07cedc6044017d641fb93e264f98e9b7959daf38b97a021a6e1ca76",  # pragma: allowlist secret
    "swing_sym_ident_score": 0.9996845317668399,
    "swing_sym_ident_verdict": "ACCEPT",
    "swing_asym_K": "cd0021bea77cada8ed6c21231b239a042783ee4d91394921d57cbca8fb765879",  # pragma: allowlist secret
    "swing_asym_P": "92ecaf44292090862a4732bf4fa71fd8cd8d29ef9e75c7c125c8d96cf5c9e55f",  # pragma: allowlist secret
    "swing_asym_omega": "759d69c0bd363d78abedf78b4fede1cd9bc9ca41605644822e7ff0566d853bec",  # pragma: allowlist secret
    "swing_asym_msr": 0.03644135053721742,
    "integral_K": "b9dc00b1f213add6039d55b1d2e25c197a9c8d6b280f315d8f693ebae80b2304",  # pragma: allowlist secret
    "integral_P": "d22c73572803701d064edbd6678577800320fbabf1522f74c5b3bcd1bce6f30c",  # pragma: allowlist secret
    "integral_omega": "2936f444e03019d5e58528559908ba4a60954a79a426f611c0522f38ddfb1490",  # pragma: allowlist secret
    "integral_msr": 0.037442981977247894,
    "integral_ident_score": 0.9981478637919239,
    "integral_ident_verdict": "ACCEPT",
}


def test_golden_vectors_bit_identical() -> None:
    """Every estimation path is byte-identical to the parent sha output.

    This is the bit-identical proof for the strategy-registry
    extraction: the dispatcher + strategies reproduce the exact ``K̂`` /
    ``ω̂`` / ``P̂`` / identifiability score / PE diagnostic / prox-operator
    bytes captured on 65666072 before any structural change.
    """
    out: dict[str, object] = {}

    pm, theta = _first_order_pm()
    dt = 0.05
    cfg = CouplingEstimationConfig(penalty="mcp", lambda_reg=0.15, dt=dt, max_iter=800, tol=1e-7)
    r1 = estimate_coupling(pm, cfg)
    out["first_order_mcp_K"] = _h(r1.K)
    out["first_order_mcp_sparsity"] = float(r1.sparsity)

    cfg_s = CouplingEstimationConfig(
        penalty="mcp",
        lambda_reg=0.02,
        dt=dt,
        max_iter=300,
        tol=1e-5,
        stability_selection=True,
        lambda_grid=(0.01, 0.03, 0.1),
        n_subsamples=4,
        subsample_fraction=0.5,
        stability_threshold=0.5,
        random_state=0,
    )
    rs = CouplingEstimator(cfg_s).estimate(pm)
    assert rs.stability_scores is not None
    out["first_order_stab_K"] = _h(rs.K)
    out["first_order_stab_scores"] = _h(rs.stability_scores)
    k_med, stab = complementary_pairs_stability(theta, cfg_s)
    out["cps_K_median"] = _h(k_med)
    out["cps_stability"] = _h(stab)

    z = np.array([-2.0, -0.3, 0.0, 0.3, 1.5, 4.0])
    out["mcp_prox"] = _h(mcp_prox(z, 1.0, 3.0, 0.5))
    out["scad_prox"] = _h(scad_prox(z, 1.0, 3.7, 0.5))
    out["soft_threshold"] = _h(soft_threshold(z, 0.2))

    pm2, m, d = _swing_pm()
    dt2 = 0.005
    sd = estimate_swing_coupling(pm2, m, d, dt=dt2, savgol_window=7, savgol_polyorder=4)
    out["swing_sym_K"] = _h(sd.K)
    out["swing_sym_P"] = _h(sd.injection)
    out["swing_sym_omega"] = _h(sd.omega)
    out["swing_sym_msr"] = float(sd.min_singular_ratio)
    sd_i = estimate_swing_coupling(
        pm2, m, d, dt=dt2, savgol_window=7, savgol_polyorder=4, identifiability_gate=True
    )
    assert sd_i.identifiability is not None
    out["swing_sym_ident_K"] = _h(sd_i.K)
    out["swing_sym_ident_score"] = float(sd_i.identifiability.score)
    out["swing_sym_ident_verdict"] = sd_i.identifiability.verdict
    sa = estimate_swing_coupling(
        pm2, m, d, dt=dt2, symmetric=False, savgol_window=7, savgol_polyorder=4, pe_guard=False
    )
    out["swing_asym_K"] = _h(sa.K)
    out["swing_asym_P"] = _h(sa.injection)
    out["swing_asym_omega"] = _h(sa.omega)
    out["swing_asym_msr"] = float(sa.min_singular_ratio)
    ig = estimate_swing_coupling_integral(
        pm2, m, d, dt=dt2, test_support=120, n_windows=120, bump_order=6
    )
    out["integral_K"] = _h(ig.K)
    out["integral_P"] = _h(ig.injection)
    out["integral_omega"] = _h(ig.omega)
    out["integral_msr"] = float(ig.min_singular_ratio)
    ig_i = estimate_swing_coupling_integral(
        pm2,
        m,
        d,
        dt=dt2,
        test_support=120,
        n_windows=120,
        bump_order=6,
        identifiability_gate=True,
    )
    assert ig_i.identifiability is not None
    out["integral_ident_score"] = float(ig_i.identifiability.score)
    out["integral_ident_verdict"] = ig_i.identifiability.verdict

    mism = [k for k in _GOLDEN if _GOLDEN[k] != out.get(k)]
    assert not mism, (
        f"NON-BIT-IDENTICAL extraction — paths drifted from the parent-sha "
        f"golden vectors: {mism}. The strategy-registry refactor is "
        "algorithm-preserving by contract; any numeric change here is a "
        "behavior breach, not a refactor."
    )
