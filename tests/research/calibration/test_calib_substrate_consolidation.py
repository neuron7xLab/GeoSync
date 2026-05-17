# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Regression rails for the CALIB-GRID structural consolidation.

These tests pin the *byte-preservation* contract of the
behavior-preserving refactor that introduced
``research.calibration.grid_kuramoto._substrate``:

* every one of the four lineage ledgers (CALIB-GRID-001, R1,
  CALIB-GRID-002, identifiability front-gate) must reproduce its
  **provenance-stripped structural sha256** captured from ``origin/main``
  *before* the refactor — i.e. the consolidation changed no emitted
  byte of any sha-pinned artifact;
* both swing estimator classes must remain **bit-identical** through
  the extracted shared ``_solve_symmetric_joint`` back end;
* the unified ``_substrate`` primitives must be the single source (no
  surviving duplicate gate dataclass / sha-pinning copy / ``_branch_sha``
  / ``_topology_f1``).

If any of these fail the refactor stopped being behavior-preserving and
must be reverted, not accommodated.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any

import numpy as np
import pytest

from core.kuramoto.coupling_estimator import (
    estimate_swing_coupling,
    estimate_swing_coupling_integral,
)
from research.calibration.grid_kuramoto import SimConfig, wscc_9_bus
from research.calibration.grid_kuramoto.calibration import ground_truth, simulate_phases
from research.calibration.grid_kuramoto.cg002 import build_cg002_ledger
from research.calibration.grid_kuramoto.identifiability.validate import (
    build_identifiability_ledger,
)
from research.calibration.grid_kuramoto.run import build_ledger, build_r1_ledger

# Provenance-stripped structural sha256 of each lineage ledger, captured
# from origin/main @ e71d1915 BEFORE the consolidation. ``branch_sha``
# and ``ledger_sha256`` are environment / commit provenance and are
# excluded (exactly as the pre-existing _deep_close artifact tests do);
# every other byte of every ledger must be reproduced unchanged.
_GOLDEN_STRUCT_SHA: dict[str, str] = {
    # audited: deterministic ledger content hashes, not credentials
    "base": "ba10ed1392f62a146e1040dc782c1097f65f17495b61f81c810d77048b751b1e",  # pragma: allowlist secret
    "r1": "8315eb0a21bb411dd97b0d96134a9c3ba25c3eb7ca2db45cd121bc92ea05f8b4",  # pragma: allowlist secret
    "cg002": "d0f89e24341b099598e2e5cc9809772ee2c47627f77bf3354f957fab860819b1",  # pragma: allowlist secret
    "ident": "b3f8afa120f704e320c6b0144d3335680f6edae4347bccc27db264e66e018648",  # pragma: allowlist secret
}


def _struct_sha(ledger: dict[str, Any]) -> str:
    """sha256 of the ledger with the two provenance fields removed."""
    stripped = dict(ledger)
    stripped.pop("branch_sha", None)
    stripped.pop("ledger_sha256", None)
    payload = json.dumps(stripped, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(_GOLDEN_STRUCT_SHA))
def test_consolidated_ledger_is_byte_identical_to_pre_refactor(name: str) -> None:
    """Each lineage ledger reproduces its pre-refactor structural sha256.

    This is the hard behavior-preserving invariant: the shared
    ``_substrate`` ledger / gate / sha-pinning back end must emit the
    *exact same bytes* the four hand-rolled copies did.
    """
    sys_, cfg = wscc_9_bus(), SimConfig()
    builders = {
        "base": build_ledger,
        "r1": build_r1_ledger,
        "cg002": build_cg002_ledger,
        "ident": build_identifiability_ledger,
    }
    ledger = builders[name](sys_, cfg)
    got = _struct_sha(ledger)
    assert got == _GOLDEN_STRUCT_SHA[name], (
        f"CALIB-GRID consolidation drifted the {name!r} ledger: "
        f"structural sha256 {got} != frozen {_GOLDEN_STRUCT_SHA[name]}. "
        f"The refactor must be byte-preserving — revert, do not retune."
    )
    # The sha-pinning field itself must still be a well-formed 64-hex.
    assert len(ledger["ledger_sha256"]) == 64
    assert int(ledger["ledger_sha256"], 16) >= 0


@pytest.mark.slow
def test_both_swing_classes_bit_identical_through_shared_back_end() -> None:
    """K̂ / P̂ / verdict are bit-identical through ``_solve_symmetric_joint``.

    The extracted shared symmetric-joint back end is structure-only; the
    differential and integral estimator classes must produce the exact
    same point estimate on the frozen WSCC-9 case as before the
    extraction (pinned here as content hashes of the raw float64 bytes).
    """
    sys_, cfg = wscc_9_bus(), SimConfig()
    k_true, omega_true = ground_truth(sys_, cfg.coupling_scale)
    phases, _ = simulate_phases(sys_, k_true, omega_true, cfg)
    m = np.asarray(sys_.inertia, dtype=np.float64)
    d = np.asarray(sys_.damping, dtype=np.float64)

    diff = estimate_swing_coupling(
        phases,
        m,
        d,
        dt=cfg.dt,
        symmetric=True,
        savgol_window=7,
        savgol_polyorder=4,
        pe_guard=True,
        identifiability_gate=True,
    )
    intg = estimate_swing_coupling_integral(
        phases,
        m,
        d,
        dt=cfg.dt,
        test_support=120,
        n_windows=400,
        bump_order=6,
        pe_guard=True,
        identifiability_gate=True,
    )

    # Frozen raw-byte hashes captured pre-refactor (origin/main).
    # audited: deterministic float64 content hashes, not credentials
    exp_diff_k = "9c1232742154b0bf"  # pragma: allowlist secret
    exp_diff_p = "ae84ba5f46cf805f"  # pragma: allowlist secret
    exp_intg_k = "8af3560026ed4934"  # pragma: allowlist secret
    exp_intg_p = "c60a9639b789a42a"  # pragma: allowlist secret
    assert hashlib.sha256(diff.K.tobytes()).hexdigest()[:16] == exp_diff_k
    assert hashlib.sha256(diff.injection.tobytes()).hexdigest()[:16] == exp_diff_p
    assert hashlib.sha256(intg.K.tobytes()).hexdigest()[:16] == exp_intg_k
    assert hashlib.sha256(intg.injection.tobytes()).hexdigest()[:16] == exp_intg_p
    assert diff.identifiability is not None and diff.identifiability.verdict.value == "REFUSE"
    assert intg.identifiability is not None and intg.identifiability.verdict.value == "REFUSE"
    # Symmetric solve ⇒ exactly symmetric K with zero diagonal.
    np.testing.assert_array_equal(diff.K, diff.K.T)
    np.testing.assert_array_equal(intg.K, intg.K.T)
    assert np.all(np.diag(diff.K) == 0.0)
    assert np.all(np.diag(intg.K) == 0.0)


def test_substrate_is_single_source_no_surviving_duplicates() -> None:
    """No lineage module re-defines a primitive the substrate now owns.

    Guards against a future lineage re-introducing the fractal copies:
    the gate dataclass, the ``json.dumps(sort_keys)->sha256`` pinning,
    ``_branch_sha`` and ``_topology_f1`` must each have exactly one
    definition (in ``_substrate``); the lineage modules may only import
    or thinly wrap them.
    """
    from research.calibration.grid_kuramoto import _substrate, calibration, gates, run
    from research.calibration.grid_kuramoto.cg002 import cg002
    from research.calibration.grid_kuramoto.identifiability import validate

    # The sha-pinning literal must not be re-implemented inline anywhere.
    for mod in (run, validate, cg002):
        src = inspect.getsource(mod)
        assert "hashlib.sha256(payload).hexdigest()" not in src, (
            f"{mod.__name__} re-implements the ledger sha-pinning inline; "
            f"it must call _substrate.ledger_sha256"
        )

    # Exactly one Gate dataclass / topology_f1 / branch_sha definition.
    assert hasattr(_substrate, "Gate") and hasattr(_substrate, "GateRow")
    assert "class GateVerdict" not in inspect.getsource(gates)
    assert "class CG002Gate" not in inspect.getsource(cg002)
    assert "def _topology_f1" not in inspect.getsource(calibration)
    assert "def _topology_f1" not in inspect.getsource(cg002)
    # gates.py aliases the substrate types (legacy import surface kept).
    assert gates.GateVerdict is _substrate.Gate
    assert gates.GateResult is _substrate.GateRow


def test_frozen_provenance_constants_are_single_valued() -> None:
    """The audited parent content-hash literals exist once, in the substrate.

    They were copied across three modules; centralising them must not
    change their value (any change would drift every child ledger that
    cites the parent lineage).
    """
    from research.calibration.grid_kuramoto import _substrate

    assert (  # pragma: allowlist secret  (audited parent prereg git sha)
        _substrate.FROZEN_PREREG_SHA
        == "d170d48afa5066c13edeb40b2c1904b3fd708516"  # pragma: allowlist secret
    )
    assert (  # audited parent calibration ledger content hash
        _substrate.PARENT_LEDGER_SHA256
        == "ed8d409b7b222eb053572d6bf9ab6e98c5f4918be1cae384864733a2b4d72aaf"  # pragma: allowlist secret
    )
    assert (  # audited CALIB-GRID-002 prereg branch base sha
        _substrate.PREREG_BRANCH_BASE_SHA
        == "a5e0d533b2201c999b31c792773e858f8da713bf"  # pragma: allowlist secret
    )
