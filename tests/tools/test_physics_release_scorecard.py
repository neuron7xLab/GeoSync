# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Tests for the physics release scorecard validator and readiness gate.

Proves: (a) the live scorecard is schema-valid and anti-theater-consistent;
(b) the readiness gate fails closed while any required dimension is below
CI_VERIFIED — the repository is honestly NOT physics-ready today; (c) a verified
state cannot float free of an existing binding artifact; (d) the state ladder is
enforced.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "check_physics_release_scorecard",
    _ROOT / "tools/release/check_physics_release_scorecard.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _mod
_SPEC.loader.exec_module(_mod)

_LIVE = _ROOT / "physics_release_scorecard.yml"


def _write(path: Path, dims: list[dict[str, Any]]) -> Path:
    doc = {"schema_version": 1, "allowed_states": list(_mod.LADDER), "dimensions": dims}
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def test_live_scorecard_is_schema_valid_and_binding_consistent() -> None:
    dims = _mod.load_scorecard(_LIVE)
    assert len(dims) >= 8
    # Default validation (schema + anti-theater) passes on the live tree.
    assert _mod.main(["--scorecard", str(_LIVE)]) == 0


def test_live_scorecard_is_honestly_not_physics_ready() -> None:
    """Required dimensions still maturing (ricci L2, second-order audit) keep the
    release honestly NOT physics-ready; the gate fails closed and names them."""
    dims = _mod.load_scorecard(_LIVE)
    blockers = _mod.readiness_blockers(dims)
    assert blockers, "expected at least one required PARTIAL/UNTESTED dimension today"
    assert _mod.main(["--scorecard", str(_LIVE), "--assert-physics-ready"]) == 1


def test_invalid_state_is_rejected() -> None:
    bad = _write(
        _ROOT / "tmp_bad_state.yml",
        [{"id": "x", "required": True, "state": "ALMOST_READY", "binding": None}],
    )
    try:
        with pytest.raises(_mod.ScorecardError, match="state"):
            _mod.load_scorecard(bad)
    finally:
        bad.unlink(missing_ok=True)


def test_verified_state_requires_existing_binding(tmp_path: Path) -> None:
    """Anti-theater: CI_VERIFIED with a missing/empty binding is a hard error."""
    sc = _write(
        tmp_path / "sc.yml",
        [
            {"id": "ghost", "required": True, "state": "CI_VERIFIED", "binding": "does/not/exist.py"},
        ],
    )
    assert _mod.main(["--scorecard", str(sc)]) == 1

    sc2 = _write(
        tmp_path / "sc2.yml",
        [{"id": "ghost", "required": True, "state": "CI_VERIFIED", "binding": None}],
    )
    assert _mod.main(["--scorecard", str(sc2)]) == 1


def test_all_required_verified_is_physics_ready(tmp_path: Path) -> None:
    binding = tmp_path / "real_gate.py"
    binding.write_text("# a real binding artifact\n", encoding="utf-8")
    sc = _write(
        tmp_path / "ready.yml",
        [
            {"id": "a", "required": True, "state": "CI_VERIFIED", "binding": "real_gate.py"},
            {"id": "b", "required": True, "state": "EVIDENCE_BEARING", "binding": "real_gate.py"},
            {"id": "c", "required": False, "state": "PARTIAL", "binding": None},
        ],
    )
    dims = _mod.load_scorecard(sc)
    assert _mod.readiness_blockers(dims) == []
    assert _mod.main(["--scorecard", str(sc), "--assert-physics-ready"]) == 0


def test_partial_required_cannot_be_upgraded_by_assertion(tmp_path: Path) -> None:
    sc = _write(
        tmp_path / "partial.yml",
        [{"id": "a", "required": True, "state": "PARTIAL", "binding": None}],
    )
    assert _mod.main(["--scorecard", str(sc), "--assert-physics-ready"]) == 1
