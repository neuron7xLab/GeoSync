# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""D-002D tests — checkpoint / resume infrastructure.

Pins the seven gates in the execution order:

  G1  checkpoint saves after each cell (atomic write via tmp+rename)
  G2  resume skips completed cells exactly
  G3  config_sha mismatch → FAIL (cannot resume different sweep)
  G4  code_sha drift logged as WARNING not FAIL
  G5  export_ledger produces same sha256 on identical inputs
  G6  mypy --strict / ruff / black (these run in CI)

Plus internal contract tests for cell_key canonicalisation,
idempotent save, and order-invariant ledger sha.

Strict scope: infrastructure tests only. No science assertion.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from research.systemic_risk.sweep_checkpoint import (
    CellResult,
    CheckpointConfigMismatch,
    CheckpointManager,
    canonical_json,
    cell_key,
    config_sha256,
    stable_ledger_sha256,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config() -> dict[str, object]:
    return {
        "substrates": ["block_structured", "ricci_flow"],
        "metrics": ["tau_onset", "auc"],
        "N_grid": [50, 100],
        "lambda_grid": [0.0, 0.5, 1.0],
        "n_seeds": 5,
        "n_bootstrap": 4,
    }


def _full_grid() -> list[str]:
    grid: list[str] = []
    for sub in ("block_structured", "ricci_flow"):
        for m in ("tau_onset", "auc"):
            for n in (50, 100):
                for lam in (0.0, 0.5, 1.0):
                    grid.append(cell_key((n, lam, sub, m)))
    return grid


def _result(key: str, *, payload: dict[str, object] | None = None) -> CellResult:
    return CellResult(
        cell_key=key,
        payload=dict(payload or {"value": 0.1, "signal_ci_ratio": 0.5}),
        duration_seconds=0.123,
    )


# ---------------------------------------------------------------------------
# cell_key + canonical_json + config_sha256
# ---------------------------------------------------------------------------


def test_cell_key_is_stable_across_call_order() -> None:
    a = cell_key((50, 0.5, "block", "tau"))
    b = cell_key((50, 0.5, "block", "tau"))
    assert a == b
    # Canonical-JSON-array form: collision-free, type-preserving,
    # delimiter-safe (cf. test_cell_key_no_type_collision and
    # test_cell_key_no_delimiter_collision below).
    assert a == '[50,0.5,"block","tau"]'


def test_cell_key_distinguishes_distinct_inputs() -> None:
    assert cell_key((50, 0.5, "a", "b")) != cell_key((50, 0.5, "a", "c"))
    assert cell_key((50, 0.5, "a", "b")) != cell_key((50, 0.6, "a", "b"))


def test_cell_key_no_type_collision() -> None:
    """Regression for Codex P1: (5,) (int) and ("5",) (str) must
    produce DIFFERENT keys — otherwise save_cell of one would
    overwrite the other and remaining_cells would silently skip
    work."""
    assert cell_key((5,)) != cell_key(("5",))
    assert cell_key((50, 0.5)) != cell_key((50, "0.5"))
    assert cell_key((True,)) != cell_key((1,))


def test_cell_key_no_delimiter_collision() -> None:
    """Regression for Codex P1: a single string component
    containing the old "|" delimiter must NOT collide with the
    multi-part tuple of the same characters split on the
    delimiter. Without this guarantee, save_cell on one cell
    would silently overwrite an unrelated cell."""
    assert cell_key(("a|b", "c")) != cell_key(("a", "b", "c"))
    assert cell_key(("[5,",)) != cell_key((5,))
    assert cell_key(('"x"', "y")) != cell_key(("x", "y"))


def test_canonical_json_rejects_nan() -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_config_sha256_identical_for_identical_config() -> None:
    a = config_sha256(_config())
    b = config_sha256(_config())
    assert a == b


def test_config_sha256_differs_when_config_differs() -> None:
    cfg = _config()
    a = config_sha256(cfg)
    cfg2 = dict(cfg)
    cfg2["n_seeds"] = 10
    b = config_sha256(cfg2)
    assert a != b


def test_config_sha256_insensitive_to_key_order() -> None:
    cfg_a = _config()
    cfg_b = {k: cfg_a[k] for k in reversed(list(cfg_a))}
    assert config_sha256(cfg_a) == config_sha256(cfg_b)


# ---------------------------------------------------------------------------
# CheckpointManager — create / save / load / resume
# ---------------------------------------------------------------------------


def test_load_or_create_emits_fresh_checkpoint(tmp_path: Path) -> None:
    p = tmp_path / "ckpt.json"
    mgr = CheckpointManager(p, _config(), code_sha="abc123")
    ckpt = mgr.load_or_create()
    assert ckpt.completed_cells == ()
    assert ckpt.results == {}
    assert ckpt.code_sha == "abc123"
    assert p.exists()


def test_g1_save_cell_writes_atomic_and_persists(tmp_path: Path) -> None:
    p = tmp_path / "ckpt.json"
    mgr = CheckpointManager(p, _config(), code_sha="abc123")
    mgr.load_or_create()
    k = cell_key((50, 0.5, "block", "tau"))
    mgr.save_cell(k, _result(k))
    # File should be readable and well-formed JSON
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["sweep_id"]
    assert k in payload["results"]
    # No leftover .tmp file
    assert not p.with_suffix(p.suffix + ".tmp").exists()


def test_g1_save_cell_idempotent(tmp_path: Path) -> None:
    p = tmp_path / "ckpt.json"
    mgr = CheckpointManager(p, _config(), code_sha="abc123")
    mgr.load_or_create()
    k = cell_key((50, 0.5, "block", "tau"))
    r = _result(k)
    mgr.save_cell(k, r)
    mtime1 = p.stat().st_mtime_ns
    mgr.save_cell(k, r)  # identical write — should be no-op on results
    assert mgr.export_ledger_sha256() == stable_ledger_sha256(mgr.load_or_create())
    # mtime may or may not change depending on OS; the key
    # invariant is that the *contents* didn't change. Verify
    # the ledger hash is identical to a fresh manager that
    # re-loads the same file.
    fresh = CheckpointManager(p, _config(), code_sha="abc123").load_or_create()
    assert stable_ledger_sha256(fresh) == mgr.export_ledger_sha256()
    _ = mtime1  # silence unused


def test_g2_resume_skips_completed_cells_exactly(tmp_path: Path) -> None:
    p = tmp_path / "ckpt.json"
    mgr = CheckpointManager(p, _config(), code_sha="abc123")
    mgr.load_or_create()
    grid = _full_grid()
    done = [grid[0], grid[3], grid[7]]
    for k in done:
        mgr.save_cell(k, _result(k))
    remaining = mgr.remaining_cells(grid)
    assert set(done).isdisjoint(set(remaining))
    assert set(remaining) | set(done) == set(grid)
    # Order preservation: remaining keeps the order of full_grid
    assert remaining == [k for k in grid if k not in set(done)]


def test_g2_resume_loads_from_disk_into_new_manager(tmp_path: Path) -> None:
    p = tmp_path / "ckpt.json"
    mgr_a = CheckpointManager(p, _config(), code_sha="abc123")
    mgr_a.load_or_create()
    k = cell_key((50, 0.5, "block", "tau"))
    mgr_a.save_cell(k, _result(k))

    # Fresh manager re-points at the same path
    mgr_b = CheckpointManager(p, _config(), code_sha="abc123")
    ckpt = mgr_b.load_or_create()
    assert k in ckpt.completed_cells
    assert mgr_b.is_done(k)


# ---------------------------------------------------------------------------
# G3 — config_sha mismatch refuses to merge
# ---------------------------------------------------------------------------


def test_g3_config_mismatch_raises(tmp_path: Path) -> None:
    p = tmp_path / "ckpt.json"
    mgr_a = CheckpointManager(p, _config(), code_sha="abc123")
    mgr_a.load_or_create()
    mgr_a.save_cell(
        cell_key((50, 0.5, "block", "tau")),
        _result(cell_key((50, 0.5, "block", "tau"))),
    )
    # Different config (n_seeds bumped)
    cfg2 = dict(_config())
    cfg2["n_seeds"] = 99
    with pytest.raises(CheckpointConfigMismatch):
        CheckpointManager(p, cfg2, code_sha="abc123").load_or_create()


# ---------------------------------------------------------------------------
# G4 — code_sha drift logs WARNING, allows resume
# ---------------------------------------------------------------------------


def test_g4_code_drift_logs_warning_but_resumes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    p = tmp_path / "ckpt.json"
    mgr_a = CheckpointManager(p, _config(), code_sha="codeA")
    mgr_a.load_or_create()
    k = cell_key((50, 0.5, "block", "tau"))
    mgr_a.save_cell(k, _result(k))

    with caplog.at_level(logging.WARNING, logger="research.systemic_risk.sweep_checkpoint"):
        mgr_b = CheckpointManager(p, _config(), code_sha="codeB")
        ckpt = mgr_b.load_or_create()

    # Resume succeeded (no exception)
    assert k in ckpt.completed_cells
    # Warning recorded
    assert any("code_sha drift" in rec.message for rec in caplog.records)
    # Drift event captured in checkpoint
    assert any(
        e.get("from_code_sha") == "codeA" and e.get("to_code_sha") == "codeB"
        for e in ckpt.code_drift_events
    )


def test_g4_drift_event_persisted_to_disk_immediately(tmp_path: Path) -> None:
    """Regression for Codex P2: drift event must be written back to
    disk during load_or_create, NOT deferred to the next save_cell.
    Without this, a run with zero remaining cells exits without
    persisting the drift, and the same WARNING repeats on every
    subsequent load — the audit trail is silently lost."""
    p = tmp_path / "ckpt.json"
    mgr_a = CheckpointManager(p, _config(), code_sha="codeA")
    mgr_a.load_or_create()
    # No save_cell — checkpoint is empty but exists on disk.

    # Drift on second open
    mgr_b = CheckpointManager(p, _config(), code_sha="codeB")
    mgr_b.load_or_create()
    # ** No save_cell on mgr_b ** — simulates a run that exits
    # immediately because remaining_cells is empty.

    # Open fresh and inspect on-disk state
    persisted = json.loads(p.read_text(encoding="utf-8"))
    assert persisted["code_sha"] == "codeB"
    drift_events = persisted["code_drift_events"]
    assert any(
        e.get("from_code_sha") == "codeA" and e.get("to_code_sha") == "codeB" for e in drift_events
    )

    # Re-opening with codeB again must NOT add a duplicate drift event
    mgr_c = CheckpointManager(p, _config(), code_sha="codeB")
    mgr_c.load_or_create()
    persisted2 = json.loads(p.read_text(encoding="utf-8"))
    drift_count_a_to_b = sum(
        1
        for e in persisted2["code_drift_events"]
        if e.get("from_code_sha") == "codeA" and e.get("to_code_sha") == "codeB"
    )
    assert (
        drift_count_a_to_b == 1
    ), f"drift event should be recorded exactly once; found {drift_count_a_to_b}"


# ---------------------------------------------------------------------------
# G5 — export_ledger sha256 stability + order invariance
# ---------------------------------------------------------------------------


def test_g5_export_ledger_sha_stable_across_managers(tmp_path: Path) -> None:
    p = tmp_path / "ckpt.json"
    mgr = CheckpointManager(p, _config(), code_sha="abc123")
    mgr.load_or_create()
    for i, k in enumerate(_full_grid()[:6]):
        mgr.save_cell(k, _result(k, payload={"value": float(i), "rank": i}))
    sha_a = mgr.export_ledger_sha256()

    # Fresh manager pointing at the same file
    mgr_b = CheckpointManager(p, _config(), code_sha="abc123")
    mgr_b.load_or_create()
    sha_b = mgr_b.export_ledger_sha256()
    assert sha_a == sha_b


def test_g5_export_ledger_sha_order_invariant(tmp_path: Path) -> None:
    grid = _full_grid()[:6]
    # Write in order A
    p_a = tmp_path / "ckpt_a.json"
    mgr_a = CheckpointManager(p_a, _config(), code_sha="x")
    mgr_a.load_or_create()
    for i, k in enumerate(grid):
        mgr_a.save_cell(k, _result(k, payload={"value": float(i)}))

    # Write the same cells in reverse order to a second file
    p_b = tmp_path / "ckpt_b.json"
    mgr_b = CheckpointManager(p_b, _config(), code_sha="x")
    mgr_b.load_or_create()
    for i, k in reversed(list(enumerate(grid))):
        mgr_b.save_cell(k, _result(k, payload={"value": float(i)}))

    assert mgr_a.export_ledger_sha256() == mgr_b.export_ledger_sha256()


def test_g5_export_ledger_records_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "ckpt.json"
    mgr = CheckpointManager(p, _config(), code_sha="abc123")
    mgr.load_or_create()
    k = cell_key((50, 0.5, "block", "tau"))
    payload = {"signal_mean": 1.23, "null_mean": 0.45, "direction": "hindered"}
    mgr.save_cell(k, _result(k, payload=payload))
    rows = mgr.export_ledger()
    assert len(rows) == 1
    assert rows[0]["cell_key"] == k
    assert rows[0]["payload"]["direction"] == "hindered"


# ---------------------------------------------------------------------------
# CellResult round-trip
# ---------------------------------------------------------------------------


def test_cell_result_round_trip() -> None:
    r = CellResult(
        cell_key="x|y|z",
        payload={"a": 1, "b": "two"},
        duration_seconds=0.5,
    )
    r2 = CellResult.from_dict(r.to_dict())
    assert r2 == r


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_load_existing_corrupt_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "ckpt.json"
    p.write_text("this is not json", encoding="utf-8")
    mgr = CheckpointManager(p, _config(), code_sha="x")
    with pytest.raises(RuntimeError):
        mgr.load_or_create()


def test_remaining_cells_empty_when_all_done(tmp_path: Path) -> None:
    p = tmp_path / "ckpt.json"
    mgr = CheckpointManager(p, _config(), code_sha="x")
    mgr.load_or_create()
    grid = _full_grid()[:3]
    for k in grid:
        mgr.save_cell(k, _result(k))
    assert mgr.remaining_cells(grid) == []


def test_remaining_cells_full_when_none_done(tmp_path: Path) -> None:
    p = tmp_path / "ckpt.json"
    mgr = CheckpointManager(p, _config(), code_sha="x")
    mgr.load_or_create()
    grid = _full_grid()[:3]
    assert mgr.remaining_cells(grid) == grid
