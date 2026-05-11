# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""D-002D — Checkpoint / resume infrastructure for long sweeps.

Rationale
=========
The D-002B-A high-budget sweep (PR #656, issue #652) kept every
completed work-unit result in coordinator RAM until end-of-run
JSON write. A 17 h run that loses RAM mid-flight loses every
cell. The D-002C Signal Amplification Sweep grid (issue #654)
is 69 120 evaluations on a substrate × metric × variance grid;
running it without per-cell checkpointing is operationally
unsafe.

This module provides:

  * ``CellResult``        — frozen dataclass per (cell_key) outcome
  * ``SweepCheckpoint``   — frozen snapshot of completed work
  * ``CheckpointManager`` — atomic per-cell save / load / resume

Strict scope
============
Driver / infrastructure ONLY. NO science change. NO test
relaxation. NO threshold tuning. NO claim layer. The capsule
emitted by any sweep using this module remains scoped per its
own protocol (D-002C, etc.); this module is callable from any
sweep but never emits a claim of its own.

The contract is small and frozen:

  * cell_key is hashable (we use a stable repr key for I/O)
  * save_cell is atomic on POSIX (write `.tmp` → ``os.replace``)
  * resume is a pure subtraction:
        ``remaining_cells(full_grid) = full_grid - completed_cells``
  * sweep identity is the ``config_sha`` over the canonicalised
    sweep_config dict; a mismatch is a hard fail (different
    sweep — refuse to merge ledgers)
  * code drift is a soft warning (``code_sha``): the sweep can
    resume across a refactor as long as the science contract
    didn't change, but we log the drift so the reviewer can
    audit
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellResult:
    """One sweep cell's result.

    ``cell_key`` is the canonicalised string form (used as JSON key
    + as the key in ``SweepCheckpoint.results``). ``payload`` is an
    arbitrary JSON-serialisable dict — the science layer decides
    its shape; this module makes no claims about its contents.
    """

    cell_key: str
    payload: dict[str, Any]
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_key": self.cell_key,
            "payload": dict(self.payload),
            "duration_seconds": float(self.duration_seconds),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CellResult:
        return cls(
            cell_key=str(d["cell_key"]),
            payload=dict(d.get("payload", {})),
            duration_seconds=float(d.get("duration_seconds", 0.0)),
        )


@dataclass(frozen=True)
class SweepCheckpoint:
    """Frozen snapshot of a sweep's checkpoint state.

    The on-disk JSON format is the canonical form: it round-trips
    through :meth:`to_dict` / :meth:`from_dict` and is the input
    to :func:`stable_ledger_sha256`.
    """

    sweep_id: str
    config_sha: str
    code_sha: str
    created_at: str
    last_updated: str
    completed_cells: tuple[str, ...]
    results: dict[str, CellResult] = field(default_factory=dict)
    code_drift_events: tuple[dict[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "sweep_id": self.sweep_id,
            "config_sha": self.config_sha,
            "code_sha": self.code_sha,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "completed_cells": list(self.completed_cells),
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "code_drift_events": [dict(e) for e in self.code_drift_events],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SweepCheckpoint:
        return cls(
            sweep_id=str(d["sweep_id"]),
            config_sha=str(d["config_sha"]),
            code_sha=str(d["code_sha"]),
            created_at=str(d["created_at"]),
            last_updated=str(d["last_updated"]),
            completed_cells=tuple(str(x) for x in d.get("completed_cells", ())),
            results={str(k): CellResult.from_dict(v) for k, v in d.get("results", {}).items()},
            code_drift_events=tuple(
                {str(kk): str(vv) for kk, vv in e.items()} for e in d.get("code_drift_events", [])
            ),
        )


# ---------------------------------------------------------------------------
# Determinism helpers
# ---------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    """Stable JSON form: sort keys, separators tight, no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_sha256(sweep_config: dict[str, Any]) -> str:
    """sha256 of the canonicalised sweep_config dict.

    Two callers with the same ``sweep_config`` always get the same
    sha. A different ``sweep_config`` always gets a different sha.
    NaN / non-finite values raise; sets and tuples are not
    JSON-canonical and must be normalised by the caller.
    """
    payload = canonical_json(sweep_config)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cell_key(parts: tuple[Any, ...]) -> str:
    """Canonical string form for a heterogeneous cell tuple.

    Example: ``cell_key((50, 0.5, "block_structured", "tau_onset"))``
    → ``"50|0.5|block_structured|tau_onset"``.

    The form is stable (no Python ``repr`` quirks), hashable as a
    str, and round-trips through JSON without losing structure
    (since each component is rendered with canonical_json).
    """
    return "|".join(canonical_json(p).strip('"') for p in parts)


def stable_ledger_sha256(checkpoint: SweepCheckpoint) -> str:
    """Order-invariant sha256 of the checkpoint's completed results.

    Use this to verify two checkpoints carry the same evidence
    independent of the order in which cells were completed.
    """
    rows = sorted((k, canonical_json(v.to_dict())) for k, v in checkpoint.results.items())
    return hashlib.sha256(canonical_json(rows).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------


class CheckpointConfigMismatch(RuntimeError):
    """Raised when an on-disk checkpoint's config_sha disagrees with
    the caller's sweep_config — refusing to merge ledgers from
    different sweeps."""


class CheckpointManager:
    """Atomic per-cell checkpoint store.

    Usage:

        mgr = CheckpointManager(path, sweep_config={...})
        ckpt = mgr.load_or_create()
        for cell in mgr.remaining_cells(full_grid):
            result = compute(cell)  # science layer
            mgr.save_cell(cell_key=cell, result=result)
        df = mgr.export_ledger()
    """

    def __init__(
        self,
        path: Path,
        sweep_config: dict[str, Any],
        *,
        code_sha: str = "unknown",
    ) -> None:
        self._path = Path(path)
        self._sweep_config = dict(sweep_config)
        self._config_sha = config_sha256(self._sweep_config)
        self._sweep_id = self._config_sha[:16]
        self._code_sha = str(code_sha)
        self._checkpoint: SweepCheckpoint | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def config_sha(self) -> str:
        return self._config_sha

    @property
    def sweep_id(self) -> str:
        return self._sweep_id

    def load_or_create(self) -> SweepCheckpoint:
        """Load on-disk checkpoint OR create a fresh one.

        Raises ``CheckpointConfigMismatch`` if an on-disk file
        exists but its ``config_sha`` doesn't match the caller's
        sweep_config — we refuse to merge across sweeps.
        """
        if self._path.exists():
            try:
                payload = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise RuntimeError(f"checkpoint at {self._path} is unreadable: {exc}") from exc
            on_disk = SweepCheckpoint.from_dict(payload)
            if on_disk.config_sha != self._config_sha:
                raise CheckpointConfigMismatch(
                    f"on-disk config_sha {on_disk.config_sha} != "
                    f"caller config_sha {self._config_sha} — refusing to "
                    f"merge ledgers from different sweeps"
                )
            # Code drift: warn but do not fail
            drift_events = list(on_disk.code_drift_events)
            if on_disk.code_sha != self._code_sha:
                drift_event = {
                    "from_code_sha": on_disk.code_sha,
                    "to_code_sha": self._code_sha,
                    "at": _now_iso(),
                }
                drift_events.append(drift_event)
                logger.warning(
                    "checkpoint code_sha drift: %s -> %s (resume permitted; "
                    "verify science contract unchanged)",
                    on_disk.code_sha,
                    self._code_sha,
                )
            self._checkpoint = SweepCheckpoint(
                sweep_id=on_disk.sweep_id,
                config_sha=on_disk.config_sha,
                code_sha=self._code_sha,
                created_at=on_disk.created_at,
                last_updated=on_disk.last_updated,
                completed_cells=on_disk.completed_cells,
                results=dict(on_disk.results),
                code_drift_events=tuple(drift_events),
            )
            return self._checkpoint

        now = _now_iso()
        self._checkpoint = SweepCheckpoint(
            sweep_id=self._sweep_id,
            config_sha=self._config_sha,
            code_sha=self._code_sha,
            created_at=now,
            last_updated=now,
            completed_cells=(),
            results={},
        )
        self._atomic_write(self._checkpoint)
        return self._checkpoint

    def save_cell(self, cell_key_str: str, result: CellResult) -> None:
        """Save one cell result to disk atomically.

        Idempotent: writing the same (cell_key, result) twice is
        a no-op on the ledger. If a different result is written
        for the same cell_key, the latest write wins (we log it).
        """
        if self._checkpoint is None:
            self.load_or_create()
        assert self._checkpoint is not None
        completed = set(self._checkpoint.completed_cells)
        if cell_key_str in completed:
            existing = self._checkpoint.results.get(cell_key_str)
            if existing is not None and existing.to_dict() == result.to_dict():
                return  # idempotent no-op
            logger.warning("checkpoint cell %s overwritten with new result", cell_key_str)
        new_results = dict(self._checkpoint.results)
        new_results[cell_key_str] = result
        new_completed = tuple(sorted(set(self._checkpoint.completed_cells) | {cell_key_str}))
        self._checkpoint = SweepCheckpoint(
            sweep_id=self._checkpoint.sweep_id,
            config_sha=self._checkpoint.config_sha,
            code_sha=self._checkpoint.code_sha,
            created_at=self._checkpoint.created_at,
            last_updated=_now_iso(),
            completed_cells=new_completed,
            results=new_results,
            code_drift_events=self._checkpoint.code_drift_events,
        )
        self._atomic_write(self._checkpoint)

    def is_done(self, cell_key_str: str) -> bool:
        if self._checkpoint is None:
            self.load_or_create()
        assert self._checkpoint is not None
        return cell_key_str in set(self._checkpoint.completed_cells)

    def remaining_cells(self, full_grid: list[str]) -> list[str]:
        """Return the subset of full_grid that is NOT yet completed."""
        if self._checkpoint is None:
            self.load_or_create()
        assert self._checkpoint is not None
        done = set(self._checkpoint.completed_cells)
        return [k for k in full_grid if k not in done]

    def export_ledger(self) -> list[dict[str, Any]]:
        """Order-stable export: a sorted list of cell dicts.

        Returning a plain list-of-dicts (not pandas) keeps this
        module free of a hard pandas dependency at the
        infrastructure layer. The science layer can wrap with
        ``pd.DataFrame.from_records(mgr.export_ledger())`` if it
        wants a frame.
        """
        if self._checkpoint is None:
            self.load_or_create()
        assert self._checkpoint is not None
        rows = sorted((k, v.to_dict()) for k, v in self._checkpoint.results.items())
        return [{"cell_key": k, **v} for k, v in rows]

    def export_ledger_sha256(self) -> str:
        """Stable sha256 of the ledger (cf. :func:`stable_ledger_sha256`)."""
        if self._checkpoint is None:
            self.load_or_create()
        assert self._checkpoint is not None
        return stable_ledger_sha256(self._checkpoint)

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _atomic_write(self, checkpoint: SweepCheckpoint) -> None:
        """Write to ``path.tmp`` then ``os.replace`` to final path.

        Atomic on POSIX: a reader either sees the old file or the
        new file, never a half-written file.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        payload = checkpoint.to_dict()
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(tmp_path, self._path)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
