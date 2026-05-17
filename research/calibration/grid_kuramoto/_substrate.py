# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Shared, byte-preserving substrate for the CALIB-GRID lineages.

Every CALIB-GRID lineage (CALIB-GRID-001, R1, the identifiability
front-gate, CALIB-GRID-002) previously hand-rolled its own copy of four
identical primitives:

* the ``json.dumps(ledger, sort_keys=True)`` → ``sha256`` ledger
  content-hash (4 verbatim copies);
* a ``_branch_sha`` git-HEAD reader (2 copies differing only in the
  ``parents[N]`` depth);
* the frozen parent pre-registration / parent-ledger git-sha string
  literals (3 copies of the same two constants);
* the thresholded edge-support F1 (2 byte-identical copies).

This module is the single source of truth for all four. It is **purely
structural**: every helper reproduces the exact bytes / numerics the
per-lineage copies produced, so every already-merged sha-pinned
``RESULTS.json`` and every existing drift / no-peek / bit-stability test
stays valid without modification. No threshold, gate, seed, σ, θ₀ or
decision rule lives here — those remain frozen in the per-lineage
pre-registration documents and gate modules.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "FROZEN_PREREG_SHA",
    "PARENT_LEDGER_SHA256",
    "PREREG_BRANCH_BASE_SHA",
    "Gate",
    "GateRow",
    "branch_sha",
    "ledger_sha256",
    "topology_f1",
]

# --- Frozen provenance constants (audited content hashes, not secrets) ----
# These pin the CALIB-GRID-001 parent lineage. They are *read* by every
# child lineage and never redefined; centralising the literal removes the
# three divergent copies (run.py / validate.py / cg002.py) without
# changing any emitted byte.
# audited: frozen parent pre-registration git sha, not a credential
FROZEN_PREREG_SHA = "d170d48afa5066c13edeb40b2c1904b3fd708516"  # pragma: allowlist secret
# audited: parent calibration ledger content hash, not a credential
PARENT_LEDGER_SHA256 = (
    "ed8d409b7b222eb053572d6bf9ab6e98c5f4918be1cae384864733a2b4d72aaf"  # pragma: allowlist secret
)
# audited: branch base sha the CALIB-GRID-002 pre-registration was committed off
PREREG_BRANCH_BASE_SHA = "a5e0d533b2201c999b31c792773e858f8da713bf"  # pragma: allowlist secret


def ledger_sha256(ledger: dict[str, Any]) -> str:
    """Content hash of ``ledger`` over its canonical ``sort_keys`` JSON.

    Byte-for-byte identical to the per-lineage copies it replaces::

        payload = json.dumps(ledger, sort_keys=True).encode("utf-8")
        hashlib.sha256(payload).hexdigest()

    ``sort_keys=True`` makes the hash invariant to key *insertion* order,
    so consolidating the four builders cannot perturb any sha-pinned
    artifact even though the unified call site differs.
    """
    payload = json.dumps(ledger, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def branch_sha(repo_root: Path) -> str:
    """Best-effort current commit sha for a ledger provenance field.

    ``repo_root`` is the repository root (the directory containing
    ``.git``). Resolves a symbolic ``ref:`` HEAD to its packed/loose
    object id; returns ``"unknown"`` on any filesystem error (the field
    is provenance only and is excluded from every drift comparison).
    """
    head = repo_root / ".git" / "HEAD"
    try:
        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref:"):
            ref_path = head.parent / ref.split(" ", 1)[1]
            return ref_path.read_text(encoding="utf-8").strip()
        return ref
    except OSError:
        return "unknown"


def topology_f1(
    k_true: NDArray[np.float64],
    k_hat: NDArray[np.float64],
    rel_threshold: float,
) -> tuple[float, int, int]:
    """Edge-support F1 of the thresholded recovered adjacency.

    The recovered edge is "present" if ``|K_hat_ij|`` exceeds
    ``rel_threshold · max|K_hat|``; the truth edge is present if
    ``|K_true_ij| > 0``. Diagonal excluded. Returns
    ``(f1, n_true_edges, n_recovered_edges)``. Numerically identical to
    the two ``_topology_f1`` copies (calibration.py / cg002.py) it
    replaces.
    """
    n = k_true.shape[0]
    off = ~np.eye(n, dtype=bool)
    true_mask = (np.abs(k_true) > 0.0) & off
    scale = float(np.max(np.abs(k_hat)))
    if scale <= 0.0:
        return 0.0, int(true_mask.sum()), 0
    hat_mask = (np.abs(k_hat) > rel_threshold * scale) & off
    tp = int((true_mask & hat_mask).sum())
    fp = int((~true_mask & hat_mask).sum())
    fn = int((true_mask & ~hat_mask).sum())
    denom = 2 * tp + fp + fn
    f1 = (2.0 * tp / denom) if denom > 0 else 1.0
    return f1, int(true_mask.sum()), int(hat_mask.sum())


@dataclass(frozen=True)
class Gate:
    """A single pre-registered numeric gate (fail-closed comparator).

    Unifies the three copy-evolved gate dataclasses (``GateVerdict`` in
    ``gates.py`` and ``CG002Gate`` in ``cg002.py``) into one shape. The
    thresholds themselves stay frozen in the per-lineage pre-registration
    documents — this only removes the duplicated *carrier* type and its
    twice-copied operator-dispatch logic.
    """

    name: str
    metric_key: str
    operator: str  # "<=" or ">="
    threshold: float
    localises_to: str

    def check(self, observed: float) -> bool:
        """Fail-closed comparison; an unknown operator raises."""
        if self.operator == "<=":
            return observed <= self.threshold
        if self.operator == ">=":
            return observed >= self.threshold
        raise ValueError(f"unknown operator {self.operator!r}")


@dataclass(frozen=True)
class GateRow:
    """Outcome of evaluating one :class:`Gate` against a metric value.

    ``to_dict`` reproduces the exact key set / order the legacy
    ``GateResult.to_dict`` and the cg002 ``_emit`` row dict emitted, so
    every ``gates`` array in every sha-pinned ledger stays byte-identical
    under ``json.dumps(sort_keys=True)``.
    """

    name: str
    metric_key: str
    observed: float
    operator: str
    threshold: float
    passed: bool
    localises_to: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view (sorted keys downstream)."""
        return {
            "name": self.name,
            "metric_key": self.metric_key,
            "observed": self.observed,
            "operator": self.operator,
            "threshold": self.threshold,
            "passed": self.passed,
            "localises_to": self.localises_to,
        }
