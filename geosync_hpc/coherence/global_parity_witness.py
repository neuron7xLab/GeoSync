# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Global parity witness — engineering analog of Kitaev-readout discipline.

pattern_id:        P4_GLOBAL_PARITY_WITNESS
source_id:         S4_KITAEV_PARITY_READOUT
claim_tier:        ENGINEERING_ANALOG
implementation:    PR #456 (this module)

Engineering analog
==================

In a Kitaev-chain readout, local charge sensing cannot distinguish the
two parity states; only a global capacitance measurement can. The
property is invisible to local probes by construction.

This module imports that *inference discipline*, not the physics. A
GeoSync system claim cannot be granted on local witnesses alone:
required global surfaces — claim ledger, dependency truth, invariant
coverage, CI gate — must independently agree. If any required surface
fails or its status is unknown, the witness refuses to declare global
consistency.

The named lie blocked here is::

    "local pass = global pass"

The witness's contract is symmetric: every required surface either
agrees, or the global verdict is FALSE. There is no numeric health
score, no confidence float, no percent. There is a boolean and a list
of named failed surfaces.

Statuses
========

The witness reports exactly one of::

    GLOBAL_PASS                 every local witness passed AND
                                every required global surface passed
    LOCAL_FAILURE               at least one local witness reported
                                passed=False
    CLAIM_LEDGER_FAILURE        claim_ledger_ok is False
    DEPENDENCY_TRUTH_FAILURE    dependency_truth_ok is False
    INVARIANT_COVERAGE_FAILURE  invariant_coverage_ok is False
    CI_GATE_FAILURE             ci_gate_ok is False
    CI_GATE_UNKNOWN             ci_gate_ok is None

Status is the *primary* failure surface; `failed_surfaces` lists every
failing surface (including locals) in deterministic order. When more
than one surface fails simultaneously, status is selected by the fixed
priority documented in ``assess_global_parity``.

Explicit non-claims
===================

This module does NOT claim:
  - any one-to-one correspondence with quantum-parity physics
  - that GLOBAL_PASS is a forecast or trading signal
  - that the witness is exhaustive of "system-level correctness" — it
    is exactly the AND-aggregation of the surfaces declared in
    ``required_surfaces`` plus the supplied locals

Determinism contract
====================

  - assess_global_parity(...) is pure: no I/O, no clock, no random, no
    global state.
  - Identical inputs produce byte-identical witness outputs.
  - failed_surfaces is sorted in a stable, deterministic order.
  - There is no module-level mutable state.
  - There is no numeric health / score / confidence / percent / index
    field anywhere on the witness.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

__all__ = [
    "GlobalParityStatus",
    "LocalWitness",
    "GlobalParityInput",
    "GlobalParityWitness",
    "assess_global_parity",
]


# ---------------------------------------------------------------------------
# Stable surface names. Strings are part of the API surface.
# ---------------------------------------------------------------------------

_SURFACE_CLAIM_LEDGER = "claim_ledger"
_SURFACE_DEPENDENCY_TRUTH = "dependency_truth"
_SURFACE_INVARIANT_COVERAGE = "invariant_coverage"
_SURFACE_CI_GATE = "ci_gate"

_GLOBAL_SURFACES: tuple[str, ...] = (
    _SURFACE_CLAIM_LEDGER,
    _SURFACE_DEPENDENCY_TRUTH,
    _SURFACE_INVARIANT_COVERAGE,
    _SURFACE_CI_GATE,
)

_VALID_TIERS: frozenset[str] = frozenset(
    {
        "FACT",
        "ENGINEERING_ANALOG",
        "HYPOTHESIS",
        "SPECULATIVE",
    }
)


# ---------------------------------------------------------------------------
# Reasons (stable) and falsifier text.
# ---------------------------------------------------------------------------

_REASON_GLOBAL_PASS = "OK_ALL_REQUIRED_SURFACES_AGREE"
_REASON_LOCAL_FAIL = "LOCAL_WITNESS_REPORTED_FAILURE"
_REASON_CLAIM_FAIL = "CLAIM_LEDGER_REPORTED_FAILURE"
_REASON_DEPS_FAIL = "DEPENDENCY_TRUTH_REPORTED_FAILURE"
_REASON_INVAR_FAIL = "INVARIANT_COVERAGE_REPORTED_FAILURE"
_REASON_CI_FAIL = "CI_GATE_REPORTED_FAILURE"
_REASON_CI_UNKNOWN = "CI_GATE_STATUS_UNKNOWN"


_FALSIFIER_TEXT = (
    "GLOBAL_PASS was returned but at least one of the following held: "
    "any local witness reported passed=False; claim_ledger_ok was False; "
    "dependency_truth_ok was False; invariant_coverage_ok was False; "
    "ci_gate_ok was False or None. OR: a required global surface was "
    "silently ignored by the aggregation. OR: failed_surfaces was "
    "non-deterministic across identical inputs."
)


# ---------------------------------------------------------------------------
# Status type
# ---------------------------------------------------------------------------

GlobalParityStatus = Literal[
    "GLOBAL_PASS",
    "LOCAL_FAILURE",
    "CLAIM_LEDGER_FAILURE",
    "DEPENDENCY_TRUTH_FAILURE",
    "INVARIANT_COVERAGE_FAILURE",
    "CI_GATE_UNKNOWN",
    "CI_GATE_FAILURE",
]


@dataclass(frozen=True)
class LocalWitness:
    """One local check entering the global aggregation.

    Locals do not have to share a tier vocabulary with the rest of the
    system; the witness only enforces that ``tier`` is one of the
    allowed strings, that ``name`` is a non-empty identifier, and that
    ``passed`` is a real bool.
    """

    name: str
    passed: bool
    tier: str
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("LocalWitness.name must be a non-empty string")
        if not isinstance(self.passed, bool):
            raise TypeError("LocalWitness.passed must be a bool")
        if not isinstance(self.tier, str) or self.tier not in _VALID_TIERS:
            raise ValueError(
                f"LocalWitness.tier must be one of {sorted(_VALID_TIERS)} (got {self.tier!r})"
            )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("LocalWitness.reason must be a non-empty string")


@dataclass(frozen=True)
class GlobalParityInput:
    """Single global-parity question.

    `local_witnesses` carries the local module checks. The aggregation
    refuses an empty tuple — a global-parity claim with no local
    witnesses is structurally meaningless.

    `claim_ledger_ok`, `dependency_truth_ok`, `invariant_coverage_ok`
    are required global surfaces; `ci_gate_ok` is `bool | None` so the
    caller can distinguish "CI definitively failed" from "CI status is
    unknown" — both fail the global verdict, but with different status
    strings.

    `required_surfaces` declares which global surfaces the caller
    wishes enforced. Every entry must be one of the canonical surface
    names; the canonical set itself is exposed via
    ``GlobalParityInput.canonical_surfaces()``.
    """

    local_witnesses: tuple[LocalWitness, ...]
    claim_ledger_ok: bool
    dependency_truth_ok: bool
    invariant_coverage_ok: bool
    ci_gate_ok: bool | None
    required_surfaces: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.local_witnesses, tuple):
            raise TypeError(
                "local_witnesses must be a tuple of LocalWitness; convert "
                "via tuple(...) at the call site"
            )
        if len(self.local_witnesses) == 0:
            raise ValueError("local_witnesses must be non-empty")
        for i, lw in enumerate(self.local_witnesses):
            if not isinstance(lw, LocalWitness):
                raise TypeError(f"local_witnesses[{i}] must be a LocalWitness instance")

        for name, value in (
            ("claim_ledger_ok", self.claim_ledger_ok),
            ("dependency_truth_ok", self.dependency_truth_ok),
            ("invariant_coverage_ok", self.invariant_coverage_ok),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool")

        if self.ci_gate_ok is not None and not isinstance(self.ci_gate_ok, bool):
            raise TypeError("ci_gate_ok must be a bool or None")

        if not isinstance(self.required_surfaces, tuple):
            raise TypeError("required_surfaces must be a tuple of strings")
        for s in self.required_surfaces:
            if not isinstance(s, str) or not s.strip():
                raise ValueError("required_surfaces entries must be non-empty strings")
            if s not in _GLOBAL_SURFACES:
                raise ValueError(
                    f"required_surfaces contains unknown surface {s!r}; "
                    f"valid surfaces are {_GLOBAL_SURFACES}"
                )

    @staticmethod
    def canonical_surfaces() -> tuple[str, ...]:
        """Canonical global surface names, in deterministic order."""
        return _GLOBAL_SURFACES


@dataclass(frozen=True)
class GlobalParityWitness:
    """Outcome of one global-parity assessment.

    `globally_consistent` is True iff status == "GLOBAL_PASS".
    `failed_surfaces` lists every failing surface (locals are listed
    by ``local:<name>`` and globals by their canonical surface name)
    in deterministic order — locals first (by name), then globals in
    canonical order.
    """

    globally_consistent: bool
    status: GlobalParityStatus
    failed_surfaces: tuple[str, ...]
    local_pass_count: int
    local_fail_count: int
    reason: str
    falsifier: str
    evidence_fields: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


# ---------------------------------------------------------------------------
# Pure assessment function
# ---------------------------------------------------------------------------


def _classify_global(
    *,
    has_local_failure: bool,
    claim_ok: bool,
    deps_ok: bool,
    invar_ok: bool,
    ci_ok: bool | None,
    required: frozenset[str],
) -> tuple[GlobalParityStatus, str]:
    """Pick the primary status by fixed priority.

    Priority (first failing surface wins):

        1. local witness failure
        2. claim_ledger_ok is False         (if required)
        3. dependency_truth_ok is False     (if required)
        4. invariant_coverage_ok is False   (if required)
        5. ci_gate_ok is False              (if required)
        6. ci_gate_ok is None               (if required)
        7. otherwise                         GLOBAL_PASS
    """
    if has_local_failure:
        return "LOCAL_FAILURE", _REASON_LOCAL_FAIL
    if _SURFACE_CLAIM_LEDGER in required and not claim_ok:
        return "CLAIM_LEDGER_FAILURE", _REASON_CLAIM_FAIL
    if _SURFACE_DEPENDENCY_TRUTH in required and not deps_ok:
        return "DEPENDENCY_TRUTH_FAILURE", _REASON_DEPS_FAIL
    if _SURFACE_INVARIANT_COVERAGE in required and not invar_ok:
        return "INVARIANT_COVERAGE_FAILURE", _REASON_INVAR_FAIL
    if _SURFACE_CI_GATE in required:
        if ci_ok is False:
            return "CI_GATE_FAILURE", _REASON_CI_FAIL
        if ci_ok is None:
            return "CI_GATE_UNKNOWN", _REASON_CI_UNKNOWN
    return "GLOBAL_PASS", _REASON_GLOBAL_PASS


def assess_global_parity(input_: GlobalParityInput) -> GlobalParityWitness:
    """Classify the global-parity question carried by ``input_``.

    Pure function. Reads only the input dataclass; no I/O, no clock,
    no global state. Returns one ``GlobalParityWitness`` describing
    the structural verdict.

    A required surface that reports False or None makes
    ``globally_consistent`` False with the corresponding status. An
    empty ``required_surfaces`` tuple means "the caller does not
    require any global surface" — only the local witnesses are
    aggregated, and a CI gate of None / False does not affect the
    verdict (it would not be required).
    """
    required = frozenset(input_.required_surfaces)

    failed_locals: list[str] = []
    pass_count = 0
    for lw in input_.local_witnesses:
        if lw.passed:
            pass_count += 1
        else:
            failed_locals.append(lw.name)
    has_local_failure = bool(failed_locals)

    failed_globals: list[str] = []
    if _SURFACE_CLAIM_LEDGER in required and not input_.claim_ledger_ok:
        failed_globals.append(_SURFACE_CLAIM_LEDGER)
    if _SURFACE_DEPENDENCY_TRUTH in required and not input_.dependency_truth_ok:
        failed_globals.append(_SURFACE_DEPENDENCY_TRUTH)
    if _SURFACE_INVARIANT_COVERAGE in required and not input_.invariant_coverage_ok:
        failed_globals.append(_SURFACE_INVARIANT_COVERAGE)
    if _SURFACE_CI_GATE in required and (input_.ci_gate_ok is None or not input_.ci_gate_ok):
        failed_globals.append(_SURFACE_CI_GATE)

    # Deterministic order: locals (sorted by name, prefixed) then globals
    # in canonical order. Globals already follow the canonical ordering
    # because the appends above are in that order.
    failed_surfaces: tuple[str, ...] = tuple(
        f"local:{name}" for name in sorted(failed_locals)
    ) + tuple(failed_globals)

    status, reason = _classify_global(
        has_local_failure=has_local_failure,
        claim_ok=input_.claim_ledger_ok,
        deps_ok=input_.dependency_truth_ok,
        invar_ok=input_.invariant_coverage_ok,
        ci_ok=input_.ci_gate_ok,
        required=required,
    )

    evidence = MappingProxyType(
        dict(
            {
                "local_count": len(input_.local_witnesses),
                "local_pass_count": pass_count,
                "local_fail_count": len(failed_locals),
                "required_surfaces": tuple(sorted(required)),
                "failed_surfaces": failed_surfaces,
                "claim_ledger_ok": input_.claim_ledger_ok,
                "dependency_truth_ok": input_.dependency_truth_ok,
                "invariant_coverage_ok": input_.invariant_coverage_ok,
                "ci_gate_ok": input_.ci_gate_ok,
            }
        )
    )

    return GlobalParityWitness(
        globally_consistent=(status == "GLOBAL_PASS"),
        status=status,
        failed_surfaces=failed_surfaces,
        local_pass_count=pass_count,
        local_fail_count=len(failed_locals),
        reason=reason,
        falsifier=_FALSIFIER_TEXT,
        evidence_fields=evidence,
    )
