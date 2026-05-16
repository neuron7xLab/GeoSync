# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""UX state-coverage gate (IERD-Q5 Phase-4 ENTRY).

IERD-PAI-FPS-UX-001 §5 requires every public endpoint to declare the
six UX states the frontend must be able to render:

    success  empty  partial  validation_error  server_error  timeout

The readiness score is

    UXRS = (declared states across endpoints) / (6 × endpoints)

and §5 demands UXRS ≥ 0.95.

This module asserts the contract at the OpenAPI layer. It reads the
frozen, versioned spec at
``schemas/openapi/geosync-online-inference-v1.json`` (the same
single-source-of-truth the Q4 schemathesis gate validates the running
app against) and maps each declared HTTP status code to a UX state via
the HTTP-canonical mapping below. An endpoint "declares" a state when
its ``responses:`` block carries at least one status code mapped to
that state.

Two independent contracts are checked:

* ``test_uxrs_meets_threshold`` — the aggregate UXRS over all public
  operations is ≥ ``UXRS_THRESHOLD``.
* ``test_error_envelope_on_all_4xx_5xx`` — every declared 4xx/5xx
  response body ``$ref``\\s the standard ``ErrorResponse`` envelope
  (``{error: {code, message, path, meta}}``), never an ad-hoc shape.

``/graphql`` is excluded by design, identical to the Q4 gate: GraphQL
carries its own typed Strawberry schema and is tracked under a separate
contract suite.

Phase-4 ENTRY: the workflow runs this suite with
``continue-on-error: true`` so the remaining acceptance criteria
(frontend rendering for every state, end-to-end matrix test) can land
under the same claim without flipping gate strictness mid-phase.
Phase-4 EXIT removes that and makes the gate fail-closed, at which
point the claim re-classifies ANCHORED.

Tracks claim ``ux-readiness-state-coverage`` in ``docs/CLAIMS.yaml``
and GitHub issue IERD-Q5 (#530).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Final

import pytest

# Frozen, versioned OpenAPI 3.1 spec — the same artifact the Q4
# schemathesis gate fuzzes the live app against. Reading the persisted
# file (rather than building the app) keeps this a pure contract gate:
# deterministic, env-free, and decoupled from runtime wiring.
_SPEC_PATH: Final[Path] = (
    Path(__file__).resolve().parents[2] / "schemas" / "openapi" / "geosync-online-inference-v1.json"
)

# §5 readiness threshold. Override via env only for shadow runs; the
# IERD number is verbatim here.
UXRS_THRESHOLD: Final[float] = 0.95

# The six required UX states, in §5 order.
REQUIRED_STATES: Final[tuple[str, ...]] = (
    "success",
    "empty",
    "partial",
    "validation_error",
    "server_error",
    "timeout",
)

# HTTP-canonical state → status-code mapping. A state counts as
# declared for an operation if any of its codes appears in the
# operation's ``responses:`` keys. The mapping is intentionally
# standards-aligned (RFC 9110) rather than bespoke so the contract is
# auditable without reading handler code:
#
#   success           200 Created/OK family
#   empty             204 No Content
#   partial           206 Partial Content
#   validation_error  400 Bad Request / 422 Unprocessable Entity
#   server_error      500 / 502 / 503 server-side failure
#   timeout           504 Gateway Timeout
_STATE_CODES: Final[dict[str, frozenset[str]]] = {
    "success": frozenset({"200", "201"}),
    "empty": frozenset({"204"}),
    "partial": frozenset({"206"}),
    "validation_error": frozenset({"400", "422"}),
    "server_error": frozenset({"500", "502", "503"}),
    "timeout": frozenset({"504"}),
}

# Excluded by design — GraphQL has its own typed schema (Strawberry);
# identical carve-out to tests/api/test_schemathesis_contract.py.
_EXCLUDE_PATH_RE: Final[re.Pattern[str]] = re.compile(r"^/graphql")

_HTTP_METHODS: Final[frozenset[str]] = frozenset(
    {"get", "post", "put", "delete", "patch", "options", "head"}
)


def _load_spec() -> dict[str, Any]:
    """Load the frozen OpenAPI document."""
    with _SPEC_PATH.open(encoding="utf-8") as handle:
        spec: dict[str, Any] = json.load(handle)
    return spec


def _public_operations(spec: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """Return [(method, path, operation)] for every gated public operation."""
    operations: list[tuple[str, str, dict[str, Any]]] = []
    for path, path_item in (spec.get("paths") or {}).items():
        if _EXCLUDE_PATH_RE.match(path):
            continue
        for method, operation in path_item.items():
            if method.lower() in _HTTP_METHODS:
                operations.append((method.upper(), path, operation))
    return operations


def _declared_states(operation: dict[str, Any]) -> set[str]:
    """The subset of REQUIRED_STATES this operation declares in responses."""
    codes = {str(code) for code in (operation.get("responses") or {})}
    return {state for state, state_codes in _STATE_CODES.items() if codes & state_codes}


def test_spec_present_and_has_public_operations() -> None:
    """The frozen spec exists and exposes at least one gated operation."""
    assert _SPEC_PATH.is_file(), (
        f"IERD-Q5 §5 gate cannot run: OpenAPI spec missing at {_SPEC_PATH}. "
        f"The UXRS contract is defined against the frozen versioned spec, "
        f"not the live app — restore schemas/openapi/ before this gate can "
        f"score state coverage."
    )
    operations = _public_operations(_load_spec())
    assert operations, (
        "IERD-Q5 §5 gate found zero public operations after the /graphql "
        "carve-out — the spec is empty or every path was excluded; UXRS is "
        "undefined with a zero denominator."
    )


# The UXRS sub-test is red by design at Phase-4 ENTRY (the spec is
# missing empty/partial/timeout declarations — that is precisely what
# the gate surfaces). It must therefore run ONLY in the dedicated
# ux-state-coverage workflow, which sets GEOSYNC_UX_STATE_GATE=1 and
# carries continue-on-error: true on its run step. In the global
# python-fast-tests / python-heavy-tests lanes the flag is absent so
# the test SKIPs — identical posture to the Q4 schemathesis gate,
# which likewise runs only in its own workflow (there via an optional
# dependency skip). The two genuinely-green tests stay unguarded and
# provide real coverage in the global lanes. Phase-4 EXIT removes
# this guard together with the fail-closed flip.
_UX_STATE_GATE_ENV: Final[str] = "GEOSYNC_UX_STATE_GATE"
_uxrs_gate_only = pytest.mark.skipif(
    os.environ.get(_UX_STATE_GATE_ENV) != "1",
    reason=(
        "IERD-Q5 Phase-4 ENTRY informational UXRS gate; runs only in the "
        "dedicated ux-state-coverage workflow "
        f"({_UX_STATE_GATE_ENV}=1, continue-on-error). Phase-4 EXIT lifts "
        "this guard when the missing state declarations land."
    ),
)


@_uxrs_gate_only
def test_uxrs_meets_threshold() -> None:
    """Aggregate UXRS over public operations is ≥ the §5 threshold."""
    operations = _public_operations(_load_spec())
    denominator = len(REQUIRED_STATES) * len(operations)
    declared_total = 0
    for method, path, operation in operations:
        declared = _declared_states(operation)
        declared_total += len(declared)
        missing = sorted(set(REQUIRED_STATES) - declared)
        print(  # surfaced via pytest -s / Step Summary
            f"\n[uxrs] {method:6s} {path:30s} "
            f"declared={len(declared)}/{len(REQUIRED_STATES)} "
            f"missing={','.join(missing) if missing else '-'}"
        )

    uxrs = declared_total / denominator if denominator else 0.0
    print(  # aggregate line consumed by the Step Summary
        f"\n[uxrs] AGGREGATE "
        f"declared={declared_total}/{denominator} "
        f"UXRS={uxrs:.4f} threshold={UXRS_THRESHOLD:.2f} "
        f"endpoints={len(operations)}"
    )

    assert uxrs >= UXRS_THRESHOLD, (
        f"IERD-Q5 §5 UXRS violated: observed UXRS={uxrs:.4f} below "
        f"threshold {UXRS_THRESHOLD:.2f} "
        f"(declared {declared_total} of {denominator} state cells across "
        f"{len(operations)} public operations). Each operation must declare "
        f"all six UX states {REQUIRED_STATES} via responses: keys mapped by "
        f"{_STATE_CODES}. Phase-4 EXIT remediation adds the missing "
        f"empty/partial/timeout declarations and flips this gate fail-closed."
    )


def test_error_envelope_on_all_4xx_5xx() -> None:
    """Every declared 4xx/5xx response body $refs the standard envelope.

    The §5 contract demands a single canonical error envelope
    (``{error: {code, message, path, meta}}``) on all failure
    responses. We assert structurally that each 4xx/5xx response's
    ``application/json`` body resolves to ``#/components/schemas/
    ErrorResponse`` — never an ad-hoc inline shape — so the frontend
    can render every failure state through one renderer.
    """
    spec = _load_spec()
    expected_ref = "#/components/schemas/ErrorResponse"
    offenders: list[str] = []

    for method, path, operation in _public_operations(spec):
        for code, response in (operation.get("responses") or {}).items():
            if not re.fullmatch(r"[45]\d\d", str(code)):
                continue
            schema = (response.get("content") or {}).get("application/json", {}).get("schema", {})
            if schema.get("$ref") != expected_ref:
                offenders.append(
                    f"{method} {path} [{code}] -> {schema or 'no application/json body'}"
                )

    assert not offenders, (
        f"IERD-Q5 §5 error-envelope contract violated: "
        f"{len(offenders)} declared 4xx/5xx response(s) do not $ref the "
        f"canonical {expected_ref} envelope. Every failure response must "
        f"resolve to the standard {{error: {{code, message, path, meta}}}} "
        f"shape so the frontend renders all failure states through one "
        f"path. Offenders: " + "; ".join(sorted(offenders))
    )
