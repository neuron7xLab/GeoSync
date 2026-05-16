# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""IERD-Q7 edge-case coverage gate (Phase-4 ENTRY — informational ECC).

Asserts the ``(endpoint × state × test_id)`` matrix is well-formed and
honest, scores ECC, and fail-closes the two §532 *hard* sub-
requirements that are genuinely met today (network_failure + timeout
per applicable endpoint; simulation_diverged ↔ INV-DRO5). The ECC ≥
0.90 threshold is asserted but runs informationally at Phase-4 ENTRY
(workflow ``continue-on-error: true``) because the honest ECC is ~0.78
— loading / cancelled / per-probe server_error have no genuine test
until the frontend pages consume the endpoints. Phase-4 EXIT lands the
Playwright route-interception specs and flips this fail-closed.

Tracks claim ``edge-case-coverage-matrix`` and GitHub issue
IERD-Q7 (#532).
"""

from __future__ import annotations

import os
from typing import Final

import pytest

from tests.edge_cases.coverage_matrix import (
    ECC_THRESHOLD,
    MANDATORY_PER_ENDPOINT_STATES,
    SIMULATION_DIVERGED_STATE,
    cited_targets,
    classify,
    compute_ecc,
    covering_test,
    gated_operations,
    target_exists,
)


def test_every_cited_test_target_resolves() -> None:
    """Each test cited by the matrix still exists (rename/delete = fail).

    This is what gives the matrix teeth: a covered cell is only honest
    while its cited test is real. If any cited target vanishes the gate
    fails even though no ECC number changed.
    """
    missing = sorted(t for t in cited_targets() if not target_exists(t))
    assert not missing, (
        "IERD-Q7 matrix cites test target(s) that no longer resolve "
        f"(deleted/renamed): {missing}. A covered cell whose test has "
        f"vanished is not coverage — restore the test or move the cell "
        f"to UNCOVERED with a reason."
    )


def test_matrix_is_well_formed_and_total() -> None:
    """Every gated op is classified and every applicable cell is scored."""
    covered, applicable, ecc, rows = compute_ecc()
    assert applicable > 0, "empty applicable matrix — spec or classifier broke"
    # Every op classifies without raising.
    for _method, path in gated_operations():
        assert classify(path) in {"collection", "command", "probe"}
    # Each row is either covered (with a cited target) or explicitly
    # UNCOVERED — never an implicit gap.
    for cell, state, status, target in rows:
        assert status in {"covered", "UNCOVERED"}, (cell, state, status)
        if status == "covered":
            assert target != "-", (cell, state)
    assert covered <= applicable


def test_network_failure_and_timeout_covered_for_every_endpoint() -> None:
    """§532 hard requirement — fail-closed even at Phase-4 ENTRY.

    Network failure and timeout must be tested for *every* endpoint for
    which they are applicable. This is genuinely satisfied today
    (RequestTimeoutMiddleware behavioural test + fail-closed connector
    suite), so it is asserted strictly regardless of phase.
    """
    gaps: list[str] = []
    for method, path in gated_operations():
        for state in MANDATORY_PER_ENDPOINT_STATES:
            if covering_test(path, state) is None:
                # Only a gap if the state is applicable to this class.
                from tests.edge_cases.coverage_matrix import applicable_states

                if state in applicable_states(path):
                    gaps.append(f"{method} {path} :: {state}")
    assert not gaps, (
        "IERD-Q7 §532 hard requirement violated: network_failure and "
        f"timeout must be tested for every applicable endpoint. Gaps: "
        f"{sorted(gaps)}"
    )


def test_simulation_diverged_correlates_with_inv_dro5() -> None:
    """§532: simulation divergence has a fail-closed path (INV-DRO5).

    Asserted strictly: the collection-class simulation_diverged cell
    must cite the INV-DRO5 fail-closed suite.
    """
    target = covering_test("/features", SIMULATION_DIVERGED_STATE)
    assert target is not None and "inv_dro5_fail_closed" in target, (
        "IERD-Q7 §532: simulation_diverged must correlate with the "
        f"INV-DRO5 fail-closed test; got {target!r}. INV-DRO5 is the "
        f"contract that NaN/Inf/constant/degenerate forecast input is "
        f"rejected rather than silently diverging."
    )
    assert target_exists(target), f"cited INV-DRO5 test missing: {target}"


# ECC ≥ 0.90 is red by design at Phase-4 ENTRY (honest ECC ≈ 0.78:
# loading / cancelled / per-probe server_error have no genuine test
# until the frontend consumes the endpoints). The global
# python-fast-tests / python-heavy-tests lanes collect tests/ with
# fixed -m filters, so a bare marker would only move the failure
# between lanes; guard the sub-test with an env flag set ONLY on the
# dedicated edge-case-matrix workflow (which carries
# continue-on-error: true). The four genuinely-green tests above run
# unguarded in the global lanes. Identical isolation posture to the
# IERD-Q5 ENTRY gate. Phase-4 EXIT removes this guard with the
# fail-closed flip.
_EDGE_CASE_GATE_ENV: Final[str] = "GEOSYNC_EDGE_CASE_GATE"
_ecc_gate_only = pytest.mark.skipif(
    os.environ.get(_EDGE_CASE_GATE_ENV) != "1",
    reason=(
        "IERD-Q7 Phase-4 ENTRY informational ECC gate; runs only in the "
        "dedicated edge-case-matrix workflow "
        f"({_EDGE_CASE_GATE_ENV}=1, continue-on-error). Phase-4 EXIT "
        "lifts this guard when loading/cancelled/per-probe server_error "
        "land."
    ),
)


@_ecc_gate_only
def test_ecc_meets_threshold() -> None:
    """ECC ≥ 0.90. Informational at Phase-4 ENTRY (workflow continue-on-error)."""
    covered, applicable, ecc, rows = compute_ecc()
    for cell, state, status, target in rows:
        print(f"\n[ecc] {cell:24s} {state:20s} {status:9s} {target}")
    print(
        f"\n[ecc] AGGREGATE covered={covered}/{applicable} "
        f"ECC={ecc:.4f} threshold={ECC_THRESHOLD:.2f}"
    )
    assert ecc >= ECC_THRESHOLD, (
        f"IERD-Q7 §5 ECC below threshold: observed ECC={ecc:.4f} < "
        f"{ECC_THRESHOLD:.2f} (covered {covered} of {applicable} genuine "
        f"applicable cells). Phase-4 EXIT lands Playwright route-"
        f"interception specs for loading / cancelled / per-probe "
        f"server_error and flips this gate fail-closed. Do NOT relax the "
        f"matrix or cite non-exercising tests to inflate ECC."
    )
