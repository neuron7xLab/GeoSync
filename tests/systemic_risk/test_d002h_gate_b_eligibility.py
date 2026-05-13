# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""D-002H Gate B — eligibility artifact schema + invariant tests.

These are fast JSON/schema checks. The heavy eligibility computation is
performed by ``scripts/x10r_d002h_gate_b_eligibility.py`` and lands in
``artifacts/d002h/eligibility/d002h_ricci_eligibility.json``; these tests
LOAD the pre-computed artifact and verify schema + invariants only — no
recomputation, no slow marker required.

Gate B is one term of the 7-gate authorisation conjunction
(A ∧ B ∧ C ∧ D ∧ E ∧ F ∧ G). PASS of Gate B alone does NOT authorise
canonical run; these tests do NOT make such a claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_RELPATH = "artifacts/d002h/eligibility/d002h_ricci_eligibility.json"
ARTIFACT_PATH = REPO_ROOT / ARTIFACT_RELPATH

# D-002C claim ledger anchor — byte-exact sha256 of
# ``docs/governance/D002C_CLAIM_LEDGER.yaml`` pinned at the D-002C
# canonical-run-report merge. Inline pragma silences detect-secrets
# HexHighEntropy; this is a content-addressed governance anchor, not
# a credential. fmt:off keeps the long literal on one line for review.
# fmt: off
D002C_LEDGER_SHA256: str = "f96ba9b5a2057d2e0bff84afc28578ab316cff73f6dc6673fb0d6d543b8bd6dd"  # noqa: E501  # pragma: allowlist secret
# fmt: on

D002C_LEDGER_RELPATH = "docs/governance/D002C_CLAIM_LEDGER.yaml"

# D-002H prereg grid — must match
# ``docs/governance/D002H_PREREGISTRATION.yaml`` canonical_grid block.
EXPECTED_N: list[int] = [50, 100, 200]
EXPECTED_LAMBDA: list[float] = [0.0, 0.05, 0.10, 0.20, 0.40, 1.0]
EXPECTED_N_CELLS: int = 18

NA_LAMBDA_ZERO: str = "N/A_M3_REQUIRES_LAMBDA_GT_ZERO"


def _load_artifact() -> dict[str, Any]:
    raw = ARTIFACT_PATH.read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert isinstance(payload, dict)
    return payload


def test_gate_b_artifact_exists() -> None:
    """The Gate B artifact exists on disk and loads as a JSON object."""
    assert ARTIFACT_PATH.is_file(), (
        f"Gate B artifact missing at {ARTIFACT_RELPATH}; "
        "run scripts/x10r_d002h_gate_b_eligibility.py"
    )
    payload = _load_artifact()
    assert isinstance(payload, dict)
    assert "cells" in payload
    assert isinstance(payload["cells"], list)


def test_gate_b_schema_version() -> None:
    """The artifact carries the locked schema_version 'D002H-GATE-B-v1'."""
    payload = _load_artifact()
    assert payload["schema_version"] == "D002H-GATE-B-v1"
    assert payload["study_id"] == "D-002H"
    assert payload["gate"] == "B"


def test_gate_b_substrate_scope_is_ricci_flow_only() -> None:
    """substrate_scope is exactly ['ricci_flow']; no other substrate touched."""
    payload = _load_artifact()
    assert payload["substrate_scope"] == ["ricci_flow"]
    for cell in payload["cells"]:
        assert cell["substrate_id"] == "ricci_flow"


def test_gate_b_grid_matches_d002h_prereg() -> None:
    """grid.N and grid.lambda_values match the D-002H prereg canonical grid."""
    payload = _load_artifact()
    grid = payload["grid"]
    assert grid["N"] == EXPECTED_N
    assert grid["lambda_values"] == EXPECTED_LAMBDA
    assert grid["base_seed"] == 42
    assert grid["null_seed_M3"] == 12345


def test_gate_b_cells_count_is_18() -> None:
    """The locked grid is 3 N × 6 λ = 18 cells. No drift allowed."""
    payload = _load_artifact()
    assert payload["n_cells_evaluated"] == EXPECTED_N_CELLS
    assert len(payload["cells"]) == EXPECTED_N_CELLS


def test_gate_b_lambda_zero_m3_is_na_literal() -> None:
    """Every λ==0 cell reports the locked N/A_M3 literal (M3 contract)."""
    payload = _load_artifact()
    lambda_zero_cells = [c for c in payload["cells"] if float(c["lambda_value"]) == 0.0]
    n_zero = len(lambda_zero_cells)
    msg_count = f"expected exactly 3 λ=0 cells; got {n_zero}"
    assert n_zero == 3, msg_count
    for cell in lambda_zero_cells:
        cell_n = cell["N"]
        cell_m3 = cell["m3_status"]
        msg_status = f"λ=0 cell N={cell_n} m3_status={cell_m3!r} expected {NA_LAMBDA_ZERO!r}"
        assert cell_m3 == NA_LAMBDA_ZERO, msg_status
        assert cell["m3_match_report"] is None
        assert cell["m3_summary_sha256"] is None


def test_gate_b_lambda_pos_cells_have_eligible_m1_and_m3() -> None:
    """Every λ>0 cell on ricci_flow is ELIGIBLE_M1 AND ELIGIBLE_M3."""
    payload = _load_artifact()
    lambda_pos_cells = [c for c in payload["cells"] if float(c["lambda_value"]) > 0.0]
    n_pos = len(lambda_pos_cells)
    msg_count = f"expected exactly 15 λ>0 cells; got {n_pos}"
    assert n_pos == 15, msg_count
    for cell in lambda_pos_cells:
        m1: str = cell["m1_status"]
        m3: str = cell["m3_status"]
        cell_n = cell["N"]
        cell_lam = cell["lambda_value"]
        msg_m1 = (
            f"λ>0 cell N={cell_n} λ={cell_lam} m1_status={m1!r} does not start with 'ELIGIBLE_M1'"
        )
        msg_m3 = (
            f"λ>0 cell N={cell_n} λ={cell_lam} m3_status={m3!r} does not start with 'ELIGIBLE_M3'"
        )
        assert m1.startswith("ELIGIBLE_M1"), msg_m1
        assert m3.startswith("ELIGIBLE_M3"), msg_m3
        assert cell["m3_match_report"] is not None
        assert cell["m3_match_report"]["all_within_tolerance"] is True
        assert cell["cell_gate_b_pass"] is True


def test_gate_b_canonical_run_authorized_is_false() -> None:
    """Gate B PASS does NOT authorise canonical run — conjunction needed."""
    payload = _load_artifact()
    assert payload["canonical_run_authorized"] is False
    assert payload["d002c_ledger_touched"] is False
    assert payload["substrate_code_edited"] is False
    assert payload["mechanism_code_edited"] is False


def test_gate_b_downstream_gates_remaining_lists_c_through_g() -> None:
    """Gates C, D, E, F, G remain open after this PR; only Gate B closes here."""
    payload = _load_artifact()
    assert set(payload["downstream_gates_remaining"]) == {
        "C",
        "D",
        "E",
        "F",
        "G",
    }


def test_gate_b_preserves_d002c_ledger() -> None:
    """D-002C claim ledger sha256 is byte-exact UNCHANGED at the locked anchor."""
    ledger_path = REPO_ROOT / D002C_LEDGER_RELPATH
    assert ledger_path.is_file(), f"D-002C ledger missing at {D002C_LEDGER_RELPATH}"
    actual = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    assert actual == D002C_LEDGER_SHA256, (
        f"D-002C claim ledger MUTATED: expected {D002C_LEDGER_SHA256}, "
        f"got {actual}. This PR (D-002H Gate B) is forbidden from "
        "touching the D-002C claim ledger."
    )
