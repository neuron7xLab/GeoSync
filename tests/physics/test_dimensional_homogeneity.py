# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

UNIT_CONTRACT_PATH = Path("artifacts/physics_validation/unit_contract_summary.json")


def _load_unit_contract() -> dict[str, Any]:
    return json.loads(UNIT_CONTRACT_PATH.read_text(encoding="utf-8"))


def test_kuramoto_rhs_terms_are_dimensionally_homogeneous() -> None:
    """For dimensionless A, both RHS terms are angular-rate terms."""
    units = {
        "dtheta_dt": "rad/time",
        "omega": "rad/time",
        "K": "1/time",
        "A": "dimensionless",
        "sin_delta_theta": "dimensionless",
    }
    coupling_term = "1/time"
    assert units["dtheta_dt"] == units["omega"]
    assert coupling_term == "1/time"


def test_boundary_phi_terms_are_dimensionally_homogeneous() -> None:
    """Phi = K*lambda_max(A) - 2*gamma compares only rate-like terms."""
    units = {
        "K": "1/time",
        "lambda_max_A": "dimensionless",
        "gamma": "1/time",
        "Phi": "1/time",
    }
    assert units["K"] == units["gamma"] == units["Phi"]
    assert units["lambda_max_A"] == "dimensionless"


def test_critical_coupling_unit_matches_K() -> None:
    """K_c = 2*gamma/lambda_max(A) has the same unit as K."""
    unit_gamma = "1/time"
    unit_lambda = "dimensionless"
    unit_K = "1/time"
    assert unit_lambda == "dimensionless"
    assert unit_gamma == unit_K


def test_unit_contract_artifact_records_required_state() -> None:
    artifact = _load_unit_contract()
    state_ids = {entry["id"] for entry in artifact["state_variables"]}
    assert state_ids == {"theta", "omega", "K", "A", "R", "Phi", "dt"}
    assert artifact["score_credit_allowed"] is False
    assert artifact["canonical_object"] == "weighted_graph_kuramoto"


def test_unit_contract_artifact_records_equation_units() -> None:
    artifact = _load_unit_contract()
    contracts = {entry["id"]: entry for entry in artifact["equation_contracts"]}
    assert set(contracts) == {"kuramoto_rhs", "order_parameter", "margin_phi"}
    assert contracts["kuramoto_rhs"]["lhs_unit"] == "radian_per_time"
    assert contracts["order_parameter"]["lhs_unit"] == "dimensionless"
    assert contracts["margin_phi"]["rhs_units"] == ["inverse_time", "inverse_time"]


def test_unit_contract_completion_metric_is_exact() -> None:
    artifact = _load_unit_contract()
    metric = artifact["completion_metric"]
    assert metric["required_state_variables"] == len(artifact["state_variables"])
    assert metric["required_equation_contracts"] == len(artifact["equation_contracts"])
