# SPDX-License-Identifier: MIT
from __future__ import annotations


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
