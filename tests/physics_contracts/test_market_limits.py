# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Witnesses for ``physics_contracts/market_limits.yaml``.

These tests prove the market approximation-limits contract stays well-formed,
fail-closed, and aligned with the canonical claim-tier vocabulary in
``FORBIDDEN_CLAIMS.md`` and the four pre-registered nulls in the Ricci
microstructure preregistration guard. Each ``witness_required`` field in the
contract points at a function here; the cross-check test proves none dangles.
"""

from __future__ import annotations

import pytest

from tools.research.market_limits import (
    KNOWN_FALSIFIERS,
    MarketLimit,
    MarketLimitsContract,
    load_contract,
)

# Canonical claim-tier ceilings, ordered weakest → strongest. A limit's
# `max_claim_tier_violated` must be one of these (mirror of FORBIDDEN_CLAIMS.md).
CANONICAL_TIERS: frozenset[str] = frozenset(
    {
        "Not Measured",
        "Instrumented",
        "Measured-Single",
        "Measured-Multi",
        "Blocked",
        "Not Deployable",
    }
)


@pytest.fixture(scope="module")
def contract() -> MarketLimitsContract:
    return load_contract()


def test_contract_loads_and_is_nonempty(contract: MarketLimitsContract) -> None:
    assert contract.version >= 1
    assert contract.limits, "market-limits contract must declare at least one limit"
    ids = [limit.limit_id for limit in contract.limits]
    assert len(ids) == len(set(ids)), f"duplicate limit_id: {ids}"


def test_claim_tier_vocabulary_matches_forbidden_claims(
    contract: MarketLimitsContract,
) -> None:
    assert set(contract.claim_tiers) == CANONICAL_TIERS


@pytest.mark.parametrize(
    "field",
    ["valid_when", "invalid_when", "forbidden_claims"],
)
def test_list_fields_are_nonempty(contract: MarketLimitsContract, field: str) -> None:
    for limit in contract.limits:
        values = getattr(limit, field)
        assert values, f"{limit.limit_id}.{field} must be non-empty"
        assert all(v.strip() for v in values), f"{limit.limit_id}.{field} has blank entry"


def test_every_falsifier_is_a_known_null(contract: MarketLimitsContract) -> None:
    for limit in contract.limits:
        assert (
            limit.falsifier_required in KNOWN_FALSIFIERS
        ), f"{limit.limit_id} names unknown falsifier {limit.falsifier_required!r}"


def test_every_max_tier_is_canonical(contract: MarketLimitsContract) -> None:
    for limit in contract.limits:
        assert (
            limit.max_claim_tier_violated in CANONICAL_TIERS
        ), f"{limit.limit_id} ceiling {limit.max_claim_tier_violated!r} not canonical"


def test_no_witness_reference_dangles(contract: MarketLimitsContract) -> None:
    """Each ``witness_required`` must point at a function defined in this module."""
    module_funcs = {name for name in globals() if name.startswith("test_")}
    for limit in contract.limits:
        ref = limit.witness_required
        assert "::" in ref, f"{limit.limit_id}.witness_required must be path::func, got {ref!r}"
        path, func = ref.split("::", 1)
        assert path == "tests/physics_contracts/test_market_limits.py", (
            f"{limit.limit_id} witness path {path!r} is not this module"
        )
        assert func in module_funcs, f"{limit.limit_id} witness {func!r} is not defined here"


# ── Named witnesses referenced by the contract's `witness_required` fields ──
# Each asserts the specific fail-closed property the limit promises.


def _limit(contract: MarketLimitsContract, limit_id: str) -> MarketLimit:
    found = contract.by_id(limit_id)
    assert found is not None, f"missing limit {limit_id!r}"
    return found


def test_no_conservation_claim(contract: MarketLimitsContract) -> None:
    limit = _limit(contract, "market.non_conservative")
    assert limit.max_claim_tier_violated == "Instrumented"
    assert any("conserv" in c.lower() for c in limit.forbidden_claims)


def test_window_is_declared(contract: MarketLimitsContract) -> None:
    limit = _limit(contract, "market.non_stationary")
    assert limit.falsifier_required == "lag_sweep_no_future_data"


def test_no_fixed_coupling_claim(contract: MarketLimitsContract) -> None:
    limit = _limit(contract, "market.agent_driven")
    assert any("coupling" in c.lower() for c in limit.forbidden_claims)


def test_degenerate_book_fails_closed(contract: MarketLimitsContract) -> None:
    limit = _limit(contract, "market.liquidity_discontinuous")
    # A degenerate book must not merely degrade — it must block promotion.
    assert limit.max_claim_tier_violated == "Blocked"


def test_shock_calm_stratified(contract: MarketLimitsContract) -> None:
    limit = _limit(contract, "market.regime_shock")
    assert limit.falsifier_required == "multi_session_replay"
    # Fail-closed: an invalid shock-regime interpretation must not retain a
    # real-data measured tier — it caps at Instrumented, never Measured-*.
    assert limit.max_claim_tier_violated == "Instrumented"
    assert "Measured" not in limit.max_claim_tier_violated


def test_open_system_acknowledged(contract: MarketLimitsContract) -> None:
    limit = _limit(contract, "market.order_flow_open_system")
    assert any("closed" in c.lower() for c in limit.forbidden_claims)


def test_no_causal_from_correlation(contract: MarketLimitsContract) -> None:
    limit = _limit(contract, "market.correlation_not_causal")
    assert any("causal" in c.lower() for c in limit.forbidden_claims)


def test_descriptor_needs_falsifier(contract: MarketLimitsContract) -> None:
    limit = _limit(contract, "market.descriptor_not_predictor")
    # The descriptor→predictor jump must be gated by a multi-session falsifier.
    assert limit.falsifier_required == "multi_session_replay"
    assert limit.max_claim_tier_violated == "Instrumented"
