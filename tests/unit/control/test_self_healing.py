# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Tests for deterministic AAR-PRO self-healing plans."""

from __future__ import annotations

from geosync_hpc.control import (
    ActionResultStatus,
    ExpectedResultModel,
    ObservedActionResult,
    RecoveryAction,
    accept_action_result,
    estimate_entropy_risk,
    prescribe_recovery,
)


def _expected(*, error_threshold: float = 0.1, rollback_threshold: float = 1.0) -> ExpectedResultModel:
    return ExpectedResultModel(
        action_id="heal-1",
        action_type="trade",
        expected_result=(0.0,),
        expected_result_variance=(1.0,),
        context_signature=(1.0,),
        model_created_seq=1,
        action_started_seq=2,
        error_threshold=error_threshold,
        rollback_threshold=rollback_threshold,
    )


def _observed(value: float, *, action_id: str = "heal-1", observed_seq: int = 3) -> ObservedActionResult:
    return ObservedActionResult(
        action_id=action_id,
        observed_seq=observed_seq,
        observed_result=(value,),
    )


def test_sanctioned_match_allows_model_update() -> None:
    witness = accept_action_result(_expected(), _observed(0.0))

    plan = prescribe_recovery(witness)

    assert witness.status is ActionResultStatus.SANCTIONED_MATCH
    assert plan.model_update_allowed is True
    assert plan.actions == (RecoveryAction.ALLOW_MODEL_UPDATE,)
    assert plan.entropy_risk == 0.0


def test_rollback_blocks_update_and_requests_action_rollback() -> None:
    witness = accept_action_result(_expected(), _observed(2.0))

    plan = prescribe_recovery(witness)

    assert witness.status is ActionResultStatus.ROLLBACK_REQUIRED
    assert plan.model_update_allowed is False
    assert RecoveryAction.BLOCK_WEIGHT_UPDATE in plan.actions
    assert RecoveryAction.ROLLBACK_ACTION in plan.actions
    assert RecoveryAction.REDUCE_RISK in plan.actions
    assert plan.entropy_risk >= 0.75


def test_invalid_or_mismatched_witness_reseals_expected_model() -> None:
    witness = accept_action_result(_expected(), _observed(0.0, action_id="forged"))

    plan = prescribe_recovery(witness)

    assert witness.status is ActionResultStatus.ACTION_MISMATCH
    assert RecoveryAction.RESEAL_EXPECTED_MODEL in plan.actions
    assert RecoveryAction.BLOCK_WEIGHT_UPDATE in plan.actions
    assert plan.model_update_allowed is False


def test_chain_or_free_energy_breach_quarantines_and_reduces_risk() -> None:
    witness = accept_action_result(_expected(rollback_threshold=10.0), _observed(0.2))

    base_risk = estimate_entropy_risk(witness)
    breached = prescribe_recovery(witness, free_energy_circuit_breaker=True, chain_verified=False)

    assert breached.entropy_risk > base_risk
    assert RecoveryAction.QUARANTINE_EPISODE in breached.actions
    assert RecoveryAction.REDUCE_RISK in breached.actions
    assert breached.model_update_allowed is False
