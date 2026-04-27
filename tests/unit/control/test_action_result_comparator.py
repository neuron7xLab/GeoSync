# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Tests for ``geosync_hpc.control.action_result_comparator``.

Falsifier probes (mutation -> test that must catch it). Probes are
documented here, not run via git mutation in this PR:

    Probe 1: Drop the ``model_created_seq < action_started_seq`` check
             in ``_validate_expected``.
             -> ``test_02_model_seq_must_be_strictly_less_than_action_seq``
    Probe 2: Relax ``observed_seq > action_started_seq`` to ``>=``.
             -> ``test_03_observed_seq_must_be_strictly_greater_than_action_seq``
    Probe 3: Replace inverse-variance weighting with a constant or
             confidence factor in ``_precision_weighted_euclidean``.
             -> ``test_14_precision_weighted_error_uses_inverse_variance``
    Probe 4: Allow ``comparator_error == rollback_threshold`` to fall
             through to UPDATE_REQUIRED instead of ROLLBACK_REQUIRED.
             -> ``test_33_comparator_error_at_rollback_threshold_is_rollback``
    Probe 5: Allow zero or negative variance to be accepted by
             ``_validate_expected``.
             -> ``test_19_zero_variance_rejected`` /
                ``test_20_negative_variance_rejected``
    Probe 6: Drop the ``reverse_afferentation_present=False`` short
             circuit so logged observations sanction the action.
             -> ``test_07_missing_reverse_afferentation``
    Probe 7: Allow ``success=True`` to override comparator_error and
             return SANCTIONED_MATCH on a breach.
             -> ``test_31_success_true_cannot_force_sanctioned_on_breach``
"""

from __future__ import annotations

import dataclasses
import inspect
import math

import pytest

from geosync_hpc import control as control_pkg
from geosync_hpc.control import (
    ActionResultStatus,
    ActionResultWitness,
    ExpectedResultModel,
    ObservedActionResult,
    accept_action_result,
)
from geosync_hpc.control import action_result_comparator as module_under_test

# --------------------------------------------------------------------------- #
# Helpers (deterministic builders only — no fixtures with state).             #
# --------------------------------------------------------------------------- #


def _make_expected(
    *,
    action_id: str = "act-1",
    expected_result: tuple[float, ...] = (1.0, 0.0, -1.0),
    expected_result_variance: tuple[float, ...] | None = None,
    error_threshold: float = 0.5,
    rollback_threshold: float = 1.0,
    expected_value: float | None = None,
    expected_latency_ms: float | None = None,
    predicted_action_signature: tuple[float, ...] | None = None,
    action_mismatch_threshold: float | None = None,
    value_mismatch_threshold: float | None = None,
    timing_mismatch_threshold_ms: float | None = None,
    model_created_seq: int = 1,
    action_started_seq: int = 2,
) -> ExpectedResultModel:
    return ExpectedResultModel(
        action_id=action_id,
        action_type="trade",
        expected_result=expected_result,
        expected_result_variance=expected_result_variance,
        context_signature=(0.1, 0.2, 0.3),
        model_created_seq=model_created_seq,
        action_started_seq=action_started_seq,
        error_threshold=error_threshold,
        rollback_threshold=rollback_threshold,
        expected_value=expected_value,
        expected_latency_ms=expected_latency_ms,
        predicted_action_signature=predicted_action_signature,
        action_mismatch_threshold=action_mismatch_threshold,
        value_mismatch_threshold=value_mismatch_threshold,
        timing_mismatch_threshold_ms=timing_mismatch_threshold_ms,
    )


def _make_observed(
    *,
    action_id: str = "act-1",
    observed_result: tuple[float, ...] | None = (1.0, 0.0, -1.0),
    observed_value: float | None = None,
    observed_latency_ms: float | None = None,
    executed_action_signature: tuple[float, ...] | None = None,
    success: bool | None = None,
    reverse_afferentation_present: bool = True,
    observed_seq: int = 3,
) -> ObservedActionResult:
    return ObservedActionResult(
        action_id=action_id,
        observed_seq=observed_seq,
        observed_result=observed_result,
        observed_value=observed_value,
        observed_latency_ms=observed_latency_ms,
        executed_action_signature=executed_action_signature,
        success=success,
        reverse_afferentation_present=reverse_afferentation_present,
    )


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_01_missing_expected_returns_invalid_input() -> None:
    """Missing expected model fails closed: no fabricated default."""
    observed = _make_observed()
    witness = accept_action_result(None, observed)
    assert witness.status is ActionResultStatus.INVALID_INPUT
    assert witness.accepted is False
    assert witness.reason.startswith("INVALID_EXPECTED_MODEL")


@pytest.mark.parametrize(
    ("created_seq", "started_seq"),
    [(2, 2), (3, 2)],
)
def test_02_model_seq_must_be_strictly_less_than_action_seq(
    created_seq: int, started_seq: int
) -> None:
    """`model_created_seq` must be strictly less than `action_started_seq`."""
    with pytest.raises(ValueError, match="SEQUENCE_ORDER_INVALID"):
        _make_expected(model_created_seq=created_seq, action_started_seq=started_seq)


@pytest.mark.parametrize(
    ("started_seq", "observed_seq"),
    [(5, 5), (5, 4)],
)
def test_03_observed_seq_must_be_strictly_greater_than_action_seq(
    started_seq: int, observed_seq: int
) -> None:
    """`observed_seq` must be strictly greater than `action_started_seq`."""
    expected = _make_expected(model_created_seq=1, action_started_seq=started_seq)
    observed = _make_observed(observed_seq=observed_seq)
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.INVALID_INPUT
    assert witness.reason.startswith("SEQUENCE_ORDER_INVALID")


def test_04_exact_match_returns_sanctioned_and_dissolved() -> None:
    expected = _make_expected()
    observed = _make_observed()
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.SANCTIONED_MATCH
    assert witness.accepted is True
    assert witness.dissolved is True
    assert witness.update_required is False
    assert witness.rollback_required is False
    assert witness.outcome_prediction_error == 0.0
    assert witness.comparator_error == 0.0


def test_05_medium_error_returns_update_required() -> None:
    expected = _make_expected(error_threshold=0.5, rollback_threshold=2.0)
    # OPE = sqrt(0.7^2) = 0.7  > 0.5 and < 2.0
    observed = _make_observed(observed_result=(1.7, 0.0, -1.0))
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.UPDATE_REQUIRED
    assert witness.update_required is True
    assert witness.rollback_required is False
    assert witness.next_context_expansion_required is True


def test_06_large_error_returns_rollback_required() -> None:
    expected = _make_expected(error_threshold=0.5, rollback_threshold=1.0)
    # OPE = sqrt(2.5^2) = 2.5 > rollback_threshold
    observed = _make_observed(observed_result=(3.5, 0.0, -1.0))
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.ROLLBACK_REQUIRED
    assert witness.rollback_required is True
    assert witness.update_required is False
    assert witness.inhibit_repetition is True


def test_07_missing_reverse_afferentation() -> None:
    expected = _make_expected()
    observed = _make_observed(reverse_afferentation_present=False)
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.INSUFFICIENT_REVERSE_AFFERENTATION
    assert witness.accepted is False
    assert witness.reason.startswith("MISSING_REVERSE_AFFERENTATION")


def test_08_missing_observed_result() -> None:
    expected = _make_expected()
    observed = _make_observed(observed_result=None)
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.INSUFFICIENT_OBSERVATION
    assert witness.accepted is False
    assert witness.reason.startswith("MISSING_OBSERVATION")


def test_09_action_id_mismatch_returns_action_mismatch() -> None:
    expected = _make_expected(action_id="alpha")
    observed = _make_observed(action_id="beta")
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.ACTION_MISMATCH
    assert witness.reason.startswith("ACTION_ID_MISMATCH")


def test_10_ape_breach_returns_action_mismatch() -> None:
    expected = _make_expected(
        error_threshold=10.0,
        rollback_threshold=10.0,
        predicted_action_signature=(0.0, 0.0),
        action_mismatch_threshold=0.5,
    )
    # APE = sqrt(1+1) ~ 1.414 > 0.5
    observed = _make_observed(executed_action_signature=(1.0, 1.0))
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.ACTION_MISMATCH
    assert witness.reason.startswith("ACTION_MISMATCH")


def test_11_rpe_breach_returns_value_mismatch() -> None:
    expected = _make_expected(
        error_threshold=10.0,
        rollback_threshold=10.0,
        expected_value=1.0,
        value_mismatch_threshold=0.25,
    )
    # |RPE| = |2.0 - 1.0| = 1.0 > 0.25
    observed = _make_observed(observed_value=2.0)
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.VALUE_MISMATCH
    assert witness.reason.startswith("VALUE_MISMATCH")


def test_12_tpe_breach_returns_timing_mismatch() -> None:
    expected = _make_expected(
        error_threshold=10.0,
        rollback_threshold=10.0,
        expected_latency_ms=100.0,
        timing_mismatch_threshold_ms=10.0,
    )
    # |TPE| = |200-100| = 100 > 10
    observed = _make_observed(observed_latency_ms=200.0)
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.TIMING_MISMATCH
    assert witness.reason.startswith("TIMING_MISMATCH")


def test_13_raw_ope_is_euclidean_norm() -> None:
    expected = _make_expected(
        expected_result=(0.0, 0.0, 0.0),
        error_threshold=10.0,
        rollback_threshold=10.0,
    )
    observed = _make_observed(observed_result=(3.0, 4.0, 0.0))
    witness = accept_action_result(expected, observed)
    assert witness.outcome_prediction_error is not None
    assert math.isclose(witness.outcome_prediction_error, 5.0, abs_tol=1e-12)


def test_14_precision_weighted_error_uses_inverse_variance() -> None:
    expected = _make_expected(
        expected_result=(0.0, 0.0),
        expected_result_variance=(1.0, 4.0),
        error_threshold=10.0,
        rollback_threshold=10.0,
    )
    # diff = (2.0, 2.0); precision = (1.0, 0.25)
    # weighted_squared = 1.0*4 + 0.25*4 = 5.0
    # weight_total     = 1.0 + 0.25 = 1.25
    # pwope = sqrt(5.0/1.25) = sqrt(4.0) = 2.0
    observed = _make_observed(observed_result=(2.0, 2.0))
    witness = accept_action_result(expected, observed)
    assert witness.precision_weighted_outcome_error is not None
    assert math.isclose(witness.precision_weighted_outcome_error, 2.0, abs_tol=1e-12)


def test_15_comparator_error_uses_pwope_when_variance_present() -> None:
    expected = _make_expected(
        expected_result=(0.0, 0.0),
        expected_result_variance=(1.0, 4.0),
        error_threshold=10.0,
        rollback_threshold=10.0,
    )
    observed = _make_observed(observed_result=(2.0, 2.0))
    witness = accept_action_result(expected, observed)
    assert witness.precision_weighted_outcome_error is not None
    assert witness.comparator_error is not None
    assert math.isclose(
        witness.comparator_error, witness.precision_weighted_outcome_error, abs_tol=1e-12
    )


def test_16_comparator_error_uses_raw_ope_when_variance_none() -> None:
    expected = _make_expected(
        expected_result=(0.0, 0.0),
        expected_result_variance=None,
        error_threshold=10.0,
        rollback_threshold=10.0,
    )
    observed = _make_observed(observed_result=(3.0, 4.0))
    witness = accept_action_result(expected, observed)
    assert witness.precision_weighted_outcome_error is None
    assert witness.outcome_prediction_error is not None
    assert witness.comparator_error is not None
    assert math.isclose(witness.comparator_error, witness.outcome_prediction_error, abs_tol=1e-12)


def test_17_dimension_mismatch_observed_vs_expected() -> None:
    expected = _make_expected(expected_result=(1.0, 2.0, 3.0))
    observed = _make_observed(observed_result=(1.0, 2.0))
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.INVALID_INPUT
    assert witness.reason.startswith("DIMENSION_MISMATCH")


def test_18_variance_dimension_mismatch_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="DIMENSION_MISMATCH"):
        _make_expected(
            expected_result=(0.0, 0.0, 0.0),
            expected_result_variance=(1.0, 1.0),
        )


def test_19_zero_variance_rejected() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        _make_expected(
            expected_result=(0.0, 0.0),
            expected_result_variance=(1.0, 0.0),
        )


def test_20_negative_variance_rejected() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        _make_expected(
            expected_result=(0.0, 0.0),
            expected_result_variance=(1.0, -0.5),
        )


def test_21_nan_rejected_anywhere() -> None:
    with pytest.raises(ValueError, match="NON_FINITE_VALUE"):
        _make_expected(expected_result=(1.0, math.nan, 0.0))
    with pytest.raises(ValueError, match="NON_FINITE_VALUE"):
        _make_observed(observed_result=(math.nan, 0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        _make_expected(error_threshold=math.nan)


def test_22_inf_rejected_anywhere() -> None:
    with pytest.raises(ValueError, match="NON_FINITE_VALUE"):
        _make_expected(expected_result=(1.0, math.inf, 0.0))
    with pytest.raises(ValueError, match="NON_FINITE_VALUE"):
        _make_observed(observed_result=(0.0, math.inf, 0.0))


def test_23_invalid_threshold_ordering_rejected() -> None:
    with pytest.raises(ValueError, match="INVALID_THRESHOLD_ORDERING"):
        _make_expected(error_threshold=2.0, rollback_threshold=1.0)


def test_24_witness_is_frozen() -> None:
    expected = _make_expected()
    observed = _make_observed()
    witness = accept_action_result(expected, observed)
    with pytest.raises(dataclasses.FrozenInstanceError):
        witness.status = ActionResultStatus.UPDATE_REQUIRED  # type: ignore[misc]


def test_25_repeated_calls_deterministic() -> None:
    expected = _make_expected(error_threshold=0.5, rollback_threshold=2.0)
    observed = _make_observed(observed_result=(1.7, 0.0, -1.0))
    w1 = accept_action_result(expected, observed)
    w2 = accept_action_result(expected, observed)
    assert w1 == w2


def test_26_no_created_before_action_field() -> None:
    field_names = {f.name for f in dataclasses.fields(ExpectedResultModel)}
    assert "created_before_action" not in field_names


def test_27_no_prior_confidence_field() -> None:
    field_names = {f.name for f in dataclasses.fields(ExpectedResultModel)}
    assert "prior_confidence" not in field_names
    witness_field_names = {f.name for f in dataclasses.fields(ActionResultWitness)}
    assert "prior_confidence" not in witness_field_names


def test_28_no_reentry_threshold_field() -> None:
    field_names = {f.name for f in dataclasses.fields(ExpectedResultModel)}
    assert "reentry_threshold" not in field_names
    witness_field_names = {f.name for f in dataclasses.fields(ActionResultWitness)}
    assert "reentry_required" not in witness_field_names
    # Status enum has no REENTRY_REQUIRED member.
    assert not any(member.name == "REENTRY_REQUIRED" for member in ActionResultStatus)


def test_29_no_threshold_modifier_or_error_history() -> None:
    # No public attribute or function on the module references error_history
    # behaviour.
    public_attrs = {name for name in dir(module_under_test) if not name.startswith("_")}
    assert "threshold_modifier" not in public_attrs
    assert "error_history" not in public_attrs
    # accept_action_result has only two parameters.
    sig = inspect.signature(accept_action_result)
    assert len(sig.parameters) == 2
    assert list(sig.parameters) == ["expected", "observed"]
    # No keyword-only parameters.
    for param in sig.parameters.values():
        assert param.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        }


def test_30_no_forecast_or_trading_fields() -> None:
    forbidden = {"forecast", "trading_signal", "biological_equivalence"}
    for cls in (ExpectedResultModel, ObservedActionResult, ActionResultWitness):
        names = {f.name for f in dataclasses.fields(cls)}
        assert names.isdisjoint(forbidden), f"{cls.__name__} has forbidden field"


def test_31_success_true_cannot_force_sanctioned_on_breach() -> None:
    expected = _make_expected(error_threshold=0.1, rollback_threshold=2.0)
    # OPE = 1.0 > error_threshold; success=True must not override.
    observed = _make_observed(observed_result=(2.0, 0.0, -1.0), success=True)
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.UPDATE_REQUIRED
    assert witness.accepted is False


def test_32_success_false_cannot_force_failure_on_exact_match() -> None:
    expected = _make_expected()
    observed = _make_observed(success=False)
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.SANCTIONED_MATCH
    assert witness.accepted is True


def test_33_comparator_error_at_rollback_threshold_is_rollback() -> None:
    # OPE = sqrt(1.0) = 1.0 == rollback_threshold (boundary >=).
    expected = _make_expected(error_threshold=0.5, rollback_threshold=1.0)
    observed = _make_observed(observed_result=(2.0, 0.0, -1.0))
    witness = accept_action_result(expected, observed)
    assert witness.comparator_error is not None
    assert math.isclose(witness.comparator_error, 1.0, abs_tol=1e-12)
    assert witness.status is ActionResultStatus.ROLLBACK_REQUIRED


def test_34_comparator_error_at_error_threshold_is_sanctioned() -> None:
    # OPE = 0.5 == error_threshold (strict > for UPDATE).
    expected = _make_expected(error_threshold=0.5, rollback_threshold=2.0)
    observed = _make_observed(observed_result=(1.5, 0.0, -1.0))
    witness = accept_action_result(expected, observed)
    assert witness.comparator_error is not None
    assert math.isclose(witness.comparator_error, 0.5, abs_tol=1e-12)
    assert witness.status is ActionResultStatus.SANCTIONED_MATCH


def test_35_rpe_ape_tpe_ignored_when_expected_component_none() -> None:
    expected = _make_expected(
        expected_value=None,
        expected_latency_ms=None,
        predicted_action_signature=None,
        value_mismatch_threshold=0.001,
        timing_mismatch_threshold_ms=0.001,
        action_mismatch_threshold=0.001,
    )
    observed = _make_observed(
        observed_value=999.0,
        observed_latency_ms=999.0,
        executed_action_signature=(99.0, 99.0),
    )
    witness = accept_action_result(expected, observed)
    # No mismatch since expected components are missing.
    assert witness.status is ActionResultStatus.SANCTIONED_MATCH
    assert witness.value_prediction_error is None
    assert witness.timing_prediction_error is None
    assert witness.action_prediction_error is None


def test_36_rpe_ape_tpe_ignored_when_threshold_none() -> None:
    expected = _make_expected(
        expected_value=1.0,
        expected_latency_ms=100.0,
        predicted_action_signature=(0.0, 0.0),
        # All mismatch thresholds None -> no mismatch fires.
        value_mismatch_threshold=None,
        timing_mismatch_threshold_ms=None,
        action_mismatch_threshold=None,
    )
    observed = _make_observed(
        observed_value=999.0,
        observed_latency_ms=999.0,
        executed_action_signature=(99.0, 99.0),
    )
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.SANCTIONED_MATCH
    # But scalars are still computed for telemetry.
    assert witness.value_prediction_error is not None
    assert witness.timing_prediction_error is not None
    assert witness.action_prediction_error is not None


def test_public_surface_unchanged() -> None:
    """Sanity check on the package-level public exports."""
    assert set(control_pkg.__all__) == {
        "ActionResultStatus",
        "ActionResultWitness",
        "ExpectedResultModel",
        "ObservedActionResult",
        "accept_action_result",
    }
