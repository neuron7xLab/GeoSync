# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Tests for ``geosync_hpc.control.action_result_acceptor``.

Falsifier probes (mutation -> test that must catch it):
    1. Flip ``OPE >= reentry_threshold`` to ``>``                   -> test_07_reentry_at_boundary
    2. Flip ``OPE >= rollback_threshold`` to ``>``                  -> test_08_rollback_at_boundary
    3. Flip ``OPE >  adapted_error_threshold`` to ``>=``            -> test_10_sanctioned_at_threshold_boundary
    4. Drop the ``reverse_afferentation_present`` short-circuit     -> test_15_missing_reverse_afferentation
    5. Drop the ``observed_result is None`` short-circuit           -> test_16_missing_observation
    6. Reorder rollback above reentry in decision cascade           -> test_07_reentry_at_boundary

These probes encode the live falsifiability contract for the acceptor: any
change that silently disables a layer of the cascade is caught by the named
test.  Probes are documented here, not executed via git mutation in this PR.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import math
from pathlib import Path

import pytest

from geosync_hpc import control as control_pkg
from geosync_hpc.control import (
    ActionResultStatus,
    ActionResultWitness,
    ExpectedResultModel,
    ObservedActionResult,
    accept_action_result,
)
from geosync_hpc.control import action_result_acceptor as module_under_test

# --------------------------------------------------------------------------- #
# Helpers (deterministic builders only — no fixtures with state).             #
# --------------------------------------------------------------------------- #


def _make_expected(
    *,
    action_id: str = "act-1",
    expected_result: tuple[float, ...] = (1.0, 0.0, -1.0),
    error_threshold: float = 0.5,
    rollback_threshold: float = 1.0,
    reentry_threshold: float = 2.0,
    expected_value: float | None = None,
    expected_latency_ms: float | None = None,
    predicted_action_signature: tuple[float, ...] | None = None,
    action_mismatch_threshold: float | None = None,
    value_mismatch_threshold: float | None = None,
    timing_mismatch_threshold_ms: float | None = None,
    prior_confidence: float = 0.5,
) -> ExpectedResultModel:
    return ExpectedResultModel(
        action_id=action_id,
        action_type="trade_open",
        expected_result=expected_result,
        context_signature=(0.1, 0.2, 0.3),
        prior_confidence=prior_confidence,
        error_threshold=error_threshold,
        rollback_threshold=rollback_threshold,
        reentry_threshold=reentry_threshold,
        expected_value=expected_value,
        expected_latency_ms=expected_latency_ms,
        predicted_action_signature=predicted_action_signature,
        action_mismatch_threshold=action_mismatch_threshold,
        value_mismatch_threshold=value_mismatch_threshold,
        timing_mismatch_threshold_ms=timing_mismatch_threshold_ms,
        created_before_action=True,
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
) -> ObservedActionResult:
    return ObservedActionResult(
        action_id=action_id,
        observed_result=observed_result,
        observed_value=observed_value,
        observed_latency_ms=observed_latency_ms,
        executed_action_signature=executed_action_signature,
        success=success,
        reverse_afferentation_present=reverse_afferentation_present,
    )


# --------------------------------------------------------------------------- #
# 1.  ExpectedResultModel construction-time guards                            #
# --------------------------------------------------------------------------- #


def test_01_expected_model_rejects_post_hoc_construction() -> None:
    """A model with ``created_before_action=False`` must raise ValueError."""

    with pytest.raises(ValueError, match="created_before_action"):
        ExpectedResultModel(
            action_id="x",
            action_type="t",
            expected_result=(0.0,),
            context_signature=(0.0,),
            prior_confidence=0.5,
            error_threshold=0.1,
            rollback_threshold=0.2,
            reentry_threshold=0.3,
            created_before_action=False,
        )


def test_02_expected_model_rejects_threshold_disorder() -> None:
    """rollback < error must be rejected with INVALID_THRESHOLD_ORDERING."""

    with pytest.raises(ValueError, match="INVALID_THRESHOLD_ORDERING"):
        ExpectedResultModel(
            action_id="x",
            action_type="t",
            expected_result=(0.0,),
            context_signature=(0.0,),
            prior_confidence=0.5,
            error_threshold=1.0,
            rollback_threshold=0.5,
            reentry_threshold=2.0,
        )


def test_03_expected_model_rejects_non_finite_field() -> None:
    """NaN in expected_result must be rejected with NON_FINITE_VALUE."""

    with pytest.raises(ValueError, match="NON_FINITE_VALUE"):
        ExpectedResultModel(
            action_id="x",
            action_type="t",
            expected_result=(math.nan,),
            context_signature=(0.0,),
            prior_confidence=0.5,
            error_threshold=0.1,
            rollback_threshold=0.2,
            reentry_threshold=0.3,
        )


# --------------------------------------------------------------------------- #
# 2.  ObservedActionResult construction-time guards                           #
# --------------------------------------------------------------------------- #


def test_04_observed_rejects_empty_action_id() -> None:
    with pytest.raises(ValueError, match="INVALID_OBSERVED_RESULT"):
        ObservedActionResult(action_id="", observed_result=(0.0,))


def test_05_observed_rejects_non_finite_value() -> None:
    with pytest.raises(ValueError, match="NON_FINITE_VALUE"):
        ObservedActionResult(action_id="x", observed_result=(0.0,), observed_value=math.inf)


# --------------------------------------------------------------------------- #
# 3.  Public-API status decisions                                             #
# --------------------------------------------------------------------------- #


def test_06_sanctioned_match_inside_threshold() -> None:
    expected = _make_expected()
    observed = _make_observed(observed_result=(1.0, 0.0, -1.0))
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.SANCTIONED_MATCH
    assert witness.accepted is True and witness.dissolved is True
    assert witness.update_required is False
    assert witness.next_context_expansion_required is False
    assert witness.outcome_prediction_error == pytest.approx(0.0)
    assert witness.reason.startswith("SANCTIONED_MATCH")


def test_07_reentry_at_boundary() -> None:
    """OPE == reentry_threshold must trigger REENTRY_REQUIRED (>=)."""

    expected = _make_expected(
        expected_result=(0.0,),
        error_threshold=0.5,
        rollback_threshold=1.0,
        reentry_threshold=2.0,
    )
    observed = _make_observed(observed_result=(2.0,))
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.REENTRY_REQUIRED
    assert witness.reentry_required is True
    assert witness.rollback_required is True
    assert witness.inhibit_repetition is True


def test_08_rollback_at_boundary() -> None:
    """OPE == rollback_threshold must trigger ROLLBACK_REQUIRED (>=)."""

    expected = _make_expected(
        expected_result=(0.0,),
        error_threshold=0.5,
        rollback_threshold=1.0,
        reentry_threshold=2.0,
    )
    observed = _make_observed(observed_result=(1.0,))
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.ROLLBACK_REQUIRED
    assert witness.rollback_required is True
    assert witness.reentry_required is False


def test_09_update_required_above_adapted_threshold() -> None:
    expected = _make_expected(
        expected_result=(0.0,),
        error_threshold=0.5,
        rollback_threshold=1.0,
        reentry_threshold=2.0,
    )
    observed = _make_observed(observed_result=(0.7,))
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.UPDATE_REQUIRED
    assert witness.update_required is True
    assert witness.rollback_required is False


def test_10_sanctioned_at_threshold_boundary() -> None:
    """OPE == adapted_error_threshold must SANCTION (strict > for UPDATE)."""

    expected = _make_expected(
        expected_result=(0.0,),
        error_threshold=0.5,
        rollback_threshold=1.0,
        reentry_threshold=2.0,
    )
    observed = _make_observed(observed_result=(0.5,))
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.SANCTIONED_MATCH


# --------------------------------------------------------------------------- #
# 4.  Mismatch lanes (action / value / timing)                                #
# --------------------------------------------------------------------------- #


def test_11_action_mismatch_via_signature() -> None:
    expected = _make_expected(
        predicted_action_signature=(1.0, 0.0),
        action_mismatch_threshold=0.5,
    )
    observed = _make_observed(executed_action_signature=(0.0, 0.0))
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.ACTION_MISMATCH
    assert witness.update_required is True
    assert witness.action_prediction_error == pytest.approx(1.0)


def test_12_value_mismatch_uses_absolute_difference() -> None:
    expected = _make_expected(
        expected_value=10.0,
        value_mismatch_threshold=0.5,
    )
    observed = _make_observed(observed_value=11.0)
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.VALUE_MISMATCH
    assert witness.value_prediction_error == pytest.approx(1.0)


def test_13_timing_mismatch_uses_absolute_difference() -> None:
    expected = _make_expected(
        expected_latency_ms=100.0,
        timing_mismatch_threshold_ms=10.0,
    )
    observed = _make_observed(observed_latency_ms=80.0)
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.TIMING_MISMATCH
    assert witness.timing_prediction_error == pytest.approx(-20.0)


def test_14_action_id_mismatch_short_circuits_before_value() -> None:
    expected = _make_expected(action_id="A")
    observed = _make_observed(action_id="B")
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.ACTION_MISMATCH
    assert witness.reason.startswith("ACTION_ID_MISMATCH")


# --------------------------------------------------------------------------- #
# 5.  Insufficient-evidence lanes                                             #
# --------------------------------------------------------------------------- #


def test_15_missing_reverse_afferentation() -> None:
    expected = _make_expected()
    observed = _make_observed(reverse_afferentation_present=False)
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.INSUFFICIENT_REVERSE_AFFERENTATION
    assert witness.accepted is False
    assert witness.inhibit_repetition is True
    assert witness.next_context_expansion_required is True


def test_16_missing_observation() -> None:
    expected = _make_expected()
    observed = _make_observed(observed_result=None)
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.INSUFFICIENT_OBSERVATION
    assert witness.accepted is False


# --------------------------------------------------------------------------- #
# 6.  INVALID_INPUT lanes                                                     #
# --------------------------------------------------------------------------- #


def test_17_missing_expected_model_returns_invalid_input() -> None:
    """Acceptor must fail closed when no pre-action model is supplied."""

    observed = _make_observed()
    witness = accept_action_result(None, observed)
    assert witness.status is ActionResultStatus.INVALID_INPUT
    assert witness.accepted is False
    assert witness.inhibit_repetition is True
    assert witness.reason.startswith("INVALID_EXPECTED_MODEL")


def test_18_dimension_mismatch_returns_invalid_input() -> None:
    expected = _make_expected(expected_result=(1.0, 0.0))
    observed = _make_observed(observed_result=(1.0, 0.0, 0.0))
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.INVALID_INPUT
    assert witness.reason.startswith("DIMENSION_MISMATCH")


def test_19_negative_error_history_returns_invalid_input() -> None:
    expected = _make_expected()
    observed = _make_observed()
    witness = accept_action_result(expected, observed, error_history=(-0.1, 0.2))
    assert witness.status is ActionResultStatus.INVALID_INPUT


# --------------------------------------------------------------------------- #
# 7.  Witness immutability + determinism                                      #
# --------------------------------------------------------------------------- #


def test_20_witness_is_frozen_dataclass() -> None:
    expected = _make_expected()
    observed = _make_observed()
    witness = accept_action_result(expected, observed)
    with pytest.raises(dataclasses.FrozenInstanceError):
        witness.status = ActionResultStatus.INVALID_INPUT  # type: ignore[misc]


def test_21_repeated_calls_are_deterministic() -> None:
    expected = _make_expected(expected_result=(0.0,))
    observed = _make_observed(observed_result=(0.7,))
    first = accept_action_result(expected, observed, error_history=(0.1, 0.2, 0.3))
    second = accept_action_result(expected, observed, error_history=(0.1, 0.2, 0.3))
    assert first == second


# --------------------------------------------------------------------------- #
# 8.  Forbidden-field / forbidden-import structural guards                    #
# --------------------------------------------------------------------------- #


_FORBIDDEN_FIELDS: frozenset[str] = frozenset(
    {"forecast", "trading_signal", "biological_equivalence"}
)


def test_22_no_forbidden_fields_in_dataclasses() -> None:
    for cls in (ExpectedResultModel, ObservedActionResult, ActionResultWitness):
        names = {f.name for f in dataclasses.fields(cls)}
        leak = names & _FORBIDDEN_FIELDS
        assert not leak, f"{cls.__name__} contains forbidden field: {leak}"


_FORBIDDEN_IMPORT_PREFIXES: tuple[str, ...] = (
    "geosync_hpc.execution",
    "geosync_hpc.policy",
    "geosync_hpc.signal",
    "geosync_hpc.regime",
    "geosync_hpc.backtest",
    "geosync_hpc.risk",
    "geosync_hpc.trading",
    "nak_controller",
    "nak_controller.aar",
    "trading",
    "execution",
    "policy",
    "forecast",
)


def test_23_module_has_no_forbidden_imports() -> None:
    """Static AST scan: the implementation must not import runtime layers."""

    source_path = Path(inspect.getsourcefile(module_under_test) or "")
    assert source_path.is_file(), "module source path must exist for AST scan"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name.startswith(p) for p in _FORBIDDEN_IMPORT_PREFIXES):
                    found.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if any(mod.startswith(p) for p in _FORBIDDEN_IMPORT_PREFIXES):
                found.append(mod)
    assert found == [], f"forbidden imports detected: {found}"


def test_24_package_surface_is_minimal() -> None:
    """``geosync_hpc.control`` exports exactly the documented public names."""

    expected_surface: frozenset[str] = frozenset(
        {
            "ExpectedResultModel",
            "ObservedActionResult",
            "ActionResultStatus",
            "ActionResultWitness",
            "accept_action_result",
        }
    )
    assert frozenset(control_pkg.__all__) == expected_surface


# --------------------------------------------------------------------------- #
# 9.  Decision-priority cascade (regression locks for 11-step order)          #
# --------------------------------------------------------------------------- #


def test_25_invalid_input_dominates_insufficient() -> None:
    """Dimension mismatch wins over missing observation logic."""

    expected = _make_expected(expected_result=(0.0, 0.0))
    observed = _make_observed(observed_result=(0.0, 0.0, 0.0))
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.INVALID_INPUT


def test_26_reverse_afferentation_dominates_observation() -> None:
    """Missing reverse afferentation wins over missing observed_result."""

    expected = _make_expected()
    observed = _make_observed(
        observed_result=None,
        reverse_afferentation_present=False,
    )
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.INSUFFICIENT_REVERSE_AFFERENTATION


def test_27_reentry_dominates_action_signature_mismatch() -> None:
    """Step 5 (REENTRY) precedes step 7 (ACTION_MISMATCH on signature)."""

    expected = _make_expected(
        expected_result=(0.0,),
        error_threshold=0.5,
        rollback_threshold=1.0,
        reentry_threshold=2.0,
        predicted_action_signature=(1.0,),
        action_mismatch_threshold=0.0,
    )
    observed = _make_observed(
        observed_result=(2.5,),
        executed_action_signature=(1.0,),
    )
    witness = accept_action_result(expected, observed)
    assert witness.status is ActionResultStatus.REENTRY_REQUIRED


# --------------------------------------------------------------------------- #
# 10.  Adaptive threshold modifier                                            #
# --------------------------------------------------------------------------- #


def test_28_error_history_inflates_threshold_within_clamp() -> None:
    expected = _make_expected(
        expected_result=(0.0,),
        error_threshold=1.0,
        rollback_threshold=2.0,
        reentry_threshold=3.0,
    )
    observed = _make_observed(observed_result=(1.05,))
    # mean(history)=1.0 -> modifier = clamp(1.1, 0.75, 1.25) = 1.1
    # adapted threshold = 1.1 -> OPE 1.05 <= 1.1 -> SANCTION
    witness = accept_action_result(expected, observed, error_history=(1.0,))
    assert witness.status is ActionResultStatus.SANCTIONED_MATCH
    assert witness.adapted_error_threshold == pytest.approx(1.1)


def test_29_threshold_modifier_clamped_to_upper_bound() -> None:
    expected = _make_expected(
        expected_result=(0.0,),
        error_threshold=1.0,
        rollback_threshold=2.0,
        reentry_threshold=3.0,
    )
    observed = _make_observed(observed_result=(0.0,))
    # mean=10.0 -> raw modifier = 1.0 + 1.0 = 2.0 -> clamped to 1.25
    witness = accept_action_result(expected, observed, error_history=(10.0,))
    assert witness.adapted_error_threshold == pytest.approx(1.25)


# --------------------------------------------------------------------------- #
# 11.  Precision-weighted gain bound                                          #
# --------------------------------------------------------------------------- #


def test_30_precision_weighted_gain_clamped_to_unit() -> None:
    expected = _make_expected(
        expected_result=(0.0,),
        prior_confidence=1.0,
        error_threshold=0.5,
        rollback_threshold=10.0,
        reentry_threshold=100.0,
    )
    observed = _make_observed(observed_result=(50.0,))
    witness = accept_action_result(expected, observed)
    assert 0.0 <= witness.precision_weighted_gain <= 1.0


# --------------------------------------------------------------------------- #
# 12.  Status-enum is StrEnum                                                 #
# --------------------------------------------------------------------------- #


def test_31_status_is_str_enum_with_stable_codes() -> None:
    assert ActionResultStatus.SANCTIONED_MATCH == "SANCTIONED_MATCH"
    assert ActionResultStatus.INVALID_INPUT == "INVALID_INPUT"
    # All members must use the documented stable code as their value.
    for member in ActionResultStatus:
        assert member.value == member.name
