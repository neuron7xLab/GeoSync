# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Deterministic ActionResultComparator for the GeoSync HPC control plane.

The comparator enforces the chain
    expected -> observed -> error -> update / rollback
by pure, side-effect-free comparison.  No IO, no wall-clock, no random,
no mutable globals, no trading / execution / policy / forecast imports,
no biological-equivalence imports.

Engineering analogue only.  Names such as ``expected_result``,
``reverse_afferentation``, ``predicted_action_signature`` are shorthand
for bounded mathematical objects described below.  No biological
equivalence claim is made.

Design corrections relative to the prior "acceptor":
    1. Chronology is encoded only via integer sequence numbers
       (``model_created_seq < action_started_seq < observed_seq``); the
       caller cannot present a ``created_before_action=True`` boolean
       and skip ordering proof.
    2. There is no ``REENTRY_REQUIRED`` status — re-entry is a higher
       level policy decision, not a comparator concern.
    3. There is no ``prior_confidence`` and no ``confidence * error``
       formula. Precision is real and uses the inverse variance of the
       expected result vector; absence of variance means precision is
       not claimed and the comparator falls back to the raw Euclidean
       Outcome Prediction Error.
    4. There is no internal adaptive-threshold magic. The witness
       reports ``adapted_error_threshold == error_threshold`` exactly;
       any adaptation is the caller's responsibility.
    5. ``success`` cannot override numerical comparison.
    6. Missing expected model fails closed (``INVALID_INPUT``); the
       comparator never fabricates a default ``Prediction``.
    7. Logged observation alone (``reverse_afferentation_present=False``)
       can never be sanctioned.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Iterable
from dataclasses import dataclass


class ActionResultStatus(enum.StrEnum):
    """Stable status codes emitted by :func:`accept_action_result`.

    Order is significant only in :func:`accept_action_result`'s decision
    cascade; the enum itself does not impose semantic ordering.

    There is intentionally no ``REENTRY_REQUIRED`` member: re-entering
    a context is a higher-level policy decision, not a comparator
    output.
    """

    SANCTIONED_MATCH = "SANCTIONED_MATCH"
    UPDATE_REQUIRED = "UPDATE_REQUIRED"
    ROLLBACK_REQUIRED = "ROLLBACK_REQUIRED"
    INSUFFICIENT_OBSERVATION = "INSUFFICIENT_OBSERVATION"
    INSUFFICIENT_REVERSE_AFFERENTATION = "INSUFFICIENT_REVERSE_AFFERENTATION"
    ACTION_MISMATCH = "ACTION_MISMATCH"
    VALUE_MISMATCH = "VALUE_MISMATCH"
    TIMING_MISMATCH = "TIMING_MISMATCH"
    INVALID_INPUT = "INVALID_INPUT"


def _is_finite_float(value: float) -> bool:
    return math.isfinite(value)


def _all_finite(items: Iterable[float]) -> bool:
    return all(_is_finite_float(x) for x in items)


@dataclass(frozen=True, slots=True)
class ExpectedResultModel:
    """Pre-action expected outcome contract.

    Construction-time validation enforces structural invariants.  The
    :func:`accept_action_result` entry point re-validates so callers
    cannot bypass the contract by reaching into ``__dict__``.

    Sequence ordering invariants (validated here):
        ``model_created_seq < action_started_seq``
    The further invariant ``action_started_seq < observed_seq`` is
    checked at comparison time because it requires both sides.

    There is intentionally NO ``created_before_action`` boolean; a
    boolean cannot be ordered against another timestamp and was
    therefore an honesty trap.
    """

    action_id: str
    action_type: str
    expected_result: tuple[float, ...]
    expected_result_variance: tuple[float, ...] | None
    context_signature: tuple[float, ...]
    model_created_seq: int
    action_started_seq: int
    error_threshold: float
    rollback_threshold: float
    expected_value: float | None = None
    expected_latency_ms: float | None = None
    predicted_action_signature: tuple[float, ...] | None = None
    action_mismatch_threshold: float | None = None
    value_mismatch_threshold: float | None = None
    timing_mismatch_threshold_ms: float | None = None

    def __post_init__(self) -> None:
        _validate_expected(self)


@dataclass(frozen=True, slots=True)
class ObservedActionResult:
    """Post-action observed outcome bundle.

    ``success`` is recorded for downstream telemetry only; it never
    overrides numerical comparison against the expected model.

    The invariant ``observed_seq > action_started_seq`` is checked
    against the supplied :class:`ExpectedResultModel` inside
    :func:`accept_action_result`, since it depends on both sides.
    """

    action_id: str
    observed_seq: int
    observed_result: tuple[float, ...] | None = None
    observed_value: float | None = None
    observed_latency_ms: float | None = None
    executed_action_signature: tuple[float, ...] | None = None
    success: bool | None = None
    reverse_afferentation_present: bool = True

    def __post_init__(self) -> None:
        _validate_observed(self)


@dataclass(frozen=True, slots=True)
class ActionResultWitness:
    """Immutable verdict produced by :func:`accept_action_result`.

    All booleans are derived from :attr:`status` per the exact flag
    table in the module docstring. ``reason`` always begins with one of
    the documented stable codes so downstream consumers can switch on a
    prefix rather than parse free-form text.

    There is intentionally no ``reentry_required`` flag and no
    ``precision_weighted_gain`` scalar: re-entry is a higher-level
    policy decision and ``precision_weighted_gain`` was a confidence
    pseudo-quantity that masqueraded as physics.  Precision is exposed
    as :attr:`precision_weighted_outcome_error` (inverse-variance
    weighted Euclidean error) and as :attr:`comparator_error`
    (precision-weighted error if available, raw OPE otherwise).
    """

    status: ActionResultStatus
    accepted: bool
    dissolved: bool
    rollback_required: bool
    update_required: bool
    inhibit_repetition: bool
    outcome_prediction_error: float | None
    precision_weighted_outcome_error: float | None
    comparator_error: float | None
    value_prediction_error: float | None
    action_prediction_error: float | None
    timing_prediction_error: float | None
    adapted_error_threshold: float
    next_context_expansion_required: bool
    reason: str
    falsifier: str


def _validate_expected(model: ExpectedResultModel) -> None:
    if not isinstance(model.action_id, str) or not model.action_id:
        raise ValueError("INVALID_EXPECTED_MODEL: action_id must be a non-empty string")
    if not isinstance(model.action_type, str) or not model.action_type:
        raise ValueError("INVALID_EXPECTED_MODEL: action_type must be a non-empty string")
    if not isinstance(model.expected_result, tuple) or len(model.expected_result) == 0:
        raise ValueError("INVALID_EXPECTED_MODEL: expected_result must be a non-empty tuple")
    if not _all_finite(model.expected_result):
        raise ValueError("NON_FINITE_VALUE: expected_result contains non-finite element")
    if not isinstance(model.context_signature, tuple) or len(model.context_signature) == 0:
        raise ValueError("INVALID_EXPECTED_MODEL: context_signature must be a non-empty tuple")
    if not _all_finite(model.context_signature):
        raise ValueError("NON_FINITE_VALUE: context_signature contains non-finite element")
    # Strict int check (bool is a subclass of int — reject it explicitly).
    if not isinstance(model.model_created_seq, int) or isinstance(model.model_created_seq, bool):
        raise ValueError("INVALID_EXPECTED_MODEL: model_created_seq must be int")
    if model.model_created_seq < 0:
        raise ValueError("INVALID_EXPECTED_MODEL: model_created_seq must be >= 0")
    if not isinstance(model.action_started_seq, int) or isinstance(model.action_started_seq, bool):
        raise ValueError("INVALID_EXPECTED_MODEL: action_started_seq must be int")
    if model.action_started_seq < 0:
        raise ValueError("INVALID_EXPECTED_MODEL: action_started_seq must be >= 0")
    if not (model.model_created_seq < model.action_started_seq):
        raise ValueError(
            "SEQUENCE_ORDER_INVALID: model_created_seq must be strictly less "
            "than action_started_seq"
        )
    if not _is_finite_float(model.error_threshold) or model.error_threshold < 0.0:
        raise ValueError("INVALID_EXPECTED_MODEL: error_threshold must be finite and >= 0")
    if not _is_finite_float(model.rollback_threshold) or model.rollback_threshold < 0.0:
        raise ValueError("INVALID_EXPECTED_MODEL: rollback_threshold must be finite and >= 0")
    if model.rollback_threshold < model.error_threshold:
        raise ValueError(
            "INVALID_THRESHOLD_ORDERING: rollback_threshold must be >= error_threshold"
        )
    if model.expected_value is not None and not _is_finite_float(model.expected_value):
        raise ValueError("NON_FINITE_VALUE: expected_value must be finite when provided")
    if model.expected_latency_ms is not None and not _is_finite_float(model.expected_latency_ms):
        raise ValueError("NON_FINITE_VALUE: expected_latency_ms must be finite when provided")
    if model.predicted_action_signature is not None:
        if (
            not isinstance(model.predicted_action_signature, tuple)
            or len(model.predicted_action_signature) == 0
        ):
            raise ValueError(
                "INVALID_EXPECTED_MODEL: predicted_action_signature must be a non-empty tuple"
            )
        if not _all_finite(model.predicted_action_signature):
            raise ValueError(
                "NON_FINITE_VALUE: predicted_action_signature contains non-finite element"
            )
    if model.expected_result_variance is not None:
        if not isinstance(model.expected_result_variance, tuple):
            raise ValueError(
                "INVALID_EXPECTED_MODEL: expected_result_variance must be a tuple or None"
            )
        if len(model.expected_result_variance) != len(model.expected_result):
            raise ValueError(
                "DIMENSION_MISMATCH: expected_result_variance length "
                f"{len(model.expected_result_variance)} does not match expected_result length "
                f"{len(model.expected_result)}"
            )
        for variance in model.expected_result_variance:
            if not _is_finite_float(variance):
                raise ValueError(
                    "NON_FINITE_VALUE: expected_result_variance contains non-finite element"
                )
            if not (variance > 0.0):
                raise ValueError(
                    "INVALID_EXPECTED_MODEL: expected_result_variance entries must be > 0 "
                    "(zero / negative variance rejected)"
                )
    for name, value in (
        ("action_mismatch_threshold", model.action_mismatch_threshold),
        ("value_mismatch_threshold", model.value_mismatch_threshold),
        ("timing_mismatch_threshold_ms", model.timing_mismatch_threshold_ms),
    ):
        if value is None:
            continue
        if not _is_finite_float(value) or value < 0.0:
            raise ValueError(
                f"INVALID_EXPECTED_MODEL: {name} must be finite and >= 0 when provided"
            )


def _validate_observed(observed: ObservedActionResult) -> None:
    if not isinstance(observed.action_id, str) or not observed.action_id:
        raise ValueError("INVALID_OBSERVED_RESULT: action_id must be a non-empty string")
    if not isinstance(observed.observed_seq, int) or isinstance(observed.observed_seq, bool):
        raise ValueError("INVALID_OBSERVED_RESULT: observed_seq must be int")
    if observed.observed_seq < 0:
        raise ValueError("INVALID_OBSERVED_RESULT: observed_seq must be >= 0")
    if observed.observed_result is not None:
        if not isinstance(observed.observed_result, tuple):
            raise ValueError("INVALID_OBSERVED_RESULT: observed_result must be a tuple or None")
        if not _all_finite(observed.observed_result):
            raise ValueError("NON_FINITE_VALUE: observed_result contains non-finite element")
    if observed.observed_value is not None and not _is_finite_float(observed.observed_value):
        raise ValueError("NON_FINITE_VALUE: observed_value must be finite when provided")
    if observed.observed_latency_ms is not None and not _is_finite_float(
        observed.observed_latency_ms
    ):
        raise ValueError("NON_FINITE_VALUE: observed_latency_ms must be finite when provided")
    if observed.executed_action_signature is not None:
        if not isinstance(observed.executed_action_signature, tuple):
            raise ValueError(
                "INVALID_OBSERVED_RESULT: executed_action_signature must be a tuple or None"
            )
        if not _all_finite(observed.executed_action_signature):
            raise ValueError(
                "NON_FINITE_VALUE: executed_action_signature contains non-finite element"
            )


def _euclidean(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    total = 0.0
    for x, y in zip(a, b, strict=True):
        diff = x - y
        total += diff * diff
    return math.sqrt(total)


def _precision_weighted_euclidean(
    observed: tuple[float, ...],
    expected: tuple[float, ...],
    variance: tuple[float, ...],
) -> float:
    """Inverse-variance weighted Euclidean error.

    Defined as ``sqrt( sum(p_i * (o_i - e_i)^2) / sum(p_i) )`` where
    ``p_i = 1 / variance_i``. Variance entries are guaranteed strictly
    positive at construction time.
    """

    weighted_squared = 0.0
    weight_total = 0.0
    for o, e, v in zip(observed, expected, variance, strict=True):
        precision = 1.0 / v
        diff = o - e
        weighted_squared += precision * diff * diff
        weight_total += precision
    return math.sqrt(weighted_squared / weight_total)


def _invalid_input_witness(reason: str, falsifier: str) -> ActionResultWitness:
    return ActionResultWitness(
        status=ActionResultStatus.INVALID_INPUT,
        accepted=False,
        dissolved=False,
        rollback_required=False,
        update_required=False,
        inhibit_repetition=True,
        outcome_prediction_error=None,
        precision_weighted_outcome_error=None,
        comparator_error=None,
        value_prediction_error=None,
        action_prediction_error=None,
        timing_prediction_error=None,
        adapted_error_threshold=0.0,
        next_context_expansion_required=False,
        reason=reason,
        falsifier=falsifier,
    )


def _insufficient_witness(
    status: ActionResultStatus, reason: str, falsifier: str
) -> ActionResultWitness:
    return ActionResultWitness(
        status=status,
        accepted=False,
        dissolved=False,
        rollback_required=False,
        update_required=False,
        inhibit_repetition=True,
        outcome_prediction_error=None,
        precision_weighted_outcome_error=None,
        comparator_error=None,
        value_prediction_error=None,
        action_prediction_error=None,
        timing_prediction_error=None,
        adapted_error_threshold=0.0,
        next_context_expansion_required=True,
        reason=reason,
        falsifier=falsifier,
    )


def accept_action_result(
    expected: ExpectedResultModel | None,
    observed: ObservedActionResult,
) -> ActionResultWitness:
    """Compare a single (expected, observed) pair and emit a witness.

    Decision priority (exact, no shuffling):
        1.  INVALID_INPUT (any failed validation, including missing
            expected model, sequence-order violation, dimension
            mismatch, non-finite values, threshold ordering issues).
        2.  INSUFFICIENT_REVERSE_AFFERENTATION
            (``observed.reverse_afferentation_present is False``).
        3.  INSUFFICIENT_OBSERVATION (``observed.observed_result is None``).
        4.  ACTION_MISMATCH on differing ``action_id``.
        5.  ROLLBACK_REQUIRED if ``comparator_error >= rollback_threshold``.
        6.  ACTION_MISMATCH on signature distance breach.
        7.  VALUE_MISMATCH on ``|RPE|`` breach.
        8.  TIMING_MISMATCH on ``|TPE|`` breach.
        9.  UPDATE_REQUIRED if ``comparator_error > error_threshold``.
        10. SANCTIONED_MATCH otherwise.

    The function is total: it never raises for normal invalid input;
    it returns a witness with status ``INVALID_INPUT`` instead.
    Exceptions are reserved for impossible programmer errors.
    """

    if expected is None:
        return _invalid_input_witness(
            reason="INVALID_EXPECTED_MODEL: expected is None",
            falsifier=(
                "Caller passed expected=None; comparator would silently fabricate a "
                "default Prediction. Comparator must fail closed."
            ),
        )

    try:
        _validate_expected(expected)
    except ValueError as exc:
        return _invalid_input_witness(
            reason=str(exc),
            falsifier=(
                "Caller mutated ExpectedResultModel internals or bypassed __post_init__; "
                "comparator would proceed with malformed contract."
            ),
        )

    try:
        _validate_observed(observed)
    except ValueError as exc:
        return _invalid_input_witness(
            reason=str(exc),
            falsifier=(
                "Caller passed malformed ObservedActionResult; comparator would compute "
                "errors against non-finite or wrongly typed observation."
            ),
        )

    if not (observed.observed_seq > expected.action_started_seq):
        return _invalid_input_witness(
            reason=(
                f"SEQUENCE_ORDER_INVALID: observed_seq={observed.observed_seq} must be "
                f"strictly greater than action_started_seq={expected.action_started_seq}"
            ),
            falsifier=(
                "Comparator accepted an observation logged before or at the action start; "
                "post-hoc observation would silently validate any action."
            ),
        )

    if not observed.reverse_afferentation_present:
        return _insufficient_witness(
            ActionResultStatus.INSUFFICIENT_REVERSE_AFFERENTATION,
            reason=(
                "MISSING_REVERSE_AFFERENTATION: observation lacks reverse afferentation "
                "channel; cannot accept action result"
            ),
            falsifier=(
                "Comparator accepted an action without reverse afferentation; logged "
                "observation alone must never be sanctioned."
            ),
        )

    if observed.observed_result is None:
        return _insufficient_witness(
            ActionResultStatus.INSUFFICIENT_OBSERVATION,
            reason="MISSING_OBSERVATION: observed_result is None",
            falsifier=(
                "Comparator accepted an action with no observed result vector; "
                "comparison is undefined."
            ),
        )

    if observed.action_id != expected.action_id:
        return ActionResultWitness(
            status=ActionResultStatus.ACTION_MISMATCH,
            accepted=False,
            dissolved=False,
            rollback_required=False,
            update_required=True,
            inhibit_repetition=True,
            outcome_prediction_error=None,
            precision_weighted_outcome_error=None,
            comparator_error=None,
            value_prediction_error=None,
            action_prediction_error=None,
            timing_prediction_error=None,
            adapted_error_threshold=expected.error_threshold,
            next_context_expansion_required=True,
            reason=(
                f"ACTION_ID_MISMATCH: expected={expected.action_id!r} "
                f"observed={observed.action_id!r}"
            ),
            falsifier=(
                "Comparator compared two different actions; cross-pairing would silently "
                "validate the wrong outcome."
            ),
        )

    if len(observed.observed_result) != len(expected.expected_result):
        return _invalid_input_witness(
            reason=(
                f"DIMENSION_MISMATCH: observed_result has {len(observed.observed_result)} "
                f"elements, expected_result has {len(expected.expected_result)}"
            ),
            falsifier=(
                "Comparator compared vectors of different shapes; Euclidean error "
                "would be ill-defined or zero-padded."
            ),
        )

    if (
        expected.predicted_action_signature is not None
        and observed.executed_action_signature is not None
        and len(observed.executed_action_signature) != len(expected.predicted_action_signature)
    ):
        return _invalid_input_witness(
            reason=(
                "DIMENSION_MISMATCH: executed_action_signature has "
                f"{len(observed.executed_action_signature)} elements, "
                f"predicted_action_signature has {len(expected.predicted_action_signature)}"
            ),
            falsifier=(
                "Comparator compared action signatures of different shapes; APE would be "
                "ill-defined."
            ),
        )

    raw_ope: float = _euclidean(observed.observed_result, expected.expected_result)

    pwope: float | None
    if expected.expected_result_variance is not None:
        pwope = _precision_weighted_euclidean(
            observed.observed_result,
            expected.expected_result,
            expected.expected_result_variance,
        )
    else:
        pwope = None

    comparator_error: float = pwope if pwope is not None else raw_ope

    rpe: float | None
    if observed.observed_value is not None and expected.expected_value is not None:
        rpe = observed.observed_value - expected.expected_value
    else:
        rpe = None

    ape: float | None
    if (
        observed.executed_action_signature is not None
        and expected.predicted_action_signature is not None
    ):
        ape = _euclidean(observed.executed_action_signature, expected.predicted_action_signature)
    else:
        ape = None

    tpe: float | None
    if observed.observed_latency_ms is not None and expected.expected_latency_ms is not None:
        tpe = observed.observed_latency_ms - expected.expected_latency_ms
    else:
        tpe = None

    adapted_error_threshold: float = expected.error_threshold

    if comparator_error >= expected.rollback_threshold:
        return ActionResultWitness(
            status=ActionResultStatus.ROLLBACK_REQUIRED,
            accepted=False,
            dissolved=False,
            rollback_required=True,
            update_required=False,
            inhibit_repetition=True,
            outcome_prediction_error=raw_ope,
            precision_weighted_outcome_error=pwope,
            comparator_error=comparator_error,
            value_prediction_error=rpe,
            action_prediction_error=ape,
            timing_prediction_error=tpe,
            adapted_error_threshold=adapted_error_threshold,
            next_context_expansion_required=True,
            reason=(
                f"ROLLBACK_REQUIRED: comparator_error={comparator_error:.6g} "
                f">= rollback_threshold={expected.rollback_threshold:.6g}"
            ),
            falsifier=(
                "Comparator accepted an action whose outcome breached the rollback "
                "threshold; the controller must roll back, not update."
            ),
        )

    if (
        ape is not None
        and expected.action_mismatch_threshold is not None
        and ape > expected.action_mismatch_threshold
    ):
        return ActionResultWitness(
            status=ActionResultStatus.ACTION_MISMATCH,
            accepted=False,
            dissolved=False,
            rollback_required=False,
            update_required=True,
            inhibit_repetition=True,
            outcome_prediction_error=raw_ope,
            precision_weighted_outcome_error=pwope,
            comparator_error=comparator_error,
            value_prediction_error=rpe,
            action_prediction_error=ape,
            timing_prediction_error=tpe,
            adapted_error_threshold=adapted_error_threshold,
            next_context_expansion_required=True,
            reason=(
                f"ACTION_MISMATCH: APE={ape:.6g} > action_mismatch_threshold="
                f"{expected.action_mismatch_threshold:.6g}"
            ),
            falsifier=(
                "Executed action signature drifted beyond contract; comparator cannot "
                "sanction divergent execution as a match."
            ),
        )

    if (
        rpe is not None
        and expected.value_mismatch_threshold is not None
        and abs(rpe) > expected.value_mismatch_threshold
    ):
        return ActionResultWitness(
            status=ActionResultStatus.VALUE_MISMATCH,
            accepted=False,
            dissolved=False,
            rollback_required=False,
            update_required=True,
            inhibit_repetition=True,
            outcome_prediction_error=raw_ope,
            precision_weighted_outcome_error=pwope,
            comparator_error=comparator_error,
            value_prediction_error=rpe,
            action_prediction_error=ape,
            timing_prediction_error=tpe,
            adapted_error_threshold=adapted_error_threshold,
            next_context_expansion_required=True,
            reason=(
                f"VALUE_MISMATCH: |RPE|={abs(rpe):.6g} > value_mismatch_threshold="
                f"{expected.value_mismatch_threshold:.6g}"
            ),
            falsifier=(
                "Realized scalar value diverged beyond contract; comparator cannot "
                "sanction the action without flagging update."
            ),
        )

    if (
        tpe is not None
        and expected.timing_mismatch_threshold_ms is not None
        and abs(tpe) > expected.timing_mismatch_threshold_ms
    ):
        return ActionResultWitness(
            status=ActionResultStatus.TIMING_MISMATCH,
            accepted=False,
            dissolved=False,
            rollback_required=False,
            update_required=True,
            inhibit_repetition=True,
            outcome_prediction_error=raw_ope,
            precision_weighted_outcome_error=pwope,
            comparator_error=comparator_error,
            value_prediction_error=rpe,
            action_prediction_error=ape,
            timing_prediction_error=tpe,
            adapted_error_threshold=adapted_error_threshold,
            next_context_expansion_required=True,
            reason=(
                f"TIMING_MISMATCH: |TPE|={abs(tpe):.6g} > timing_mismatch_threshold_ms="
                f"{expected.timing_mismatch_threshold_ms:.6g}"
            ),
            falsifier=(
                "Observed latency diverged beyond contract; comparator cannot sanction "
                "out-of-window execution."
            ),
        )

    if comparator_error > expected.error_threshold:
        return ActionResultWitness(
            status=ActionResultStatus.UPDATE_REQUIRED,
            accepted=False,
            dissolved=False,
            rollback_required=False,
            update_required=True,
            inhibit_repetition=False,
            outcome_prediction_error=raw_ope,
            precision_weighted_outcome_error=pwope,
            comparator_error=comparator_error,
            value_prediction_error=rpe,
            action_prediction_error=ape,
            timing_prediction_error=tpe,
            adapted_error_threshold=adapted_error_threshold,
            next_context_expansion_required=True,
            reason=(
                f"UPDATE_REQUIRED: comparator_error={comparator_error:.6g} > "
                f"error_threshold={expected.error_threshold:.6g}"
            ),
            falsifier=(
                "Outcome error sits inside the safe band but above the error threshold; "
                "comparator must request a model update rather than dissolve the action."
            ),
        )

    return ActionResultWitness(
        status=ActionResultStatus.SANCTIONED_MATCH,
        accepted=True,
        dissolved=True,
        rollback_required=False,
        update_required=False,
        inhibit_repetition=False,
        outcome_prediction_error=raw_ope,
        precision_weighted_outcome_error=pwope,
        comparator_error=comparator_error,
        value_prediction_error=rpe,
        action_prediction_error=ape,
        timing_prediction_error=tpe,
        adapted_error_threshold=adapted_error_threshold,
        next_context_expansion_required=False,
        reason=(
            f"SANCTIONED_MATCH: comparator_error={comparator_error:.6g} <= "
            f"error_threshold={expected.error_threshold:.6g}"
        ),
        falsifier=(
            "Comparator sanctioned an action whose outcome already exceeded the error "
            "threshold; mutation that flips the comparison must produce a different "
            "verdict."
        ),
    )


__all__ = [
    "ActionResultStatus",
    "ActionResultWitness",
    "ExpectedResultModel",
    "ObservedActionResult",
    "accept_action_result",
]
