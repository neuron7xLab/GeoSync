# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Deterministic ActionResultAcceptor for the GeoSync HPC control plane.

The acceptor enforces the chain
    expected -> observed -> error -> update / rollback / reentry
by pure, side-effect-free comparison.  No IO, no wall-clock, no random,
no mutable globals, no trading / execution / policy / forecast imports.

Engineering analogue only.  Names such as "expected_result",
"reverse_afferentation", "predicted_action_signature" are shorthand for
bounded mathematical objects described below.  No biological-equivalence
claim is made.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from dataclasses import dataclass


class ActionResultStatus(enum.StrEnum):
    """Stable status codes emitted by :func:`accept_action_result`.

    Order is significant only in :func:`accept_action_result`'s decision
    cascade; the enum itself does not impose semantic ordering.
    """

    SANCTIONED_MATCH = "SANCTIONED_MATCH"
    UPDATE_REQUIRED = "UPDATE_REQUIRED"
    ROLLBACK_REQUIRED = "ROLLBACK_REQUIRED"
    REENTRY_REQUIRED = "REENTRY_REQUIRED"
    INSUFFICIENT_OBSERVATION = "INSUFFICIENT_OBSERVATION"
    INSUFFICIENT_REVERSE_AFFERENTATION = "INSUFFICIENT_REVERSE_AFFERENTATION"
    ACTION_MISMATCH = "ACTION_MISMATCH"
    VALUE_MISMATCH = "VALUE_MISMATCH"
    TIMING_MISMATCH = "TIMING_MISMATCH"
    INVALID_INPUT = "INVALID_INPUT"


def _is_finite_float(value: float) -> bool:
    return math.isfinite(value)


def _all_finite(items: tuple[float, ...]) -> bool:
    return all(_is_finite_float(x) for x in items)


@dataclass(frozen=True, slots=True)
class ExpectedResultModel:
    """Pre-action expected outcome contract.

    Construction-time validation enforces structural invariants; the
    :func:`accept_action_result` entry point re-validates so that callers
    cannot bypass the contract by reaching into ``__dict__``.
    """

    action_id: str
    action_type: str
    expected_result: tuple[float, ...]
    context_signature: tuple[float, ...]
    prior_confidence: float
    error_threshold: float
    rollback_threshold: float
    reentry_threshold: float
    expected_value: float | None = None
    expected_latency_ms: float | None = None
    predicted_action_signature: tuple[float, ...] | None = None
    action_mismatch_threshold: float | None = None
    value_mismatch_threshold: float | None = None
    timing_mismatch_threshold_ms: float | None = None
    created_before_action: bool = True

    def __post_init__(self) -> None:
        _validate_expected(self)


@dataclass(frozen=True, slots=True)
class ObservedActionResult:
    """Post-action observed outcome bundle.

    ``success`` is recorded for downstream telemetry only; it never
    overrides numerical comparison against the expected model.
    """

    action_id: str
    observed_result: tuple[float, ...] | None
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

    All booleans are derived from :attr:`status` per the exact flag table
    in the module docstring.  ``reason`` always begins with one of the
    documented stable codes so downstream consumers can switch on a
    prefix rather than parse free-form text.
    """

    status: ActionResultStatus
    accepted: bool
    dissolved: bool
    reentry_required: bool
    rollback_required: bool
    update_required: bool
    inhibit_repetition: bool
    outcome_prediction_error: float | None
    value_prediction_error: float | None
    action_prediction_error: float | None
    timing_prediction_error: float | None
    precision_weighted_gain: float
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
    if model.created_before_action is not True:
        raise ValueError(
            "INVALID_EXPECTED_MODEL: created_before_action must be True (no post-hoc model)"
        )
    if not _is_finite_float(model.prior_confidence):
        raise ValueError("NON_FINITE_VALUE: prior_confidence must be finite")
    if not 0.0 <= model.prior_confidence <= 1.0:
        raise ValueError("INVALID_EXPECTED_MODEL: prior_confidence must lie in [0, 1]")
    if not _is_finite_float(model.error_threshold) or model.error_threshold < 0.0:
        raise ValueError("INVALID_EXPECTED_MODEL: error_threshold must be finite and >= 0")
    if not _is_finite_float(model.rollback_threshold) or model.rollback_threshold < 0.0:
        raise ValueError("INVALID_EXPECTED_MODEL: rollback_threshold must be finite and >= 0")
    if not _is_finite_float(model.reentry_threshold) or model.reentry_threshold < 0.0:
        raise ValueError("INVALID_EXPECTED_MODEL: reentry_threshold must be finite and >= 0")
    if model.rollback_threshold < model.error_threshold:
        raise ValueError(
            "INVALID_THRESHOLD_ORDERING: rollback_threshold must be >= error_threshold"
        )
    if model.reentry_threshold < model.rollback_threshold:
        raise ValueError(
            "INVALID_THRESHOLD_ORDERING: reentry_threshold must be >= rollback_threshold"
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


def _validate_error_history(history: tuple[float, ...]) -> None:
    if not isinstance(history, tuple):
        raise ValueError("INVALID_OBSERVED_RESULT: error_history must be a tuple")
    for sample in history:
        if not _is_finite_float(sample):
            raise ValueError("NON_FINITE_VALUE: error_history contains non-finite element")
        if sample < 0.0:
            raise ValueError("INVALID_OBSERVED_RESULT: error_history must be non-negative")


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _euclidean(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    total = 0.0
    for x, y in zip(a, b, strict=True):
        diff = x - y
        total += diff * diff
    return math.sqrt(total)


def _invalid_input_witness(reason: str, falsifier: str) -> ActionResultWitness:
    return ActionResultWitness(
        status=ActionResultStatus.INVALID_INPUT,
        accepted=False,
        dissolved=False,
        reentry_required=False,
        rollback_required=False,
        update_required=False,
        inhibit_repetition=True,
        outcome_prediction_error=None,
        value_prediction_error=None,
        action_prediction_error=None,
        timing_prediction_error=None,
        precision_weighted_gain=0.0,
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
        reentry_required=False,
        rollback_required=False,
        update_required=False,
        inhibit_repetition=True,
        outcome_prediction_error=None,
        value_prediction_error=None,
        action_prediction_error=None,
        timing_prediction_error=None,
        precision_weighted_gain=0.0,
        adapted_error_threshold=0.0,
        next_context_expansion_required=True,
        reason=reason,
        falsifier=falsifier,
    )


def accept_action_result(
    expected: ExpectedResultModel | None,
    observed: ObservedActionResult,
    *,
    error_history: tuple[float, ...] = (),
) -> ActionResultWitness:
    """Compare a single (expected, observed) pair and emit a witness.

    Decision priority (exact, no shuffling):
        1.  INVALID_INPUT (any failed validation, including missing
            expected model or non-finite / dimension / threshold issues).
        2.  INSUFFICIENT_REVERSE_AFFERENTATION
            (``observed.reverse_afferentation_present is False``).
        3.  INSUFFICIENT_OBSERVATION (``observed.observed_result is None``).
        4.  ACTION_MISMATCH on differing ``action_id``.
        5.  REENTRY_REQUIRED  if OPE >= reentry_threshold.
        6.  ROLLBACK_REQUIRED if OPE >= rollback_threshold.
        7.  ACTION_MISMATCH  on signature distance breach.
        8.  VALUE_MISMATCH   on |RPE| breach.
        9.  TIMING_MISMATCH  on |TPE| breach.
        10. UPDATE_REQUIRED  if OPE >  adapted_error_threshold.
        11. SANCTIONED_MATCH otherwise.

    The function is total: it never raises for normal invalid input;
    it returns a witness with status ``INVALID_INPUT`` instead.
    Exceptions are reserved for impossible programmer errors.
    """

    if expected is None:
        return _invalid_input_witness(
            reason="INVALID_EXPECTED_MODEL: expected is None",
            falsifier=(
                "Caller passed expected=None; acceptor would silently fabricate a default "
                "Prediction. Acceptor must fail closed."
            ),
        )

    try:
        _validate_expected(expected)
    except ValueError as exc:
        return _invalid_input_witness(
            reason=str(exc),
            falsifier=(
                "Caller mutated ExpectedResultModel internals or bypassed __post_init__; "
                "acceptor would proceed with malformed contract."
            ),
        )

    try:
        _validate_observed(observed)
    except ValueError as exc:
        return _invalid_input_witness(
            reason=str(exc),
            falsifier=(
                "Caller passed malformed ObservedActionResult; acceptor would compute "
                "errors against non-finite or wrongly typed observation."
            ),
        )

    try:
        _validate_error_history(error_history)
    except ValueError as exc:
        return _invalid_input_witness(
            reason=str(exc),
            falsifier=(
                "Caller passed an invalid error_history; acceptor would corrupt the adapted "
                "threshold via NaN, Inf, or negative samples."
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
                "Acceptor accepted an action without reverse afferentation; logged "
                "observation alone must never be sanctioned."
            ),
        )

    if observed.observed_result is None:
        return _insufficient_witness(
            ActionResultStatus.INSUFFICIENT_OBSERVATION,
            reason="MISSING_OBSERVATION: observed_result is None",
            falsifier=(
                "Acceptor accepted an action with no observed result vector; comparison "
                "is undefined."
            ),
        )

    if observed.action_id != expected.action_id:
        return ActionResultWitness(
            status=ActionResultStatus.ACTION_MISMATCH,
            accepted=False,
            dissolved=False,
            reentry_required=False,
            rollback_required=False,
            update_required=True,
            inhibit_repetition=True,
            outcome_prediction_error=None,
            value_prediction_error=None,
            action_prediction_error=None,
            timing_prediction_error=None,
            precision_weighted_gain=0.0,
            adapted_error_threshold=0.0,
            next_context_expansion_required=True,
            reason=(
                f"ACTION_ID_MISMATCH: expected={expected.action_id!r} "
                f"observed={observed.action_id!r}"
            ),
            falsifier=(
                "Acceptor compared two different actions; cross-pairing would silently "
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
                "Acceptor compared vectors of different shapes; Euclidean error would be "
                "ill-defined or zero-padded."
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
                "Acceptor compared action signatures of different shapes; APE would be ill-defined."
            ),
        )

    ope: float = _euclidean(observed.observed_result, expected.expected_result)

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

    precision_weighted_gain: float = _clamp(expected.prior_confidence * ope, 0.0, 1.0)

    if len(error_history) == 0:
        threshold_modifier: float = 1.0
    else:
        mean_history = sum(error_history) / float(len(error_history))
        threshold_modifier = _clamp(1.0 + 0.1 * mean_history, 0.75, 1.25)
    adapted_error_threshold: float = expected.error_threshold * threshold_modifier

    if ope >= expected.reentry_threshold:
        return ActionResultWitness(
            status=ActionResultStatus.REENTRY_REQUIRED,
            accepted=False,
            dissolved=False,
            reentry_required=True,
            rollback_required=True,
            update_required=False,
            inhibit_repetition=True,
            outcome_prediction_error=ope,
            value_prediction_error=rpe,
            action_prediction_error=ape,
            timing_prediction_error=tpe,
            precision_weighted_gain=precision_weighted_gain,
            adapted_error_threshold=adapted_error_threshold,
            next_context_expansion_required=True,
            reason=(
                f"REENTRY_REQUIRED: OPE={ope:.6g} >= reentry_threshold="
                f"{expected.reentry_threshold:.6g}"
            ),
            falsifier=(
                "Acceptor accepted an action whose outcome breached the reentry threshold; "
                "the controller must reopen the context, not silently update."
            ),
        )

    if ope >= expected.rollback_threshold:
        return ActionResultWitness(
            status=ActionResultStatus.ROLLBACK_REQUIRED,
            accepted=False,
            dissolved=False,
            reentry_required=False,
            rollback_required=True,
            update_required=False,
            inhibit_repetition=True,
            outcome_prediction_error=ope,
            value_prediction_error=rpe,
            action_prediction_error=ape,
            timing_prediction_error=tpe,
            precision_weighted_gain=precision_weighted_gain,
            adapted_error_threshold=adapted_error_threshold,
            next_context_expansion_required=True,
            reason=(
                f"ROLLBACK_REQUIRED: OPE={ope:.6g} >= rollback_threshold="
                f"{expected.rollback_threshold:.6g}"
            ),
            falsifier=(
                "Acceptor accepted an action whose outcome breached the rollback threshold; "
                "the controller must roll back, not update."
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
            reentry_required=False,
            rollback_required=False,
            update_required=True,
            inhibit_repetition=True,
            outcome_prediction_error=ope,
            value_prediction_error=rpe,
            action_prediction_error=ape,
            timing_prediction_error=tpe,
            precision_weighted_gain=precision_weighted_gain,
            adapted_error_threshold=adapted_error_threshold,
            next_context_expansion_required=True,
            reason=(
                f"ACTION_MISMATCH: APE={ape:.6g} > action_mismatch_threshold="
                f"{expected.action_mismatch_threshold:.6g}"
            ),
            falsifier=(
                "Executed action signature drifted beyond contract; acceptor cannot "
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
            reentry_required=False,
            rollback_required=False,
            update_required=True,
            inhibit_repetition=True,
            outcome_prediction_error=ope,
            value_prediction_error=rpe,
            action_prediction_error=ape,
            timing_prediction_error=tpe,
            precision_weighted_gain=precision_weighted_gain,
            adapted_error_threshold=adapted_error_threshold,
            next_context_expansion_required=True,
            reason=(
                f"VALUE_MISMATCH: |RPE|={abs(rpe):.6g} > value_mismatch_threshold="
                f"{expected.value_mismatch_threshold:.6g}"
            ),
            falsifier=(
                "Realized scalar value diverged beyond contract; acceptor cannot sanction "
                "the action without flagging update."
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
            reentry_required=False,
            rollback_required=False,
            update_required=True,
            inhibit_repetition=True,
            outcome_prediction_error=ope,
            value_prediction_error=rpe,
            action_prediction_error=ape,
            timing_prediction_error=tpe,
            precision_weighted_gain=precision_weighted_gain,
            adapted_error_threshold=adapted_error_threshold,
            next_context_expansion_required=True,
            reason=(
                f"TIMING_MISMATCH: |TPE|={abs(tpe):.6g} > timing_mismatch_threshold_ms="
                f"{expected.timing_mismatch_threshold_ms:.6g}"
            ),
            falsifier=(
                "Observed latency diverged beyond contract; acceptor cannot sanction "
                "out-of-window execution."
            ),
        )

    if ope > adapted_error_threshold:
        return ActionResultWitness(
            status=ActionResultStatus.UPDATE_REQUIRED,
            accepted=False,
            dissolved=False,
            reentry_required=False,
            rollback_required=False,
            update_required=True,
            inhibit_repetition=False,
            outcome_prediction_error=ope,
            value_prediction_error=rpe,
            action_prediction_error=ape,
            timing_prediction_error=tpe,
            precision_weighted_gain=precision_weighted_gain,
            adapted_error_threshold=adapted_error_threshold,
            next_context_expansion_required=True,
            reason=(
                f"UPDATE_REQUIRED: OPE={ope:.6g} > adapted_error_threshold="
                f"{adapted_error_threshold:.6g}"
            ),
            falsifier=(
                "Outcome error sits inside the safe band but above the adaptive threshold; "
                "acceptor must request a model update rather than dissolve the action."
            ),
        )

    return ActionResultWitness(
        status=ActionResultStatus.SANCTIONED_MATCH,
        accepted=True,
        dissolved=True,
        reentry_required=False,
        rollback_required=False,
        update_required=False,
        inhibit_repetition=False,
        outcome_prediction_error=ope,
        value_prediction_error=rpe,
        action_prediction_error=ape,
        timing_prediction_error=tpe,
        precision_weighted_gain=precision_weighted_gain,
        adapted_error_threshold=adapted_error_threshold,
        next_context_expansion_required=False,
        reason=(
            f"SANCTIONED_MATCH: OPE={ope:.6g} <= adapted_error_threshold="
            f"{adapted_error_threshold:.6g}"
        ),
        falsifier=(
            "Acceptor sanctioned an action whose outcome already exceeded the adaptive "
            "threshold; mutation that flips the comparison must produce a different verdict."
        ),
    )


# Re-export a tuple of public names for ``import *`` discipline; the actual
# package surface is set in ``geosync_hpc.control.__init__``.
__all__ = [
    "ActionResultStatus",
    "ActionResultWitness",
    "ExpectedResultModel",
    "ObservedActionResult",
    "accept_action_result",
]


# ``dataclasses`` is imported only to make :func:`dataclasses.fields` reachable
# for downstream introspection (used by tests asserting absence of forbidden
# fields).  Reference it here so that ``ruff`` does not flag the import as
# unused in environments where slots elide module-level attribute lookup.
_PUBLIC_DATACLASS_FIELDS: tuple[str, ...] = tuple(
    field.name
    for cls in (ExpectedResultModel, ObservedActionResult, ActionResultWitness)
    for field in dataclasses.fields(cls)
)
