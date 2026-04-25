# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
# no-bio-claim
"""Tests for INV-SIMULATION-FALSIFICATION (P1, statistical).

The simulation hypothesis is operationalized as a registry of enumerable
signatures with explicit thresholds and observation status. These tests
exercise the registry's contract — they do NOT perform any cosmological
measurement. The empirical observation status of each signature is
maintained by hand against published peer-reviewed references.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from core.physics.simulation_falsification import (
    CANONICAL_SIGNATURES,
    FalsificationLadder,
    FalsificationSignature,
    ObservationStatus,
    build_canonical_ladder,
)

# ---------------------------------------------------------------------------
# Canonical ladder shape
# ---------------------------------------------------------------------------


def test_canonical_ladder_has_six_pre_registered_signatures() -> None:
    """The pre-registered ladder ships with exactly the six documented entries."""
    ladder = build_canonical_ladder()
    assert len(ladder.signatures) == 6


def test_canonical_signature_ids_are_unique() -> None:
    """No duplicate signature_id in the canonical ladder."""
    ids = [s.signature_id for s in CANONICAL_SIGNATURES]
    assert len(ids) == len(set(ids)), f"duplicate signature_ids: {ids}"


def test_canonical_signatures_carry_published_reference() -> None:
    """Every signature must cite a peer-reviewed source — no naked claims."""
    for sig in CANONICAL_SIGNATURES:
        assert sig.reference, f"{sig.signature_id} has no reference"
        assert len(sig.reference) > 20, f"{sig.signature_id} reference too terse: {sig.reference!r}"


def test_canonical_thresholds_are_finite_and_positive() -> None:
    """Detectability thresholds must be finite positive numbers."""
    for sig in CANONICAL_SIGNATURES:
        assert math.isfinite(sig.detectability_threshold)
        msg = f"{sig.signature_id} non-positive threshold {sig.detectability_threshold}"
        assert sig.detectability_threshold > 0.0, msg


def test_canonical_status_values_are_valid_enum_members() -> None:
    """Status field must be an ObservationStatus enum value, not a free string."""
    valid = set(ObservationStatus)
    for sig in CANONICAL_SIGNATURES:
        assert sig.current_observation_status in valid


# ---------------------------------------------------------------------------
# status_summary
# ---------------------------------------------------------------------------


def test_status_summary_counts_match_registry() -> None:
    """status_summary returns one bucket per ObservationStatus, totals match."""
    ladder = build_canonical_ladder()
    summary = ladder.status_summary()
    assert set(summary.keys()) == {s.value for s in ObservationStatus}
    assert sum(summary.values()) == len(ladder.signatures)


def test_status_summary_returns_zero_for_unused_buckets() -> None:
    """Custom ladder with only NOT_OBSERVED entries reports zero elsewhere."""
    custom = FalsificationLadder(
        signatures=(
            FalsificationSignature(
                signature_id="TEST-A",
                name="A",
                prediction_under_simulation="x",
                detectability_threshold=1.0,
                detectability_units="u",
                current_observation_status=ObservationStatus.NOT_OBSERVED,
                current_observation_value=None,
                reference="ref",
            ),
        )
    )
    summary = custom.status_summary()
    assert summary["NOT_OBSERVED"] == 1
    assert summary["OPEN"] == 0
    assert summary["RULED_OUT"] == 0


# ---------------------------------------------------------------------------
# signature_by_id and rule_out
# ---------------------------------------------------------------------------


def test_signature_by_id_returns_canonical_entry() -> None:
    """Lookup by id returns the matching canonical signature."""
    ladder = build_canonical_ladder()
    sig = ladder.signature_by_id("SIM-HOLOGRAPHIC-SATURATION")
    assert sig.signature_id == "SIM-HOLOGRAPHIC-SATURATION"


def test_signature_by_id_unknown_raises_key_error() -> None:
    """Unknown signature_id is fail-closed: KeyError, not None."""
    ladder = build_canonical_ladder()
    with pytest.raises(KeyError):
        ladder.signature_by_id("SIM-DOES-NOT-EXIST")


def test_hardware_class_ruled_out_when_observed_above_threshold() -> None:
    """observed > threshold ⇒ hardware-class predicting <threshold is ruled out."""
    ladder = build_canonical_ladder()
    sig = ladder.signature_by_id("SIM-HOLOGRAPHIC-SATURATION")
    above = sig.detectability_threshold * 2.0
    assert ladder.hardware_class_ruled_out("SIM-HOLOGRAPHIC-SATURATION", above) is True


def test_hardware_class_not_ruled_out_when_observed_below_threshold() -> None:
    """observed <= threshold ⇒ no rule-out (consistent with simulation)."""
    ladder = build_canonical_ladder()
    sig = ladder.signature_by_id("SIM-HOLOGRAPHIC-SATURATION")
    below = sig.detectability_threshold * 0.5
    assert ladder.hardware_class_ruled_out("SIM-HOLOGRAPHIC-SATURATION", below) is False


def test_hardware_class_at_exact_threshold_not_ruled_out() -> None:
    """observed == threshold is the boundary; strict > required to rule out."""
    ladder = build_canonical_ladder()
    sig = ladder.signature_by_id("SIM-HOLOGRAPHIC-SATURATION")
    assert (
        ladder.hardware_class_ruled_out("SIM-HOLOGRAPHIC-SATURATION", sig.detectability_threshold)
        is False
    )


def test_hardware_class_ruled_out_non_finite_observed_raises() -> None:
    """NaN / Inf observation is fail-closed (INV-HPC2)."""
    ladder = build_canonical_ladder()
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            ladder.hardware_class_ruled_out("SIM-HOLOGRAPHIC-SATURATION", bad)


def test_hardware_class_ruled_out_unknown_signature_raises() -> None:
    """KeyError, not silent False, on unknown signature_id."""
    ladder = build_canonical_ladder()
    with pytest.raises(KeyError):
        ladder.hardware_class_ruled_out("SIM-DOES-NOT-EXIST", 1.0)


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_signature_dataclass_is_frozen() -> None:
    """FalsificationSignature is frozen — fields cannot be mutated post-construction."""
    sig = CANONICAL_SIGNATURES[0]
    with pytest.raises(AttributeError):
        sig.detectability_threshold = 0.0  # type: ignore[misc]


def test_ladder_dataclass_is_frozen() -> None:
    """FalsificationLadder is frozen — signatures tuple is immutable."""
    ladder = build_canonical_ladder()
    with pytest.raises(AttributeError):
        ladder.signatures = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Property
# ---------------------------------------------------------------------------


@given(observed=st.floats(allow_nan=False, allow_infinity=False))
def test_property_rule_out_iff_strictly_above_threshold(observed: float) -> None:
    """Property: rule_out is the boolean (observed > threshold) for any
    finite observation — no hidden tolerance, no silent clamp."""
    ladder = build_canonical_ladder()
    sig = ladder.signature_by_id("SIM-HOLOGRAPHIC-SATURATION")
    expected = observed > sig.detectability_threshold
    actual = ladder.hardware_class_ruled_out("SIM-HOLOGRAPHIC-SATURATION", observed)
    assert actual is expected
