# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
# no-bio-claim
"""Tests for INV-ANCHORED-SUBSTRATE-GATE (P0, conditional, ANCHORED).

Composes INV-BEKENSTEIN-COGNITIVE and INV-ARROW-OF-TIME. The
composition was verbal in PR descriptions; this test suite makes it
checkable.
"""

from __future__ import annotations

import math

import pytest

from core.physics.anchored_substrate_gate import (
    PROVENANCE_TIER,
    SubstrateGateInputs,
    assess_anchored_substrate_gate,
)
from core.physics.arrow_of_time import ObserverEntropyLedger
from core.physics.thermodynamic_budget import (
    SPEED_OF_LIGHT_M_S,
    bekenstein_cognitive_ceiling,
)


def _ledger(
    system_entropy_change_bits: float = 0.0,
    observer_information_gain_bits: float = 0.0,
) -> ObserverEntropyLedger:
    return ObserverEntropyLedger(
        system_entropy_change_bits=system_entropy_change_bits,
        observer_information_gain_bits=observer_information_gain_bits,
    )


def _inputs(
    radius_m: float = 0.07,
    mass_kg: float = 1.4,
    observed_information_bits: float = 0.0,
    system_entropy_change_bits: float = 0.0,
    observer_information_gain_bits: float = 0.0,
) -> SubstrateGateInputs:
    energy_J = mass_kg * SPEED_OF_LIGHT_M_S**2
    return SubstrateGateInputs(
        radius_m=radius_m,
        energy_J=energy_J,
        observed_information_bits=observed_information_bits,
        entropy_ledger=_ledger(
            system_entropy_change_bits=system_entropy_change_bits,
            observer_information_gain_bits=observer_information_gain_bits,
        ),
    )


def test_provenance_tier_is_anchored() -> None:
    """Gate composes only ANCHORED ingredients; tier must be ANCHORED."""
    assert PROVENANCE_TIER == "ANCHORED"


def test_admissible_when_both_axes_hold() -> None:
    """Brain-scale substrate, modest information, positive entropy production
    ⇒ both axes hold ⇒ admissible."""
    w = assess_anchored_substrate_gate(
        _inputs(observed_information_bits=1e30, system_entropy_change_bits=1.0)
    )
    assert w.bekenstein_axis_holds is True
    assert w.arrow_axis_holds is True
    assert w.is_thermodynamically_admissible is True
    assert w.failure_axes == ()
    assert w.reason is None


def test_inadmissible_when_information_exceeds_bekenstein_ceiling() -> None:
    """Brain at 1.4 kg / 7 cm has ceiling ~ 10^42 bits. Claim 10^60 ⇒
    Bekenstein fails."""
    w = assess_anchored_substrate_gate(
        _inputs(observed_information_bits=1e60, system_entropy_change_bits=1.0)
    )
    assert w.bekenstein_axis_holds is False
    assert w.arrow_axis_holds is True
    assert w.is_thermodynamically_admissible is False
    assert w.failure_axes == ("BEKENSTEIN",)
    assert w.reason is not None
    assert "INV-BEKENSTEIN-COGNITIVE" in w.reason


def test_inadmissible_when_arrow_violated_by_unpaid_local_reduction() -> None:
    """ΔS_system = -2 with ΔI_observer = 0 ⇒ arrow violated, bekenstein OK."""
    w = assess_anchored_substrate_gate(
        _inputs(
            observed_information_bits=1e30,
            system_entropy_change_bits=-2.0,
            observer_information_gain_bits=0.0,
        )
    )
    assert w.bekenstein_axis_holds is True
    assert w.arrow_axis_holds is False
    assert w.is_thermodynamically_admissible is False
    assert w.failure_axes == ("ARROW",)
    assert w.reason is not None
    assert "INV-ARROW-OF-TIME" in w.reason


def test_both_axes_fail_simultaneously() -> None:
    """Construct inputs that violate both anchored bounds."""
    w = assess_anchored_substrate_gate(
        _inputs(
            observed_information_bits=1e60,
            system_entropy_change_bits=-2.0,
            observer_information_gain_bits=0.0,
        )
    )
    assert w.bekenstein_axis_holds is False
    assert w.arrow_axis_holds is False
    assert w.is_thermodynamically_admissible is False
    assert set(w.failure_axes) == {"BEKENSTEIN", "ARROW"}
    assert w.reason is not None
    assert "INV-BEKENSTEIN-COGNITIVE" in w.reason
    assert "INV-ARROW-OF-TIME" in w.reason


def test_bekenstein_saturation_admissible_at_equality() -> None:
    """Information exactly at ceiling ⇒ admissible (boundary case)."""
    radius = 0.07
    mass = 1.4
    energy = mass * SPEED_OF_LIGHT_M_S**2
    ceiling = bekenstein_cognitive_ceiling(radius, energy)
    w = assess_anchored_substrate_gate(
        SubstrateGateInputs(
            radius_m=radius,
            energy_J=energy,
            observed_information_bits=ceiling,
            entropy_ledger=_ledger(system_entropy_change_bits=0.0),
        )
    )
    assert w.bekenstein_axis_holds is True
    assert w.is_thermodynamically_admissible is True


def test_negative_observed_information_raises() -> None:
    """INV-HPC2: negative information is unphysical."""
    with pytest.raises(ValueError):
        assess_anchored_substrate_gate(_inputs(observed_information_bits=-1.0))


def test_non_finite_observed_information_raises() -> None:
    """INV-HPC2: NaN / Inf bit count is fail-closed."""
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            assess_anchored_substrate_gate(_inputs(observed_information_bits=bad))


def test_witness_dataclass_is_frozen() -> None:
    """AnchoredSubstrateGateWitness is immutable post-construction."""
    w = assess_anchored_substrate_gate(_inputs(observed_information_bits=1.0))
    with pytest.raises(AttributeError):
        w.is_thermodynamically_admissible = False  # type: ignore[misc]


def test_witness_carries_per_axis_sub_witness() -> None:
    """The composite witness exposes the arrow-of-time sub-witness for caller
    introspection (not just a boolean)."""
    w = assess_anchored_substrate_gate(_inputs(observed_information_bits=1.0))
    assert hasattr(w.arrow_witness, "is_arrow_consistent")
    assert hasattr(w.arrow_witness, "net_entropy_production_bits")


def test_failure_axes_ordering_is_stable() -> None:
    """`failure_axes` is a tuple of strings; ordering is BEKENSTEIN before
    ARROW when both fail (stable for downstream consumers)."""
    w = assess_anchored_substrate_gate(
        _inputs(
            observed_information_bits=1e60,
            system_entropy_change_bits=-2.0,
        )
    )
    assert w.failure_axes == ("BEKENSTEIN", "ARROW")


def test_zero_information_with_zero_entropy_is_admissible() -> None:
    """Trivial substrate (no claims, no change) is degenerate but admissible:
    nothing exceeds the ceiling, nothing violates the arrow."""
    w = assess_anchored_substrate_gate(_inputs(observed_information_bits=0.0))
    assert w.is_thermodynamically_admissible is True
    assert math.isfinite(w.bekenstein_ceiling_bits)
