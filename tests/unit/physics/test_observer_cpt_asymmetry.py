# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
# no-bio-claim
"""Tests for INV-OBSERVER-CPT (P2, qualitative, SPECULATIVE).

This module is schema-only. It enforces the contract that any proposal
of "observer-induced effective baryon asymmetry in a CPT-symmetric
universe" must declare a concrete asymmetric decoherence kernel.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from core.physics.observer_cpt_asymmetry import (
    OBSERVED_BARYON_ASYMMETRY,
    PROVENANCE_LEVEL,
    TRUTH_COHERENCE_SCORE,
    DecoherenceKernel,
    assess_observer_cpt,
    asymmetry_from_kernel,
)


def test_provenance_metadata_is_speculative() -> None:
    """Provenance must mark this as SPECULATIVE — no specific kernel cited."""
    assert PROVENANCE_LEVEL == "SPECULATIVE"
    assert 0.1 <= TRUTH_COHERENCE_SCORE <= 0.45


def test_observed_baryon_asymmetry_constant_is_finite_and_small() -> None:
    """Sanity: η_observed ~ 6e-10 from BBN/CMB."""
    assert math.isfinite(OBSERVED_BARYON_ASYMMETRY)
    assert 1e-11 < OBSERVED_BARYON_ASYMMETRY < 1e-8


def test_symmetric_kernel_yields_zero_asymmetry() -> None:
    """K(matter) == K(antimatter) ⇒ η = 0."""
    k = DecoherenceKernel(matter_rate_hz=1.0, antimatter_rate_hz=1.0)
    assert asymmetry_from_kernel(k) == 0.0


def test_asymmetric_kernel_yields_non_zero_asymmetry() -> None:
    """K(matter) > K(antimatter) ⇒ η > 0."""
    k = DecoherenceKernel(matter_rate_hz=1.5, antimatter_rate_hz=0.5)
    eta = asymmetry_from_kernel(k)
    assert eta == 0.5
    assert eta > 0.0


def test_zero_zero_kernel_yields_zero_asymmetry() -> None:
    """No observation at all ⇒ no asymmetry to report (degenerate case)."""
    k = DecoherenceKernel(matter_rate_hz=0.0, antimatter_rate_hz=0.0)
    assert asymmetry_from_kernel(k) == 0.0


def test_negative_rate_raises() -> None:
    """Negative rate is unphysical — fail-closed."""
    with pytest.raises(ValueError):
        asymmetry_from_kernel(DecoherenceKernel(matter_rate_hz=-1.0, antimatter_rate_hz=1.0))


def test_non_finite_rate_raises() -> None:
    """INV-HPC2: NaN / Inf rates are fail-closed."""
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            asymmetry_from_kernel(DecoherenceKernel(matter_rate_hz=bad, antimatter_rate_hz=1.0))


def test_population_per_rate_non_positive_raises() -> None:
    """The scale factor must be strictly positive — zero or negative is fail-closed."""
    k = DecoherenceKernel(matter_rate_hz=1.0, antimatter_rate_hz=1.0)
    with pytest.raises(ValueError):
        asymmetry_from_kernel(k, population_per_rate=0.0)
    with pytest.raises(ValueError):
        asymmetry_from_kernel(k, population_per_rate=-1.0)


def test_witness_symmetric_kernel_is_consistent() -> None:
    """Symmetric kernel ⇒ η=0 ⇒ contract consistent."""
    k = DecoherenceKernel(matter_rate_hz=1.0, antimatter_rate_hz=1.0)
    w = assess_observer_cpt(k)
    assert w.is_kernel_cpt_symmetric is True
    assert w.is_asymmetry_zero is True
    assert w.is_contract_consistent is True
    assert w.reason is None


def test_witness_asymmetric_kernel_is_consistent() -> None:
    """Asymmetric kernel ⇒ η≠0 ⇒ contract consistent."""
    k = DecoherenceKernel(matter_rate_hz=2.0, antimatter_rate_hz=1.0)
    w = assess_observer_cpt(k)
    assert w.is_kernel_cpt_symmetric is False
    assert w.is_asymmetry_zero is False
    assert w.is_contract_consistent is True


def test_witness_dataclass_is_frozen() -> None:
    """ObserverCPTWitness is immutable post-construction."""
    k = DecoherenceKernel(matter_rate_hz=1.0, antimatter_rate_hz=1.0)
    w = assess_observer_cpt(k)
    with pytest.raises(AttributeError):
        w.is_contract_consistent = False  # type: ignore[misc]


def test_assess_negative_tolerance_raises() -> None:
    """Tolerances must be non-negative."""
    k = DecoherenceKernel(matter_rate_hz=1.0, antimatter_rate_hz=1.0)
    with pytest.raises(ValueError):
        assess_observer_cpt(k, kernel_symmetry_tol=-1.0)
    with pytest.raises(ValueError):
        assess_observer_cpt(k, asymmetry_tol=-1.0)


@given(
    matter=st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False),
    antimatter=st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False),
)
def test_property_eta_in_unit_interval(matter: float, antimatter: float) -> None:
    """Property: η = (a-b)/(a+b) ∈ [-1, 1] for any non-negative finite a, b."""
    k = DecoherenceKernel(matter_rate_hz=matter, antimatter_rate_hz=antimatter)
    eta = asymmetry_from_kernel(k)
    assert -1.0 <= eta <= 1.0


@given(
    matter=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    antimatter=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
)
def test_property_witness_consistency_iff_kernel_eta_match(
    matter: float, antimatter: float
) -> None:
    """Property: contract consistent iff (kernel symmetric ↔ η ≈ 0)."""
    k = DecoherenceKernel(matter_rate_hz=matter, antimatter_rate_hz=antimatter)
    w = assess_observer_cpt(k)
    assert w.is_contract_consistent is (w.is_kernel_cpt_symmetric is w.is_asymmetry_zero)
