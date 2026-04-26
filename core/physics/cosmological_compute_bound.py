# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Cosmological compute bound on a causal diamond (P4).

INV-COSMOLOGICAL-COMPUTE (P1, statistical, EXTRAPOLATED):
    The total useful reversible computation that can be performed
    inside any causal diamond is bounded above by the holographic bit
    capacity of its bounding horizon, scaled by an efficiency
    coefficient ε ∈ (0, 1]:

        I_useful_max = ε · A / (4 · ℓ_p² · ln 2)   bits

    where A is the horizon area and ℓ_p is the Planck length. The
    Bekenstein-Hawking formula gives I_max = A / (4 · ℓ_p²) nats =
    A / (4 · ℓ_p² · ln 2) bits. The efficiency ε ≤ 1 absorbs the
    practical fact that not all holographic degrees of freedom are
    addressable as useful logical bits at any given epoch — the
    de Sitter complexity literature places ε in the 10^-3 .. 1 range
    depending on the construction.

Provenance: EXTRAPOLATED.
    - Bekenstein 1973 (PRD 7, 2333): black-hole entropy ∝ horizon area.
    - 't Hooft 1993 (gr-qc/9310026): dimensional reduction.
    - Susskind 1995 (J. Math. Phys. 36, 6377): holographic principle.
    - Susskind 2014 (Fortschr. Phys. 64, 24, arXiv:1402.5674):
      computational complexity and black-hole horizons.

The holographic-bound side is settled physics. The interpretation as
a hard ceiling on "useful" cosmological computation is a research
direction; the efficiency coefficient ε cannot be derived from first
principles in this module. Truth-coherence ~0.5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from core.physics.thermodynamic_budget import LANDAUER_LN2

__all__ = [
    "BH_BIT_COEFF",
    "CausalDiamond",
    "ComputeBudgetWitness",
    "ComputeBudget",
    "PLANCK_LENGTH_M",
    "PROVENANCE_TIER",
    "assess_compute_claim",
    "diamond_compute_budget",
    "holographic_bit_capacity",
]

# Discrete provenance tier; see arrow_of_time.py for definition.
PROVENANCE_TIER: Literal["ANCHORED", "EXTRAPOLATED", "SPECULATIVE"] = "EXTRAPOLATED"

# CODATA 2018 Planck length (square root of ℏ·G/c³).
PLANCK_LENGTH_M: float = 1.616_255e-35
# Bekenstein-Hawking bit coefficient: I_max = A · BH_BIT_COEFF [bits, A in m²].
# = 1 / (4 · ℓ_p² · ln 2)
BH_BIT_COEFF: float = 1.0 / (4.0 * (PLANCK_LENGTH_M**2) * LANDAUER_LN2)


@dataclass(frozen=True, slots=True)
class CausalDiamond:
    """A causal diamond defined by its horizon area or a derived radius.

    horizon_area_m2 must be finite and non-negative. The diamond's
    interior bulk volume is not needed for the holographic bound.
    """

    horizon_area_m2: float


@dataclass(frozen=True, slots=True)
class ComputeBudget:
    """Holographic compute budget of a causal diamond."""

    diamond: CausalDiamond
    holographic_max_bits: float
    efficiency: float
    useful_max_bits: float


@dataclass(frozen=True, slots=True)
class ComputeBudgetWitness:
    """Diagnostic for INV-COSMOLOGICAL-COMPUTE on a single claim."""

    budget: ComputeBudget
    claimed_bits: float
    margin_bits: float
    is_within_budget: bool
    reason: str | None


def _check_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(
            f"INV-HPC2 VIOLATED: {name} must be finite, got {value!r}. "
            "Finite inputs → finite outputs is a P0 contract; no silent repair."
        )


def _check_non_negative(value: float, name: str) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def holographic_bit_capacity(area_m2: float) -> float:
    """Bekenstein-Hawking bit count for a horizon of area `area_m2`.

    I_max = A / (4 · ℓ_p² · ln 2) bits. Returns 0.0 for area = 0.
    Fail-closed on negative or non-finite inputs.
    """
    _check_finite(area_m2, "area_m2")
    _check_non_negative(area_m2, "area_m2")
    bits: float = BH_BIT_COEFF * area_m2
    _check_finite(bits, "holographic_bit_capacity")
    return bits


def diamond_compute_budget(
    diamond: CausalDiamond,
    *,
    efficiency: float = 1.0,
) -> ComputeBudget:
    """Compute the holographic budget for a causal diamond.

    `efficiency` is ε ∈ (0, 1] — the fraction of holographic bits
    treated as useful logical-compute. Default 1.0 reproduces the
    raw Bekenstein-Hawking bound. Values < 0 or > 1 are fail-closed.
    """
    _check_finite(efficiency, "efficiency")
    if not (0.0 < efficiency <= 1.0):
        raise ValueError(f"efficiency must be in (0, 1], got {efficiency}")
    holographic = holographic_bit_capacity(diamond.horizon_area_m2)
    useful = holographic * efficiency
    _check_finite(useful, "useful_max_bits")
    return ComputeBudget(
        diamond=diamond,
        holographic_max_bits=holographic,
        efficiency=efficiency,
        useful_max_bits=useful,
    )


def assess_compute_claim(
    diamond: CausalDiamond,
    claimed_bits: float,
    *,
    efficiency: float = 1.0,
) -> ComputeBudgetWitness:
    """Witness for INV-COSMOLOGICAL-COMPUTE on one claim.

    Returns a non-raising witness; caller fail-closes on
    `is_within_budget is False`.
    """
    _check_finite(claimed_bits, "claimed_bits")
    _check_non_negative(claimed_bits, "claimed_bits")
    budget = diamond_compute_budget(diamond, efficiency=efficiency)
    margin = budget.useful_max_bits - claimed_bits
    within = claimed_bits <= budget.useful_max_bits
    reason: str | None
    if within:
        reason = None
    else:
        reason = (
            "INV-COSMOLOGICAL-COMPUTE: claim exceeds budget; "
            f"claimed_bits={claimed_bits} > useful_max_bits={budget.useful_max_bits} "
            f"(holographic={budget.holographic_max_bits}, efficiency={efficiency})"
        )
    return ComputeBudgetWitness(
        budget=budget,
        claimed_bits=claimed_bits,
        margin_bits=margin,
        is_within_budget=within,
        reason=reason,
    )
