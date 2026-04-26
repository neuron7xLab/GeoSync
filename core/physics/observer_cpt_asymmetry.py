# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Observer-induced effective asymmetry in CPT-symmetric dynamics (P5).

INV-OBSERVER-CPT (P2, qualitative, SPECULATIVE):
    For a CPT-symmetric Lagrangian coupled to an observer with an
    asymmetric decoherence kernel K, the observed matter / antimatter
    counts integrated over a window are equal iff K is CPT-symmetric:

        η_observed = (N_matter - N_antimatter) / (N_matter + N_antimatter)
        η_observed = 0   iff   K(matter) = K(antimatter)

    Equivalently: any observed baryon-asymmetry-like signal η ≠ 0 in a
    Lagrangian-CPT-symmetric universe REQUIRES an observer with
    K(matter) ≠ K(antimatter). This module does NOT derive that such
    a kernel exists in our universe; it operationalizes the contract
    so that any future model proposing one must declare a concrete
    asymmetric K and pass falsification.

Provenance: SPECULATIVE.
    Sakharov 1967 (JETP Lett. 5, 24) lists three conditions for
    baryogenesis in standard Big Bang cosmology: (i) baryon-number
    violation, (ii) C and CP violation, (iii) departure from thermal
    equilibrium. The observer-asymmetry route is an alternative
    framing that requires the observer's coupling to break a discrete
    symmetry without modifying the Lagrangian. No specific peer-
    reviewed model of such a kernel for our universe is cited here;
    this module is schema only. Truth-coherence ~0.3.

The intended use of this module is to keep proposals of "the universe
is CPT-symmetric and our observed baryon asymmetry comes from
observer-side decoherence" honest: any such proposal must supply a
concrete kernel K(matter) and K(antimatter) satisfying the contract
and produce a measurable η consistent with cosmological observation
(η_observed ≈ 6 × 10^-10 from BBN / CMB).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "DecoherenceKernel",
    "ObservedAsymmetry",
    "ObserverCPTWitness",
    "PROVENANCE_LEVEL",
    "TRUTH_COHERENCE_SCORE",
    "asymmetry_from_kernel",
    "assess_observer_cpt",
]

PROVENANCE_LEVEL: str = "SPECULATIVE"
TRUTH_COHERENCE_SCORE: float = 0.3

# Observed baryon asymmetry parameter η = (n_b - n_b̄) / n_γ from BBN/CMB.
# Order ~6e-10 (Planck Collaboration 2018; PDG 2024 review).
OBSERVED_BARYON_ASYMMETRY: float = 6.0e-10


@dataclass(frozen=True, slots=True)
class DecoherenceKernel:
    """An asymmetric decoherence kernel applied by an observer.

    Both rates are in Hz (1/s). A CPT-symmetric kernel has equal rates;
    observer-induced asymmetry exists when the rates differ.
    """

    matter_rate_hz: float
    antimatter_rate_hz: float


@dataclass(frozen=True, slots=True)
class ObservedAsymmetry:
    """Cumulative matter / antimatter counts integrated over a window."""

    n_matter: float
    n_antimatter: float


@dataclass(frozen=True, slots=True)
class ObserverCPTWitness:
    """Diagnostic for INV-OBSERVER-CPT.

    Fields:
    - kernel: the asymmetric or symmetric decoherence kernel
    - asymmetry: derived (n_m - n_m̄) / (n_m + n_m̄)
    - is_kernel_cpt_symmetric: K(matter) == K(antimatter) within tolerance
    - is_asymmetry_zero: η == 0 within tolerance
    - is_contract_consistent:
        (kernel symmetric AND η == 0) OR (kernel asymmetric AND η != 0)
    """

    kernel: DecoherenceKernel
    asymmetry: float
    is_kernel_cpt_symmetric: bool
    is_asymmetry_zero: bool
    is_contract_consistent: bool
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


def asymmetry_from_kernel(
    kernel: DecoherenceKernel,
    *,
    population_per_rate: float = 1.0,
) -> float:
    """Derive η = (n_m - n_m̄) / (n_m + n_m̄) from kernel rates.

    Convention: the cumulative counts after a unit window are
    proportional to the kernel rates. `population_per_rate` is a
    multiplicative scale that cancels in the ratio; included only
    for unit-handling clarity at higher-level callers.

    Returns 0.0 when both rates are exactly zero (no observation —
    no asymmetry to report).
    """
    _check_finite(kernel.matter_rate_hz, "kernel.matter_rate_hz")
    _check_finite(kernel.antimatter_rate_hz, "kernel.antimatter_rate_hz")
    _check_non_negative(kernel.matter_rate_hz, "kernel.matter_rate_hz")
    _check_non_negative(kernel.antimatter_rate_hz, "kernel.antimatter_rate_hz")
    _check_finite(population_per_rate, "population_per_rate")
    if population_per_rate <= 0.0:
        raise ValueError(f"population_per_rate must be > 0, got {population_per_rate}")
    n_matter = kernel.matter_rate_hz * population_per_rate
    n_antimatter = kernel.antimatter_rate_hz * population_per_rate
    total = n_matter + n_antimatter
    if total == 0.0:
        return 0.0
    return (n_matter - n_antimatter) / total


def assess_observer_cpt(
    kernel: DecoherenceKernel,
    *,
    population_per_rate: float = 1.0,
    kernel_symmetry_tol: float = 1e-12,
    asymmetry_tol: float = 1e-12,
) -> ObserverCPTWitness:
    """Witness for INV-OBSERVER-CPT on one kernel.

    Reports both the kernel symmetry status and the resulting
    observed-asymmetry status, and confirms they agree:

        kernel CPT-symmetric  iff  η == 0

    The witness is non-raising; caller fail-closes on
    `is_contract_consistent is False`.
    """
    _check_finite(kernel_symmetry_tol, "kernel_symmetry_tol")
    _check_finite(asymmetry_tol, "asymmetry_tol")
    if kernel_symmetry_tol < 0.0 or asymmetry_tol < 0.0:
        raise ValueError("tolerances must be non-negative")
    eta = asymmetry_from_kernel(kernel, population_per_rate=population_per_rate)
    delta_kernel = abs(kernel.matter_rate_hz - kernel.antimatter_rate_hz)
    is_symmetric = delta_kernel <= kernel_symmetry_tol
    is_zero = abs(eta) <= asymmetry_tol
    consistent = is_symmetric == is_zero
    reason: str | None
    if consistent:
        reason = None
    else:
        reason = (
            "INV-OBSERVER-CPT: kernel-symmetry / observed-asymmetry mismatch — "
            f"|ΔK|={delta_kernel} (tol {kernel_symmetry_tol}), "
            f"|η|={abs(eta)} (tol {asymmetry_tol}); "
            "a CPT-symmetric kernel must produce η=0, and any η≠0 must "
            "carry an asymmetric kernel."
        )
    return ObserverCPTWitness(
        kernel=kernel,
        asymmetry=eta,
        is_kernel_cpt_symmetric=is_symmetric,
        is_asymmetry_zero=is_zero,
        is_contract_consistent=consistent,
        reason=reason,
    )
