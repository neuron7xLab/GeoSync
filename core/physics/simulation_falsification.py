# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Simulation hypothesis as a physical, experimentally testable problem.

This module does NOT compute P(we are in a simulation). That framing is
philosophical and unfalsifiable. Instead it enumerates a set of concrete
physical signatures whose presence or absence would either:

  (a) constrain the hardware class on which a simulation could run, or
  (b) rule out a finite-resolution simulation at currently observable scales.

INV-SIMULATION-FALSIFICATION (P1, statistical):
    The simulation hypothesis is operationalized as a registry of
    enumerable signatures, each with a published prediction, an
    explicit detectability threshold, and a current observation
    status. The hypothesis is *falsifiable* iff at least one such
    signature has a threshold within reach of current or near-future
    instruments AND a documented null observation up to that
    threshold.

References:

- Bekenstein, J. D. (1981). Universal upper bound on the entropy-to-energy
  ratio for bounded systems. Phys. Rev. D 23, 287.
- 't Hooft, G. (1993). Dimensional reduction in quantum gravity.
  arXiv:gr-qc/9310026.
- Susskind, L. (1995). The world as a hologram. J. Math. Phys. 36, 6377.
- Beane, S. R., Davoudi, Z., Savage, M. J. (2014). Constraints on the
  universe as a numerical simulation. Eur. Phys. J. A 50, 148.
  arXiv:1210.1847.

This module is opt-in research infrastructure. It makes no claim about
whether the universe is simulated; it only mechanizes the logical
contract that *any* such claim must reduce to enumerable signatures
with measurable thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

__all__ = [
    "CANONICAL_SIGNATURES",
    "FalsificationLadder",
    "FalsificationSignature",
    "ObservationStatus",
    "build_canonical_ladder",
]


class ObservationStatus(str, Enum):
    """Discrete states of a falsification signature against observation.

    NOT_OBSERVED: signature has been searched for at least to its
        detectability threshold and no positive detection exists.
        This *does not* refute the simulation hypothesis; it bounds
        a hardware class above the threshold.
    OPEN: the signature is below current instrument reach. Neither
        confirmed nor refuted; included to make the ladder honest
        about its blind spots.
    RULED_OUT: a positive detection at or above threshold has been
        reported and independently corroborated. Continuous-physics
        models that forbid the signature are then constrained.
    """

    NOT_OBSERVED = "NOT_OBSERVED"
    OPEN = "OPEN"
    RULED_OUT = "RULED_OUT"


@dataclass(frozen=True, slots=True)
class FalsificationSignature:
    """One enumerable signature linking the simulation hypothesis to a
    measurable physical observable.

    Each signature is a hard contract: a violation has units, a
    threshold, and a published reference. Probability is not a field;
    only status against observation.
    """

    signature_id: str
    name: str
    prediction_under_simulation: str
    detectability_threshold: float
    detectability_units: str
    current_observation_status: ObservationStatus
    current_observation_value: float | None
    reference: str


@dataclass(frozen=True, slots=True)
class FalsificationLadder:
    """An immutable enumeration of falsification signatures.

    The ladder is *not* an inference engine. It is a registry whose
    purpose is to make implicit assumptions about the simulation
    hypothesis explicit and testable. A ladder is honest if every
    signature has either a published null bound (NOT_OBSERVED) or a
    documented inaccessibility (OPEN).
    """

    signatures: tuple[FalsificationSignature, ...]

    def status_summary(self) -> dict[str, int]:
        """Return counts of signatures by ObservationStatus.

        Useful for triage: a ladder dominated by OPEN entries is not
        yet a productive falsification surface.
        """
        counts: dict[str, int] = {s.value: 0 for s in ObservationStatus}
        for sig in self.signatures:
            counts[sig.current_observation_status.value] += 1
        return counts

    def signature_by_id(self, signature_id: str) -> FalsificationSignature:
        """O(N) lookup. Raises KeyError if no such signature."""
        for sig in self.signatures:
            if sig.signature_id == signature_id:
                return sig
        raise KeyError(f"unknown signature_id: {signature_id!r}")

    def hardware_class_ruled_out(self, signature_id: str, observed_value: float) -> bool:
        """Return True iff `observed_value` exceeds the threshold of the
        named signature, i.e. the simulation hardware class predicting
        signal below threshold is ruled out by the observation.

        Raises:
            KeyError: signature_id not in ladder.
            ValueError: observed_value is non-finite.
        """
        if not _is_finite(observed_value):
            raise ValueError(f"observed_value must be finite, got {observed_value!r}")
        sig = self.signature_by_id(signature_id)
        return observed_value > sig.detectability_threshold


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


# ---------------------------------------------------------------------------
# Canonical ladder — six pre-registered signatures.
#
# Each entry cites a peer-reviewed source. The thresholds are taken
# from the published bounds; the status reflects the latest published
# measurement at the time of registration. This registry is intended
# to be amended only by appending new signatures or by updating
# `current_observation_status` when a new measurement is published —
# never by silently editing thresholds.
# ---------------------------------------------------------------------------


_GZK_CUTOFF_EV: Final[float] = 5.0e19  # Greisen-Zatsepin-Kuzmin cutoff
_PLANCK_LENGTH_M: Final[float] = 1.616_255e-35


CANONICAL_SIGNATURES: Final[tuple[FalsificationSignature, ...]] = (
    FalsificationSignature(
        signature_id="SIM-HOLOGRAPHIC-SATURATION",
        name="Holographic information-bound saturation",
        prediction_under_simulation=(
            "If the substrate is a finite-resolution simulation, no "
            "physical region should store I > 2π·E·R/(ℏ·c·ln 2) bits. "
            "Saturation has only been observed in black-hole entropy "
            "calculations; any non-BH system at saturation would be a "
            "direct measurement of the substrate's resolution limit."
        ),
        detectability_threshold=1.0,
        detectability_units="ratio I_observed / I_max(Bekenstein)",
        current_observation_status=ObservationStatus.NOT_OBSERVED,
        current_observation_value=None,
        reference=(
            "Bekenstein 1981 (Phys. Rev. D 23, 287); "
            "'t Hooft 1993 (gr-qc/9310026); "
            "Susskind 1995 (J. Math. Phys. 36, 6377)."
        ),
    ),
    FalsificationSignature(
        signature_id="SIM-LATTICE-UHECR",
        name="Cubic-lattice anisotropy in ultra-high-energy cosmic rays",
        prediction_under_simulation=(
            "A simulation on a cubic spacetime lattice would imprint "
            "preferred-direction structure on cosmic rays above the "
            "GZK cutoff (~5×10^19 eV). Continuous physics predicts "
            "isotropy at the percent level after deflection."
        ),
        detectability_threshold=_GZK_CUTOFF_EV,
        detectability_units="eV (cosmic-ray energy threshold)",
        current_observation_status=ObservationStatus.NOT_OBSERVED,
        current_observation_value=None,
        reference=(
            "Beane, Davoudi, Savage (2014). "
            "Constraints on the universe as a numerical simulation. "
            "Eur. Phys. J. A 50, 148. arXiv:1210.1847."
        ),
    ),
    FalsificationSignature(
        signature_id="SIM-PLANCK-DISCRETIZATION",
        name="Direct probe of spacetime discretization at Planck scale",
        prediction_under_simulation=(
            "If the simulation grid is Planck-scale, energies above "
            "~10^28 eV should reveal pixelation or grid axes. Current "
            "instruments are 14 orders of magnitude below this scale; "
            "the signature is enumerated to keep the ladder honest "
            "about its inaccessibility."
        ),
        detectability_threshold=_PLANCK_LENGTH_M,
        detectability_units="m (probed length scale)",
        current_observation_status=ObservationStatus.OPEN,
        current_observation_value=None,
        reference=("'t Hooft 1993 (gr-qc/9310026); Susskind 1995 (J. Math. Phys. 36, 6377)."),
    ),
    FalsificationSignature(
        signature_id="SIM-LATTICE-DISPERSION",
        name="Energy-dependent photon dispersion from cosmological sources",
        prediction_under_simulation=(
            "A discretized substrate would induce energy-dependent "
            "differences in photon arrival times from gamma-ray bursts "
            "and other transients at MeV–GeV energies. Continuous "
            "physics predicts arrival simultaneity to first order."
        ),
        detectability_threshold=1.0,
        detectability_units="ratio Δt_observed / Δt_predicted_lattice",
        current_observation_status=ObservationStatus.NOT_OBSERVED,
        current_observation_value=None,
        reference=("Beane, Davoudi, Savage (2014), §IV. Eur. Phys. J. A 50, 148."),
    ),
    FalsificationSignature(
        signature_id="SIM-COMPUTE-COMPLEXITY-WALL",
        name="Asymptotic wall in deep quantum-circuit fidelity",
        prediction_under_simulation=(
            "A finite-hardware substrate must asymptotically degrade "
            "fidelity of large quantum circuits beyond some qubit "
            "count or depth. Continuous physics predicts no such "
            "wall — only environmental decoherence governs fidelity."
        ),
        detectability_threshold=1.0,
        detectability_units="ratio degradation_observed / decoherence_predicted",
        current_observation_status=ObservationStatus.OPEN,
        current_observation_value=None,
        reference=(
            "Bekenstein-bound holography ('t Hooft 1993; Susskind 1995) "
            "applied to quantum-information capacity."
        ),
    ),
    FalsificationSignature(
        signature_id="SIM-CMB-MULTIPOLE-CUTOFF",
        name="Power-spectrum cutoff at high CMB multipole",
        prediction_under_simulation=(
            "A finite-grid simulation would impose a maximum effective "
            "multipole on the cosmic microwave background power "
            "spectrum. Continuous physics predicts smooth roll-off "
            "governed by Silk damping only."
        ),
        detectability_threshold=2500.0,
        detectability_units="multipole ℓ_max",
        current_observation_status=ObservationStatus.NOT_OBSERVED,
        current_observation_value=None,
        reference=("Beane, Davoudi, Savage (2014); context for cosmological observables."),
    ),
)


def build_canonical_ladder() -> FalsificationLadder:
    """Return the immutable canonical ladder of six pre-registered signatures.

    Calling this function multiple times returns equal-by-value ladders
    that share the same underlying CANONICAL_SIGNATURES tuple.
    """
    return FalsificationLadder(signatures=CANONICAL_SIGNATURES)
