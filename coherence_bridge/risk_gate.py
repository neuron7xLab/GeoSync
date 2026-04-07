"""Risk gate middleware for OTP order router integration.

Sits between strategy signal and order submission. Applies GeoSync
regime classification as a position-sizing gate.

Logic:
  METASTABLE + risk_scalar > 0.7  → pass, full size
  COHERENT   + risk_scalar > 0.5  → pass, reduced size (×0.6)
  DECOHERENT                      → block, log reason
  CRITICAL                        → block, alert
  signal_unavailable              → block (fail-closed)

Invariants (enforced, never relaxed):
  - fail_closed=True: no signal = no trade
  - adjusted_size <= intended_size ALWAYS (risk gate never amplifies)
  - risk_scalar from gamma distance (derived, never assigned)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coherence_bridge.engine_interface import SignalEngine

logger = logging.getLogger("coherence_bridge.risk_gate")


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Result of risk gate evaluation."""

    allowed: bool
    adjusted_size: float
    reason: str
    regime: str
    risk_scalar: float


class CoherenceRiskGate:
    """Middleware between OTP strategy and order submission.

    Parameters
    ----------
    engine
        SignalEngine providing regime signals.
    fail_closed
        If True (default), missing signal → block order.
    metastable_threshold
        Minimum risk_scalar to pass in METASTABLE regime.
    coherent_threshold
        Minimum risk_scalar to pass in COHERENT regime.
    coherent_size_factor
        Size multiplier applied in COHERENT regime (reduced vs full).
    """

    def __init__(
        self,
        engine: SignalEngine,
        *,
        fail_closed: bool = True,
        metastable_threshold: float = 0.7,
        coherent_threshold: float = 0.5,
        coherent_size_factor: float = 0.6,
    ) -> None:
        self.engine = engine
        self.fail_closed = fail_closed
        self.metastable_threshold = metastable_threshold
        self.coherent_threshold = coherent_threshold
        self.coherent_size_factor = coherent_size_factor

    def apply(self, instrument: str, intended_size: float) -> GateDecision:
        """Evaluate whether an order should proceed and at what size.

        Returns GateDecision with:
          allowed: bool — True if order can proceed
          adjusted_size: float — always <= intended_size
          reason: str — human-readable explanation
          regime: str — current regime
          risk_scalar: float — current risk_scalar
        """
        sig = self.engine.get_signal(instrument)

        # Fail-closed: no signal = no trade
        if sig is None:
            if self.fail_closed:
                return GateDecision(
                    allowed=False,
                    adjusted_size=0.0,
                    reason=f"No signal available for {instrument} (fail-closed)",
                    regime="UNAVAILABLE",
                    risk_scalar=0.0,
                )
            return GateDecision(
                allowed=True,
                adjusted_size=intended_size,
                reason="No signal, fail-open mode",
                regime="UNAVAILABLE",
                risk_scalar=1.0,
            )

        regime = str(sig.get("regime", "UNKNOWN"))
        risk_scalar = float(sig.get("risk_scalar", 0.0) or 0.0)  # type: ignore[arg-type]

        # CRITICAL → block, alert
        if regime == "CRITICAL":
            logger.warning(
                "RISK GATE BLOCK: %s regime=CRITICAL risk=%.3f",
                instrument,
                risk_scalar,
            )
            return GateDecision(
                allowed=False,
                adjusted_size=0.0,
                reason=(
                    f"CRITICAL regime: herding detected (R={sig.get('order_parameter_R', 0):.3f})"
                ),
                regime=regime,
                risk_scalar=risk_scalar,
            )

        # DECOHERENT → block
        if regime == "DECOHERENT":
            return GateDecision(
                allowed=False,
                adjusted_size=0.0,
                reason="DECOHERENT regime: no signal edge (low R, high entropy)",
                regime=regime,
                risk_scalar=risk_scalar,
            )

        # UNKNOWN → block
        if regime == "UNKNOWN":
            return GateDecision(
                allowed=False,
                adjusted_size=0.0,
                reason="UNKNOWN regime: gamma non-finite or classification failed",
                regime=regime,
                risk_scalar=risk_scalar,
            )

        # METASTABLE → pass if risk_scalar above threshold
        if regime == "METASTABLE":
            if risk_scalar >= self.metastable_threshold:
                size = min(intended_size, intended_size * risk_scalar)
                return GateDecision(
                    allowed=True,
                    adjusted_size=size,
                    reason=f"METASTABLE: gamma near 1.0, risk={risk_scalar:.3f}",
                    regime=regime,
                    risk_scalar=risk_scalar,
                )
            return GateDecision(
                allowed=False,
                adjusted_size=0.0,
                reason=(
                    f"METASTABLE but risk_scalar={risk_scalar:.3f} < {self.metastable_threshold}"
                ),
                regime=regime,
                risk_scalar=risk_scalar,
            )

        # COHERENT → pass with reduced size if above threshold
        if regime == "COHERENT":
            if risk_scalar >= self.coherent_threshold:
                size = min(
                    intended_size,
                    intended_size * risk_scalar * self.coherent_size_factor,
                )
                return GateDecision(
                    allowed=True,
                    adjusted_size=size,
                    reason=f"COHERENT: high sync, reduced size ×{self.coherent_size_factor}",
                    regime=regime,
                    risk_scalar=risk_scalar,
                )
            return GateDecision(
                allowed=False,
                adjusted_size=0.0,
                reason=f"COHERENT but risk_scalar={risk_scalar:.3f} < {self.coherent_threshold}",
                regime=regime,
                risk_scalar=risk_scalar,
            )

        # Fallback: unknown regime → block (defensive)
        return GateDecision(
            allowed=False,
            adjusted_size=0.0,
            reason=f"Unrecognized regime: {regime}",
            regime=regime,
            risk_scalar=risk_scalar,
        )
