"""Uncertainty quantification for regime signal confidence.

Separates expected uncertainty (known unknowns) from unexpected
uncertainty (unknown unknowns) using prediction error decomposition.

Theory (Active Inference / Karl Friston):
  Total uncertainty = aleatoric (irreducible) + epistemic (model ignorance)

  Aleatoric: variance of signal within stable regime — cannot reduce
  Epistemic: disagreement between gamma, Kuramoto, Ricci — can reduce

  When epistemic > aleatoric: model doesn't know enough — reduce position
  When aleatoric > epistemic: noisy but confident — proceed

  Second-order ambiguity: |dγ/dt| / γ — uncertainty about uncertainty itself
  When ambiguity > risk: system in Ellsberg zone — no trading.

Trading application:
  - High epistemic → epistemic action (small probe, not full trade)
  - High aleatoric → wider stops, reduced leverage
  - Both low → maximum confidence → full position

Connects to: NeuroSignalBus (modulates serotonin threshold + kelly fraction)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


def _to_float(v: object) -> float:
    """Safe float conversion — NaN/Inf → 0.0 (fail-closed)."""
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        f = float(v)
        return f if math.isfinite(f) else 0.0
    return 0.0


@dataclass(frozen=True, slots=True)
class UncertaintyEstimate:
    """Decomposed uncertainty at one time step."""

    aleatoric: float  # irreducible noise [0, 1]
    epistemic: float  # model ignorance [0, 1]
    total: float  # combined [0, 1]
    surprise: float  # |observed - expected| normalized [0, 1]
    ambiguity_index: float  # second-order: σ₂/σ₁ — uncertainty about uncertainty
    is_novel: bool  # true if current state never seen in memory


class UncertaintyEstimator:
    """Online uncertainty decomposition from regime signal history.

    Two-level uncertainty from neuroeconomics:
      Level 1 (Risk):     known unknowns → 1 - regime_confidence
      Level 2 (Ambiguity): unknown unknowns → |dγ/dt| / γ

      Ambiguity Index A = σ₂ / (σ₁ + ε)
      A > 1.0 → system more uncertain about its uncertainty → epistemic zone

    Parameters
    ----------
    window_size
        Rolling history length for variance estimation.
    novelty_threshold
        Surprise level above which state is classified as novel.
    """

    def __init__(
        self,
        window_size: int = 100,
        novelty_threshold: float = 0.7,
    ) -> None:
        self.window_size = window_size
        self.novelty_threshold = novelty_threshold

        self._gamma_history: deque[float] = deque(maxlen=window_size)
        self._r_history: deque[float] = deque(maxlen=window_size)
        self._ricci_history: deque[float] = deque(maxlen=window_size)
        self._regime_history: deque[str] = deque(maxlen=window_size)
        self._risk_history: deque[float] = deque(maxlen=window_size)

    def update(self, signal: dict[str, object]) -> UncertaintyEstimate:
        """Ingest one signal and return decomposed uncertainty."""
        gamma = _to_float(signal.get("gamma"))
        r_val = _to_float(signal.get("order_parameter_R"))
        ricci = _to_float(signal.get("ricci_curvature"))
        regime = str(signal.get("regime") or "UNKNOWN")
        risk = _to_float(signal.get("risk_scalar"))

        self._gamma_history.append(gamma)
        self._r_history.append(r_val)
        self._ricci_history.append(ricci)
        self._regime_history.append(regime)
        self._risk_history.append(risk)

        if len(self._gamma_history) < 10:
            return UncertaintyEstimate(
                aleatoric=1.0,
                epistemic=1.0,
                total=1.0,
                surprise=1.0,
                ambiguity_index=2.0,
                is_novel=True,
            )

        # === Level 1: Risk (aleatoric) ===
        # Within-regime variance of risk_scalar
        current_regime = regime
        regime_risks = [
            r
            for r, reg in zip(self._risk_history, self._regime_history)
            if reg == current_regime
        ]
        if len(regime_risks) >= 3:
            aleatoric = min(1.0, _std(regime_risks) * 3.0)
        else:
            aleatoric = 0.5

        # === Level 1: Risk (epistemic) ===
        # Disagreement between signals — all normalized to [0, 1]
        # High disagreement = high epistemic uncertainty
        gamma_vote = max(0.0, min(1.0, 1.0 - abs(gamma - 1.0)))
        r_vote = max(0.0, min(1.0, r_val))
        ricci_vote = max(0.0, min(1.0, 0.5 + 0.5 * math.tanh(ricci)))
        votes = [gamma_vote, r_vote, ricci_vote]
        # Scale: std of 3 uniform [0,1] vars ≈ 0.29 max. Use ×2 for [0,~0.6] range
        epistemic = min(1.0, _std(votes) * 2.0)

        # === Level 2: Ambiguity ===
        # |dγ/dt| — how fast gamma itself is changing
        sigma_1 = 1.0 - _to_float(signal.get("regime_confidence"))
        gamma_list = list(self._gamma_history)
        if len(gamma_list) >= 3:
            diffs = [
                abs(gamma_list[i] - gamma_list[i - 1])
                for i in range(1, len(gamma_list))
            ]
            mean_gamma = sum(gamma_list) / len(gamma_list)
            sigma_2 = _std(diffs) / (abs(mean_gamma) + 1e-6)
        else:
            sigma_2 = 1.0
        ambiguity_index = sigma_2 / (sigma_1 + 1e-6)

        # Surprise: deviation from rolling mean
        mean_risk = sum(self._risk_history) / len(self._risk_history)
        surprise = min(1.0, abs(risk - mean_risk) / (mean_risk + 0.01))

        # Novelty
        regime_counts: dict[str, int] = {}
        for r in self._regime_history:
            regime_counts[r] = regime_counts.get(r, 0) + 1
        regime_freq = regime_counts.get(current_regime, 0) / len(self._regime_history)
        is_novel = regime_freq < 0.05 or surprise > self.novelty_threshold

        total = min(1.0, max(aleatoric, epistemic))

        return UncertaintyEstimate(
            aleatoric=round(aleatoric, 4),
            epistemic=round(epistemic, 4),
            total=round(total, 4),
            surprise=round(surprise, 4),
            ambiguity_index=round(ambiguity_index, 4),
            is_novel=is_novel,
        )

    def kelly_discount(self, estimate: UncertaintyEstimate) -> float:
        """Discount factor for Kelly fraction based on uncertainty.

        Returns ∈ [0.1, 1.0]:
          1.0 = full confidence → full Kelly
          0.1 = maximum uncertainty → 10% Kelly
        """
        return max(0.1, 1.0 - 0.9 * estimate.total)

    def is_ambiguity_zone(self, estimate: UncertaintyEstimate) -> bool:
        """True if second-order uncertainty dominates first-order."""
        return estimate.ambiguity_index > 1.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(max(0.0, var))
