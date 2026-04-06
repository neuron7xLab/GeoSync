# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Emergent dynamics and orchestration metrics for GeoSync.

The module translates qualitative orchestration concepts into quantitative signals
that can be consumed by strategy, risk, and neuro-orchestrator components.

Core concepts
-------------
- Emergence from local interactions:
  measured by Kuramoto synchrony and pairwise phase locking.
- Orchestration via constraints:
  measured by coupling strength, latency pressure and E/I balance.
- Self-organization outcomes:
  encoded as discrete network regimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

__all__ = [
    "EmergentDynamicsOrchestrator",
    "EmergentState",
    "NetworkRegime",
]


class NetworkRegime(str, Enum):
    """High-level regimes produced by local-to-global synchronization."""

    CHAOTIC = "CHAOTIC"
    EXPLORATORY = "EXPLORATORY"
    FOCUSED = "FOCUSED"
    OVERCLOCKED = "OVERCLOCKED"


@dataclass(frozen=True)
class EmergentState:
    """Snapshot of collective network behavior at one time step."""

    synchrony: float
    phase_locking: float
    coupling_strength: float
    latency_pressure: float
    ei_balance: float
    orchestration_index: float
    regime: NetworkRegime


class EmergentDynamicsOrchestrator:
    """Estimate emergent network state from local oscillator activity.

    Parameters
    ----------
    sync_weight:
        Weight of synchrony in orchestration index.
    lock_weight:
        Weight of phase locking in orchestration index.
    coupling_weight:
        Weight of coupling in orchestration index.
    latency_weight:
        Penalty coefficient for delay pressure.
    """

    def __init__(
        self,
        *,
        sync_weight: float = 0.40,
        lock_weight: float = 0.25,
        coupling_weight: float = 0.35,
        latency_weight: float = 0.20,
    ) -> None:
        total = sync_weight + lock_weight + coupling_weight
        if total <= 0:
            raise ValueError("sum of positive weights must be > 0")

        self.sync_weight = sync_weight / total
        self.lock_weight = lock_weight / total
        self.coupling_weight = coupling_weight / total
        self.latency_weight = max(0.0, latency_weight)

    def compute_state(
        self,
        phases: np.ndarray,
        excitatory_drive: float,
        inhibitory_drive: float,
        *,
        dt_ms: float,
        adjacency: np.ndarray | None = None,
    ) -> EmergentState:
        """Compute orchestration metrics from oscillator phases.

        Parameters
        ----------
        phases:
            1D radians array, one phase per local element.
        excitatory_drive:
            Aggregate excitatory activity (must be non-negative).
        inhibitory_drive:
            Aggregate inhibitory activity (must be non-negative).
        dt_ms:
            Effective interaction delay in milliseconds.
        adjacency:
            Optional NxN coupling matrix. If omitted, dense coupling is assumed.
        """
        phase_vec = self._validate_phase_vector(phases)
        exc = self._validate_non_negative(excitatory_drive, "excitatory_drive")
        inh = self._validate_non_negative(inhibitory_drive, "inhibitory_drive")

        synchrony = self._kuramoto_order_parameter(phase_vec)
        phase_locking = self._phase_locking_value(phase_vec)
        coupling_strength = self._coupling_strength(phase_vec, adjacency)
        latency_pressure = self._latency_pressure(dt_ms)
        ei_balance = self._ei_balance(exc, inh)

        orchestration_index = (
            self.sync_weight * synchrony
            + self.lock_weight * phase_locking
            + self.coupling_weight * coupling_strength
            - self.latency_weight * latency_pressure
        ) * ei_balance
        orchestration_index = float(np.clip(orchestration_index, 0.0, 1.0))

        regime = self._classify_regime(
            synchrony=synchrony,
            phase_locking=phase_locking,
            ei_balance=ei_balance,
            latency_pressure=latency_pressure,
            orchestration_index=orchestration_index,
        )

        return EmergentState(
            synchrony=synchrony,
            phase_locking=phase_locking,
            coupling_strength=coupling_strength,
            latency_pressure=latency_pressure,
            ei_balance=ei_balance,
            orchestration_index=orchestration_index,
            regime=regime,
        )

    def compute_frame(
        self,
        phases_frame: pd.DataFrame,
        excitatory: pd.Series,
        inhibitory: pd.Series,
        *,
        dt_ms: float,
    ) -> pd.DataFrame:
        """Compute state trajectory from tabular phase data.

        Returns a DataFrame with one row per index and regime as string label.
        """
        if not phases_frame.index.equals(excitatory.index) or not phases_frame.index.equals(
            inhibitory.index
        ):
            raise ValueError("indices must align between phases_frame, excitatory, inhibitory")

        rows: list[dict[str, float | str]] = []
        for ts in phases_frame.index:
            state = self.compute_state(
                phases=phases_frame.loc[ts].to_numpy(dtype=float),
                excitatory_drive=float(excitatory.loc[ts]),
                inhibitory_drive=float(inhibitory.loc[ts]),
                dt_ms=dt_ms,
            )
            rows.append(
                {
                    "synchrony": state.synchrony,
                    "phase_locking": state.phase_locking,
                    "coupling_strength": state.coupling_strength,
                    "latency_pressure": state.latency_pressure,
                    "ei_balance": state.ei_balance,
                    "orchestration_index": state.orchestration_index,
                    "regime": state.regime.value,
                }
            )

        return pd.DataFrame(rows, index=phases_frame.index)

    @staticmethod
    def _validate_phase_vector(phases: np.ndarray) -> np.ndarray:
        vector = np.asarray(phases, dtype=float)
        if vector.ndim != 1:
            raise ValueError("phases must be 1D")
        if vector.size < 2:
            raise ValueError("phases must contain at least 2 elements")
        if not np.all(np.isfinite(vector)):
            raise ValueError("phases must be finite")
        return vector

    @staticmethod
    def _validate_non_negative(value: float, name: str) -> float:
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"{name} must be finite")
        if numeric < 0.0:
            raise ValueError(f"{name} must be >= 0")
        return numeric

    @staticmethod
    def _kuramoto_order_parameter(phases: np.ndarray) -> float:
        return float(np.abs(np.mean(np.exp(1j * phases))))

    @staticmethod
    def _phase_locking_value(phases: np.ndarray) -> float:
        pairwise = phases[:, None] - phases[None, :]
        plv_matrix = np.abs(np.exp(1j * pairwise))
        upper = plv_matrix[np.triu_indices_from(plv_matrix, k=1)]
        return float(upper.mean())

    @staticmethod
    def _coupling_strength(phases: np.ndarray, adjacency: np.ndarray | None) -> float:
        phase_diff = np.abs(phases[:, None] - phases[None, :])
        phase_diff = np.minimum(phase_diff, 2 * np.pi - phase_diff)
        coherence = 1.0 - np.clip(phase_diff / np.pi, 0.0, 1.0)

        if adjacency is None:
            weights = np.ones_like(coherence)
        else:
            weights = np.asarray(adjacency, dtype=float)
            if weights.shape != coherence.shape:
                raise ValueError("adjacency must be NxN with N=len(phases)")
            if np.any(weights < 0):
                raise ValueError("adjacency weights must be non-negative")

        np.fill_diagonal(weights, 0.0)
        np.fill_diagonal(coherence, 0.0)

        denom = weights.sum()
        if denom <= 0.0:
            return 0.0
        return float(np.sum(weights * coherence) / denom)

    @staticmethod
    def _latency_pressure(dt_ms: float) -> float:
        if not np.isfinite(dt_ms) or dt_ms <= 0:
            raise ValueError("dt_ms must be > 0 and finite")
        # Saturating transform: <10ms small penalty, >100ms high pressure
        return float(1.0 - np.exp(-dt_ms / 50.0))

    @staticmethod
    def _ei_balance(excitatory_drive: float, inhibitory_drive: float) -> float:
        total = excitatory_drive + inhibitory_drive
        if total == 0:
            return 0.0

        ratio = excitatory_drive / total
        # Ideal around 0.6 (softly excitation-dominant but bounded).
        distance = abs(ratio - 0.6)
        balance = 1.0 - (distance / 0.6)
        return float(np.clip(balance, 0.0, 1.0))

    @staticmethod
    def _classify_regime(
        *,
        synchrony: float,
        phase_locking: float,
        ei_balance: float,
        latency_pressure: float,
        orchestration_index: float,
    ) -> NetworkRegime:
        if ei_balance < 0.35 or orchestration_index < 0.25:
            return NetworkRegime.CHAOTIC

        if latency_pressure > 0.80 and synchrony > 0.75:
            return NetworkRegime.OVERCLOCKED

        if synchrony > 0.72 and phase_locking > 0.72 and orchestration_index > 0.6:
            return NetworkRegime.FOCUSED

        return NetworkRegime.EXPLORATORY
