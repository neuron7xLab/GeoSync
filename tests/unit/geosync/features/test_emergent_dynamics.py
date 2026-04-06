from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geosync.features.emergent_dynamics import (
    EmergentDynamicsOrchestrator,
    NetworkRegime,
)


def test_compute_state_high_synchrony_focused_regime() -> None:
    orchestrator = EmergentDynamicsOrchestrator()
    phases = np.array([0.0, 0.01, -0.01, 0.0])

    state = orchestrator.compute_state(
        phases=phases,
        excitatory_drive=0.65,
        inhibitory_drive=0.35,
        dt_ms=5.0,
    )

    assert state.synchrony > 0.95
    assert state.phase_locking >= 0.99
    assert state.coupling_strength > 0.98
    assert state.regime in (NetworkRegime.FOCUSED, NetworkRegime.EXPLORATORY)


def test_compute_state_detects_chaotic_for_unbalanced_ei() -> None:
    orchestrator = EmergentDynamicsOrchestrator()
    phases = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])

    state = orchestrator.compute_state(
        phases=phases,
        excitatory_drive=1.0,
        inhibitory_drive=0.0,
        dt_ms=20.0,
    )

    assert state.synchrony < 0.1
    assert state.ei_balance == 0.0
    assert state.regime == NetworkRegime.CHAOTIC


def test_compute_frame_returns_index_aligned_dataframe() -> None:
    orchestrator = EmergentDynamicsOrchestrator()
    idx = pd.date_range("2026-01-01", periods=3, freq="h")
    phase_frame = pd.DataFrame(
        {
            "n1": [0.0, 0.1, 0.2],
            "n2": [0.0, 0.11, 0.21],
            "n3": [0.0, 0.09, 0.19],
        },
        index=idx,
    )
    excitatory = pd.Series([0.6, 0.58, 0.61], index=idx)
    inhibitory = pd.Series([0.4, 0.42, 0.39], index=idx)

    out = orchestrator.compute_frame(
        phases_frame=phase_frame,
        excitatory=excitatory,
        inhibitory=inhibitory,
        dt_ms=8.0,
    )

    assert list(out.index) == list(idx)
    assert set(out.columns) == {
        "synchrony",
        "phase_locking",
        "coupling_strength",
        "latency_pressure",
        "ei_balance",
        "orchestration_index",
        "regime",
    }


def test_compute_state_rejects_invalid_inputs() -> None:
    orchestrator = EmergentDynamicsOrchestrator()

    with pytest.raises(ValueError, match="phases must be 1D"):
        orchestrator.compute_state(
            phases=np.array([[0.0, 0.1], [0.2, 0.3]]),
            excitatory_drive=0.6,
            inhibitory_drive=0.4,
            dt_ms=10.0,
        )

    with pytest.raises(ValueError, match="excitatory_drive must be >= 0"):
        orchestrator.compute_state(
            phases=np.array([0.0, 0.1]),
            excitatory_drive=-0.1,
            inhibitory_drive=0.4,
            dt_ms=10.0,
        )

    with pytest.raises(ValueError, match="dt_ms must be > 0"):
        orchestrator.compute_state(
            phases=np.array([0.0, 0.1]),
            excitatory_drive=0.6,
            inhibitory_drive=0.4,
            dt_ms=0.0,
        )
