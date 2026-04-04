"""Tests for GABAPositionGate — inhibition-based position sizing.

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from core.neuro.signal_bus import NeuroSignalBus
from core.neuro.gaba_position_gate import GABAPositionGate


@pytest.fixture
def bus() -> NeuroSignalBus:
    return NeuroSignalBus()


@pytest.fixture
def gate(bus: NeuroSignalBus) -> GABAPositionGate:
    return GABAPositionGate(bus)


# ── Gating reduces position size ─────────────────────────────────────


@pytest.mark.L3
class TestGatePositionSize:
    """GABA inhibition multiplicatively reduces position size."""

    def test_zero_inhibition_passes_through(
        self, bus: NeuroSignalBus, gate: GABAPositionGate
    ) -> None:
        bus.publish_gaba(0.0)
        assert gate.gate_position_size(100.0) == pytest.approx(100.0)

    def test_full_inhibition_zeros_position(
        self, bus: NeuroSignalBus, gate: GABAPositionGate
    ) -> None:
        bus.publish_gaba(1.0)
        assert gate.gate_position_size(100.0) == pytest.approx(0.0)

    def test_partial_inhibition_reduces_proportionally(
        self, bus: NeuroSignalBus, gate: GABAPositionGate
    ) -> None:
        bus.publish_gaba(0.5)
        result = gate.gate_position_size(100.0)
        assert 0.0 < result < 100.0
        assert result == pytest.approx(50.0)

    def test_result_never_negative(
        self, bus: NeuroSignalBus, gate: GABAPositionGate
    ) -> None:
        bus.publish_gaba(1.0)
        assert gate.gate_position_size(100.0) >= 0.0


# ── Inhibition update from market state ──────────────────────────────


@pytest.mark.L3
class TestUpdateInhibition:
    """Market-driven inhibition via sigmoid activation."""

    def test_high_vix_increases_inhibition(
        self, bus: NeuroSignalBus, gate: GABAPositionGate
    ) -> None:
        low = gate.update_inhibition(vix=10.0, volatility=0.1, rpe=0.0)
        high = gate.update_inhibition(vix=60.0, volatility=0.1, rpe=0.0)
        assert high > low

    def test_negative_rpe_increases_inhibition(
        self, bus: NeuroSignalBus, gate: GABAPositionGate
    ) -> None:
        pos = gate.update_inhibition(vix=20.0, volatility=0.1, rpe=0.5)
        neg = gate.update_inhibition(vix=20.0, volatility=0.1, rpe=-0.5)
        assert neg > pos

    def test_inhibition_bounded_zero_one(
        self, bus: NeuroSignalBus, gate: GABAPositionGate
    ) -> None:
        for vix in [0.0, 15.0, 50.0, 100.0]:
            for vol in [0.0, 0.1, 0.5, 1.0]:
                for rpe in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                    inh = gate.update_inhibition(vix=vix, volatility=vol, rpe=rpe)
                    assert 0.0 <= inh <= 1.0, (
                        f"Inhibition {inh} out of [0,1] for "
                        f"vix={vix}, vol={vol}, rpe={rpe}"
                    )

    def test_publishes_to_bus(
        self, bus: NeuroSignalBus, gate: GABAPositionGate
    ) -> None:
        inh = gate.update_inhibition(vix=30.0, volatility=0.2, rpe=0.0)
        assert bus.snapshot().gaba_inhibition == pytest.approx(inh, abs=1e-6)


# ── STDP-like plasticity ─────────────────────────────────────────────


@pytest.mark.L3
class TestSTDPPlasticity:
    """Negative RPE increases inhibition sensitivity (learn from pain)."""

    def test_repeated_negative_rpe_increases_w_rpe(
        self, bus: NeuroSignalBus
    ) -> None:
        gate = GABAPositionGate(bus, plasticity_rate=0.1)
        initial_w = gate.w_rpe

        # Repeatedly feed negative RPE
        for _ in range(10):
            gate.update_inhibition(vix=20.0, volatility=0.1, rpe=-0.8)

        assert gate.w_rpe > initial_w, "w_rpe should grow after negative RPE"

    def test_positive_rpe_does_not_change_w_rpe(
        self, bus: NeuroSignalBus
    ) -> None:
        gate = GABAPositionGate(bus, plasticity_rate=0.1)
        initial_w = gate.w_rpe

        for _ in range(10):
            gate.update_inhibition(vix=20.0, volatility=0.1, rpe=0.5)

        assert gate.w_rpe == pytest.approx(initial_w)

    def test_plasticity_increases_inhibition_over_time(
        self, bus: NeuroSignalBus
    ) -> None:
        gate = GABAPositionGate(bus, plasticity_rate=0.1)

        # First call: baseline sensitivity
        inh_first = gate.update_inhibition(vix=20.0, volatility=0.1, rpe=-0.5)
        # After repeated negative RPE, same inputs should yield higher inhibition
        for _ in range(20):
            gate.update_inhibition(vix=20.0, volatility=0.1, rpe=-0.5)
        inh_later = gate.update_inhibition(vix=20.0, volatility=0.1, rpe=-0.5)

        assert inh_later > inh_first
