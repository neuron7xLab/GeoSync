"""Tests for CoherenceRiskGate middleware."""

from __future__ import annotations

from coherence_bridge.mock_engine import MockEngine
from coherence_bridge.risk_gate import CoherenceRiskGate


def test_unknown_instrument_blocked_fail_closed() -> None:
    gate = CoherenceRiskGate(MockEngine(), fail_closed=True)
    decision = gate.apply("NONEXISTENT", 1.0)
    assert not decision.allowed
    assert decision.adjusted_size == 0.0
    assert "fail-closed" in decision.reason


def test_adjusted_size_never_exceeds_intended() -> None:
    """Invariant: risk gate NEVER amplifies position size."""
    engine = MockEngine()
    gate = CoherenceRiskGate(engine)
    for _ in range(100):
        for inst in engine.instruments:
            d = gate.apply(inst, 1.0)
            assert (
                d.adjusted_size <= 1.0
            ), f"Gate amplified size: {d.adjusted_size} > 1.0 for {inst} regime={d.regime}"


def test_critical_regime_always_blocked() -> None:
    """CRITICAL = herding/crash precursor → always block."""
    # Force a signal check — we can't control mock regime,
    # but we can verify the logic via direct signal injection
    from unittest.mock import MagicMock

    mock_engine = MagicMock()
    mock_engine.get_signal.return_value = {
        "regime": "CRITICAL",
        "risk_scalar": 0.9,
        "order_parameter_R": 0.95,
    }
    gate_crit = CoherenceRiskGate(mock_engine)
    d = gate_crit.apply("EURUSD", 1.0)
    assert not d.allowed
    assert d.regime == "CRITICAL"


def test_decoherent_regime_blocked() -> None:
    from unittest.mock import MagicMock

    mock_engine = MagicMock()
    mock_engine.get_signal.return_value = {
        "regime": "DECOHERENT",
        "risk_scalar": 0.1,
    }
    gate = CoherenceRiskGate(mock_engine)
    d = gate.apply("EURUSD", 1.0)
    assert not d.allowed
    assert "DECOHERENT" in d.reason


def test_metastable_high_risk_passes() -> None:
    from unittest.mock import MagicMock

    mock_engine = MagicMock()
    mock_engine.get_signal.return_value = {
        "regime": "METASTABLE",
        "risk_scalar": 0.95,
    }
    gate = CoherenceRiskGate(mock_engine)
    d = gate.apply("EURUSD", 1.0)
    assert d.allowed
    assert d.adjusted_size > 0
    assert d.adjusted_size <= 1.0


def test_coherent_applies_size_reduction() -> None:
    from unittest.mock import MagicMock

    mock_engine = MagicMock()
    mock_engine.get_signal.return_value = {
        "regime": "COHERENT",
        "risk_scalar": 0.8,
    }
    gate = CoherenceRiskGate(mock_engine, coherent_size_factor=0.6)
    d = gate.apply("EURUSD", 1.0)
    assert d.allowed
    # 1.0 * 0.8 * 0.6 = 0.48
    assert abs(d.adjusted_size - 0.48) < 0.01


def test_gate_decision_fields() -> None:
    gate = CoherenceRiskGate(MockEngine())
    d = gate.apply("EURUSD", 1.0)
    # All fields present
    assert isinstance(d.allowed, bool)
    assert isinstance(d.adjusted_size, float)
    assert isinstance(d.reason, str)
    assert isinstance(d.regime, str)
    assert isinstance(d.risk_scalar, float)
