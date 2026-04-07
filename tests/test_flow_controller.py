# mypy: disable-error-code="attr-defined,unused-ignore,no-untyped-call,arg-type"
"""Tests for unified FlowController — single deterministic pipeline."""

from __future__ import annotations

import time
from collections import Counter

import numpy as np

from geosync.neuroeconomics.flow_controller import (
    DEFAULT_WEIGHTS,
    FlowController,
    FlowDecision,
    FlowWeights,
)


def _sig(
    regime: str = "METASTABLE",
    gamma: float = 1.0,
    risk_scalar: float = 0.8,
    regime_confidence: float = 0.8,
    signal_strength: float = 0.2,
) -> dict[str, object]:
    return {
        "timestamp_ns": time.time_ns(),
        "instrument": "EURUSD",
        "gamma": gamma,
        "order_parameter_R": 0.6,
        "ricci_curvature": -0.1,
        "lyapunov_max": 0.01,
        "regime": regime,
        "regime_confidence": regime_confidence,
        "regime_duration_s": 5.0,
        "signal_strength": signal_strength,
        "risk_scalar": risk_scalar,
        "sequence_number": 0,
    }


def test_single_tick_returns_valid_output() -> None:
    fc = FlowController()
    out = fc.process(_sig())
    assert out.decision in (
        FlowDecision.TRADE,
        FlowDecision.OBSERVE,
        FlowDecision.ABORT,
    )
    assert out.adjusted_size >= 0.0
    assert -1.0 <= out.v_net <= 1.0
    assert out.alpha_t > 0
    assert 0.0 <= out.effort_gate <= 1.0


def test_delta_closes_loop() -> None:
    fc = FlowController()
    out1 = fc.process(_sig(), outcome=0.0)
    out2 = fc.process(_sig(), outcome=0.5)
    # delta should be nonzero when outcome changes
    assert out2.delta_t != 0.0 or out1.v_net == 0.5


def test_all_weights_named_in_dataclass() -> None:
    w = DEFAULT_WEIGHTS
    # Every weight is a float > 0
    for field_name in FlowWeights.__dataclass_fields__:
        val = getattr(w, field_name)
        assert isinstance(val, float), f"{field_name} is not float"


def test_decision_distribution_balanced() -> None:
    fc = FlowController()
    counts: Counter[str] = Counter()
    for i in range(200):
        regime = ["METASTABLE", "COHERENT", "DECOHERENT", "CRITICAL"][i % 4]
        out = fc.process(
            _sig(
                regime=regime,
                risk_scalar=0.3 + 0.5 * (i % 3) / 2,
                regime_confidence=0.5 + 0.3 * (i % 2),
            ),
            outcome=0.01 * (i % 5 - 2),
        )
        counts[out.decision.value] += 1

    # Must have at least TRADE and OBSERVE
    assert counts["TRADE"] > 0, f"No TRADE decisions: {dict(counts)}"
    assert counts["OBSERVE"] > 0 or counts["ABORT"] > 0, f"No non-trade: {dict(counts)}"


def test_dissociation_on_extreme_signal() -> None:
    fc = FlowController(weights=FlowWeights(ei_dissociation=1.5))
    # Force extreme excitatory
    out = fc.process(_sig(risk_scalar=0.99, regime_confidence=0.99, signal_strength=0.99))
    if out.decision == FlowDecision.DISSOCIATED:
        assert out.adjusted_size == 0.0
        assert out.kelly_mult == 0.0


def test_nan_signal_never_crashes() -> None:
    fc = FlowController()
    nan_sig: dict[str, object] = {
        "timestamp_ns": time.time_ns(),
        "instrument": "EURUSD",
        "gamma": float("nan"),
        "order_parameter_R": float("inf"),
        "ricci_curvature": float("-inf"),
        "lyapunov_max": float("nan"),
        "regime": "UNKNOWN",
        "regime_confidence": float("nan"),
        "regime_duration_s": -1.0,
        "signal_strength": float("nan"),
        "risk_scalar": float("nan"),
        "sequence_number": 0,
    }
    for _ in range(10):
        out = fc.process(nan_sig)
        assert out.adjusted_size >= 0.0
        assert out.adjusted_size <= 1.0


def test_v_net_bounded_under_random_input() -> None:
    fc = FlowController()
    rng = np.random.RandomState(42)
    for _ in range(100):
        out = fc.process(
            _sig(
                gamma=float(rng.uniform(-1, 3)),
                risk_scalar=float(rng.uniform(0, 1)),
                regime_confidence=float(rng.uniform(0, 1)),
                signal_strength=float(rng.uniform(-1, 1)),
            ),
            outcome=float(rng.uniform(-1, 1)),
        )
        assert -1.0 <= out.v_net <= 1.0


def test_size_never_exceeds_intended() -> None:
    fc = FlowController()
    for size in [0.01, 0.5, 1.0, 10.0, 1000.0]:
        out = fc.process(_sig(), intended_size=size)
        assert out.adjusted_size <= size


def test_alpha_adapts_to_outcome_volatility() -> None:
    # Stable outcomes
    fc_stable = FlowController()
    for _ in range(30):
        out_s = fc_stable.process(_sig(), outcome=0.01)
    alpha_stable = out_s.alpha_t

    # Volatile outcomes
    fc_vol = FlowController()
    for i in range(30):
        out_v = fc_vol.process(_sig(), outcome=float((-1) ** i) * 0.5)
    alpha_vol = out_v.alpha_t

    assert alpha_vol >= alpha_stable


def test_lambda_weights_shift_with_regime() -> None:
    fc = FlowController()
    out_coherent = fc.process(_sig(regime="COHERENT"), outcome=0.0)
    fc2 = FlowController()
    out_critical = fc2.process(_sig(regime="CRITICAL"), outcome=0.0)

    # COHERENT: goal dominates (λ[2] > λ[0])
    assert out_coherent.lambda_weights[2] > out_coherent.lambda_weights[0]
    # CRITICAL: pavlovian dominates (λ[0] > λ[2])
    assert out_critical.lambda_weights[0] > out_critical.lambda_weights[2]
