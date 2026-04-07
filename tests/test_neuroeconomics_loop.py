# mypy: disable-error-code="attr-defined,unused-ignore,no-untyped-call,arg-type"
"""Integration test: closed-loop neuroeconomic decision cycle.

Proves: signal → uncertainty → control_value → context_memory
→ prior_integration → decision_currency → delta_t → back to uncertainty.
"""

from __future__ import annotations

import numpy as np

from geosync.neuroeconomics.context_memory import ContextMemory
from geosync.neuroeconomics.control_value import ControlValueGate
from geosync.neuroeconomics.decision_currency import DecisionCurrency
from geosync.neuroeconomics.prior_integration import PriorIntegrator
from geosync.neuroeconomics.uncertainty import UncertaintyController, UncertaintyType

# === Module unit tests ===


def test_uncertainty_adapts_alpha_to_volatility() -> None:
    uc = UncertaintyController(alpha_min=0.01, alpha_max=0.5)
    # Stable environment: small deltas
    for _ in range(20):
        state = uc.update(delta_t=0.01)
    alpha_stable = state.alpha

    # Volatile environment: large deltas
    uc2 = UncertaintyController(alpha_min=0.01, alpha_max=0.5)
    for _ in range(20):
        state2 = uc2.update(delta_t=float(np.random.choice([-1.0, 1.0])))
    alpha_volatile = state2.alpha

    assert (
        alpha_volatile > alpha_stable
    ), f"Volatile alpha={alpha_volatile} should > stable alpha={alpha_stable}"


def test_uncertainty_classifies_surprise() -> None:
    uc = UncertaintyController()
    for _ in range(20):
        uc.update(delta_t=0.01)
    # Now inject large surprise
    state = uc.update(delta_t=5.0)
    assert state.surprise > 2.0
    assert state.uncertainty_type == UncertaintyType.UNEXPECTED


def test_control_value_effort_gate() -> None:
    cv = ControlValueGate()
    # High VOI, low cost → high effort
    high = cv.compute(prior_entropy=2.0, expected_posterior_entropy=0.5, latency_ms=10)
    assert high.effort_gate > 0.5

    # Low VOI, high cost → low effort
    low = cv.compute(prior_entropy=0.1, expected_posterior_entropy=0.09, latency_ms=500)
    assert low.effort_gate < high.effort_gate


def test_context_memory_pessimism_bias() -> None:
    cm = ContextMemory(alpha_gain=0.05, alpha_loss=0.15)
    # Loss → alpha_loss
    state_loss = cm.update(regime="DECOHERENT", outcome=-0.5)
    assert state_loss.effective_alpha == 0.15

    # Gain → alpha_gain
    state_gain = cm.update(regime="COHERENT", outcome=0.5)
    assert state_gain.effective_alpha == 0.05


def test_context_memory_history_decay() -> None:
    cm = ContextMemory(decay=0.9)
    cm.update(regime="METASTABLE", outcome=1.0)
    s1 = cm.update(regime="METASTABLE", outcome=0.0)
    s2 = cm.update(regime="METASTABLE", outcome=0.0)
    # History should decay toward 0
    assert abs(s2.history_score) < abs(s1.history_score)


def test_prior_integration_bayesian_update() -> None:
    pi = PriorIntegrator(n_states=3)
    # Strong evidence for state 0
    state = pi.update(likelihood=[10.0, 0.1, 0.1])
    assert state.posterior[0] > 0.9
    assert state.suppression_weight < 0.5  # expected signal → low suppression


def test_prior_integration_surprise_amplification() -> None:
    pi = PriorIntegrator(n_states=3)
    # First: build strong prior for state 0
    for _ in range(5):
        pi.update(likelihood=[10.0, 0.1, 0.1])
    # Now: evidence contradicts → high suppression weight
    state = pi.update(likelihood=[0.01, 10.0, 0.01])
    assert state.suppression_weight > 0.5


def test_prior_entropy_decreases_with_evidence() -> None:
    pi = PriorIntegrator(n_states=5)
    s0 = pi.update(likelihood=[1.0, 1.0, 1.0, 1.0, 1.0])  # uniform → high entropy
    for _ in range(5):
        s = pi.update(likelihood=[10.0, 0.1, 0.1, 0.1, 0.1])
    assert s.prior_entropy < s0.prior_entropy


def test_decision_currency_regime_lambda_shift() -> None:
    dc = DecisionCurrency()
    # COHERENT: goal-directed dominates
    s1 = dc.update(
        goal_value=0.8,
        signal_strength=0.0,
        outcome=0.0,
        alpha=0.1,
        regime="COHERENT",
    )
    assert s1.lambda_weights[2] > s1.lambda_weights[1]  # goal > habit

    # CRITICAL: pavlovian dominates
    dc2 = DecisionCurrency()
    s2 = dc2.update(
        goal_value=0.8,
        signal_strength=-0.5,
        outcome=0.0,
        alpha=0.1,
        regime="CRITICAL",
    )
    assert s2.lambda_weights[0] > s2.lambda_weights[2]  # pav > goal


def test_decision_currency_delta_closes_loop() -> None:
    dc = DecisionCurrency()
    dc.update(
        goal_value=0.5,
        signal_strength=0.0,
        outcome=0.0,
        alpha=0.1,
        regime="METASTABLE",
    )
    # outcome differs from v_net_prev → delta nonzero
    s2 = dc.update(
        goal_value=0.5,
        signal_strength=0.0,
        outcome=0.3,
        alpha=0.1,
        regime="METASTABLE",
    )
    assert s2.delta != 0.0, "delta must be nonzero when outcome ≠ v_net_prev"


# === INTEGRATION: Full closed loop ===


def test_full_closed_loop_5_modules() -> None:
    """signal → uncertainty → control → context → prior → currency → delta → loop.

    This is THE test that proves the neuroeconomic cycle works end-to-end.
    """
    uc = UncertaintyController()
    cv = ControlValueGate()
    cm = ContextMemory()
    pi = PriorIntegrator(n_states=5)
    dc = DecisionCurrency()

    # Simulate 30 ticks
    delta_t = 0.0
    outcome = 0.0

    for tick in range(30):
        # 1. Uncertainty: receive delta from previous cycle
        unc = uc.update(delta_t=delta_t, outcome=outcome)

        # 2. Control value: is deliberation worth it?
        ctrl = cv.compute(
            prior_entropy=1.5,
            expected_posterior_entropy=1.0,
            latency_ms=20.0,
        )

        # 3. Context memory: update regime + experience
        regime = ["METASTABLE", "COHERENT", "DECOHERENT", "CRITICAL"][tick % 4]
        ctx = cm.update(regime=regime, outcome=outcome)

        # 4. Prior integration: Bayesian update
        lik = [0.5, 0.5, 0.5, 0.5, 0.5]
        lik[tick % 5] = 5.0  # strong evidence for one state
        prior_state = pi.update(likelihood=lik)

        # 5. Decision currency: compute V_net and delta
        goal_value = 0.5 * (1.0 + 0.3 * np.sin(tick * 0.5))
        signal_strength = 0.2 * np.cos(tick * 0.3)

        dec = dc.update(
            goal_value=goal_value,
            signal_strength=signal_strength,
            outcome=outcome,
            alpha=unc.alpha,
            regime=regime,
            policy_delta=ctx.policy_delta,
            drift_bias=prior_state.drift_bias,
            effort_gate=ctrl.effort_gate,
        )

        # Close the loop: delta feeds back to uncertainty on next tick
        delta_t = dec.delta
        outcome = dec.v_net * 0.1  # simulated execution

    # Verify the loop ran and produced valid state
    assert -1.0 <= dec.v_net <= 1.0
    assert unc.alpha > 0
    assert 0.0 <= ctrl.effort_gate <= 1.0
    assert isinstance(ctx.regime, str)
    assert sum(prior_state.posterior) > 0.99  # normalized


def test_loop_alpha_responds_to_regime_volatility() -> None:
    """Alpha should be higher after volatile regime switches than stable."""
    uc_stable = UncertaintyController()
    dc_stable = DecisionCurrency()
    delta = 0.0
    for _ in range(30):
        unc = uc_stable.update(delta_t=delta)
        dec = dc_stable.update(
            goal_value=0.5,
            signal_strength=0.0,
            outcome=0.0,
            alpha=unc.alpha,
            regime="METASTABLE",
        )
        delta = dec.delta
    alpha_stable = unc.alpha

    uc_volatile = UncertaintyController()
    dc_volatile = DecisionCurrency()
    delta = 0.0
    for i in range(30):
        unc = uc_volatile.update(delta_t=delta)
        regime = "CRITICAL" if i % 2 == 0 else "COHERENT"
        dec = dc_volatile.update(
            goal_value=0.5,
            signal_strength=0.0,
            outcome=float(i % 3) * 0.5,
            alpha=unc.alpha,
            regime=regime,
        )
        delta = dec.delta
    alpha_volatile = unc.alpha

    assert alpha_volatile >= alpha_stable


def test_v_net_bounded() -> None:
    """V_net ∈ [-1, 1] always, regardless of inputs."""
    dc = DecisionCurrency()
    for _ in range(100):
        dec = dc.update(
            goal_value=float(np.random.uniform(-5, 5)),
            signal_strength=float(np.random.uniform(-2, 2)),
            outcome=float(np.random.uniform(-3, 3)),
            alpha=float(np.random.uniform(0, 1)),
            regime=np.random.choice(["COHERENT", "METASTABLE", "DECOHERENT", "CRITICAL"]),
            policy_delta=float(np.random.uniform(-1, 1)),
            drift_bias=float(np.random.uniform(-1, 1)),
        )
        assert -1.0 <= dec.v_net <= 1.0, f"V_net={dec.v_net} out of [-1,1]"
