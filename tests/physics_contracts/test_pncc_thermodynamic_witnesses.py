# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Mathematical witnesses for the PNCC thermodynamic-control laws in the catalog.

Each binds a pytest function to a first-principles catalog law via ``@law`` and
proves it holds in the real ``core.physics`` PNCC modules. Two previously-
unwitnessed laws gain witnesses:

* ``pncc.reversible_gate_byte_exact`` — a reversible-tagged action's rollback
  recovers the byte-exact pre-state (INV-REVERSIBLE-GATE; Bennett 1973).
* ``pncc.landauer_proxy_dominance`` — the irreversible-action proxy cost
  dominates its reversible alternative, strictly when the irreversibility score
  is positive (INV-LANDAUER-PROXY; Landauer 1961).

Nothing here is a market claim. These are reversibility / thermodynamic-budget
facts about the code.
"""

from __future__ import annotations

import random

from core.physics.reversible_gate import ReversibleGate, ReversibleGateConfig
from core.physics.thermodynamic_budget import (
    ThermodynamicBudgetConfig,
    aggregate_entry,
    compute_entropy_cost,
    compute_irreversibility_cost,
    compute_latency_cost,
    compute_token_cost,
    reversible_alternative_cost,
)
from physics_contracts.law import law

# Byte-exact recovery is exact; landauer dominance is a strict physical bound.
COST_EPS = 1e-12


@law("pncc.reversible_gate_byte_exact", n_actions=200, seed=42)
def test_reversible_action_rolls_back_byte_exact() -> None:
    """INV-REVERSIBLE-GATE: rollback of a reversible action recovers the pre-state.

    For any action whose irreversibility score is at or below the gate's
    threshold, ``gate.rollback(trace.audit_hash).state_bytes`` must equal the
    original ``pre_state`` byte-for-byte. The witness drives many random
    reversible actions and requires every recovery to be exact.
    """
    n_actions = 200
    cfg = ReversibleGateConfig()
    gate = ReversibleGate(cfg)
    rng = random.Random(42)
    mismatches = 0
    for index in range(n_actions):
        pre_state = bytes(rng.randint(0, 255) for _ in range(rng.randint(0, 48)))
        post_state = bytes(rng.randint(0, 255) for _ in range(rng.randint(0, 48)))
        payload = bytes(rng.randint(0, 255) for _ in range(rng.randint(0, 32)))
        score = rng.uniform(0.0, cfg.irreversibility_threshold)
        trace = gate.gate(
            action_id=f"act-{index}",
            pre_state=pre_state,
            action_payload=payload,
            post_state=post_state,
            irreversibility_score=score,
            timestamp_ns=index,
            rollback_payload=pre_state,
        )
        if gate.rollback(trace.audit_hash).state_bytes != pre_state:
            mismatches += 1
    assert mismatches == 0, (
        "INV-REVERSIBLE-GATE violated: observed byte mismatches = "
        f"{mismatches} of n_actions={n_actions} with seed=42 must be 0; a reversible "
        "action's rollback must recover the pre-state byte-for-byte"
    )


@law("pncc.landauer_proxy_dominance", n_actions=100, seed=0)
def test_irreversible_cost_dominates_reversible_alternative() -> None:
    """INV-LANDAUER-PROXY: C_irr ≥ C_rev, strictly when irreversibility > 0.

    For every aggregated budget entry, the irreversible total proxy cost must be
    at least the cost of its reversible alternative, and strictly greater whenever
    the irreversibility score is positive — the Landauer-cost proxy of erasing
    information. The witness sweeps random token/latency/entropy/irreversibility
    entries and tracks the worst margin.
    """
    n_actions = 100
    cfg = ThermodynamicBudgetConfig()
    rng = random.Random(0)
    worst_margin = float("inf")
    strict_violations = 0
    for index in range(n_actions):
        token = compute_token_cost(rng.randint(1, 100), rng.randint(1, 100))
        latency = compute_latency_cost(rng.randint(1, 10**6), rng.randint(1, 10**6))
        entropy = compute_entropy_cost(rng.uniform(0.0, 10.0), rng.uniform(0.0, 10.0))
        score = rng.uniform(0.0, 1.0)
        irreversible = compute_irreversibility_cost(True, score, cfg)
        entry = aggregate_entry(f"act-{index}", index, token, latency, entropy, irreversible)
        c_irr = entry.total_proxy_cost
        c_rev = reversible_alternative_cost(entry, cfg)
        worst_margin = min(worst_margin, c_irr - c_rev)
        if score > 0.0 and not (c_irr > c_rev):
            strict_violations += 1
    assert worst_margin >= -COST_EPS, (
        "INV-LANDAUER-PROXY violated: observed worst C_irr − C_rev = "
        f"{worst_margin:.6f} must be ≥ −{COST_EPS:.0e} over n_actions={n_actions} with "
        "seed=0; an irreversible action's proxy cost must dominate its reversible alternative"
    )
    assert strict_violations == 0, (
        "INV-LANDAUER-PROXY violated: observed strict_violations = "
        f"{strict_violations} of n_actions={n_actions} with seed=0 must be 0; C_irr must "
        "be strictly greater than C_rev whenever the irreversibility score is positive"
    )
