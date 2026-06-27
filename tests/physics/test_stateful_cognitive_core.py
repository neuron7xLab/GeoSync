# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Stateful (model-based) verification of the cognitive core value function.

Property tests fuzz single inputs; this fuzzes SEQUENCES. A Hypothesis
``RuleBasedStateMachine`` drives the cognitive core through arbitrary
interleavings of in-manifold and out-of-manifold field assignments, maintaining
an independent model of which fields are valid. After every transition it asserts
the core's verdict equals the model's prediction:

    admissible  <=>  (at least one field is set)  AND  (no field is out-of-manifold)

and that the binding constraint, when NO_GO, names an actually-violated invariant.
This catches state-interaction bugs (ordering, accumulation, stale routing) that
no single-input test can reach. The model is the oracle; the core is the system
under test.
"""
from __future__ import annotations

from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    rule,
    run_state_machine_as_test,
)

from core.physics.cognitive_core import CognitiveCore, Decision, Status

# Per gated field: (a value strictly inside the manifold, a value strictly outside).
_VALID: dict[str, object] = {
    "order_parameter_R": 0.5,
    "z_abs": 0.4,
    "ollivier_kappa": 0.2,
    "gamma": 1.5,
    "serotonin_level": 0.3,
    "gaba_gate": 0.8,
    "dopamine_signal": 0.5,
    "kelly_fraction": 0.25,
    "free_energy_components": (1.0, 0.5, 0.2),
}
_INVALID: dict[str, object] = {
    "order_parameter_R": 1.5,
    "z_abs": 1.2,
    "ollivier_kappa": 2.0,
    "gamma": -0.1,
    "serotonin_level": 1.3,
    "gaba_gate": -0.1,
    "dopamine_signal": 1.5,
    "kelly_fraction": 1.5,
    "free_energy_components": (-1.0, 0.5, 0.2),
}
_FIELDS = sorted(_VALID)


class CognitiveCoreModel(RuleBasedStateMachine):
    """Model-based machine: an independent model predicts each admissibility verdict."""

    def __init__(self) -> None:
        super().__init__()
        self._core = CognitiveCore()
        self._snapshot: dict[str, object] = {}
        self._violated: set[str] = set()  # invariants currently out-of-manifold

    @rule(field=st.sampled_from(_FIELDS))
    def set_valid(self, field: str) -> None:
        self._snapshot[field] = _VALID[field]
        self._violated.discard(_INVARIANT_OF[field])

    @rule(field=st.sampled_from(_FIELDS))
    def set_invalid(self, field: str) -> None:
        self._snapshot[field] = _INVALID[field]
        self._violated.add(_INVARIANT_OF[field])

    @rule(field=st.sampled_from(_FIELDS))
    def clear(self, field: str) -> None:
        self._snapshot.pop(field, None)
        self._violated.discard(_INVARIANT_OF[field])

    @invariant()
    def verdict_matches_model(self) -> None:
        verdict = self._core.admissibility(self._snapshot)
        model_admissible = bool(self._snapshot) and not self._violated
        assert verdict.admissible == model_admissible, (
            f"STATEFUL-CORE VIOLATED: core.admissible={verdict.admissible} != "
            f"model={model_admissible}. snapshot_fields={sorted(self._snapshot)}, "
            f"model_violated={sorted(self._violated)}, status={verdict.status}, "
            f"violations={verdict.violations}."
        )
        if not model_admissible and self._snapshot:
            expected_status = Status.INADMISSIBLE if self._violated else Status.INSUFFICIENT
            assert verdict.status is expected_status and verdict.decision is Decision.NO_GO, (
                f"STATEFUL-CORE ROUTING VIOLATED: status={verdict.status} (expected "
                f"{expected_status}); model_violated={sorted(self._violated)}."
            )
            if self._violated:
                assert verdict.binding_constraint in self._violated, (
                    f"STATEFUL-CORE BINDING VIOLATED: binding_constraint="
                    f"{verdict.binding_constraint!r} not in model-violated set "
                    f"{sorted(self._violated)}."
                )


_INVARIANT_OF: dict[str, str] = {
    "order_parameter_R": "INV-K1",
    "z_abs": "INV-OA1",
    "ollivier_kappa": "INV-RC1",
    "gamma": "INV-DRO2",
    "serotonin_level": "INV-5HT2",
    "gaba_gate": "INV-GABA1",
    "dopamine_signal": "INV-DA8",
    "kelly_fraction": "INV-KELLY2",
    "free_energy_components": "INV-FE2",
}

def test_stateful_admissibility_holds_over_sequences() -> None:
    """Positive witness: over arbitrary mutation sequences the core matches the model.

    Drives the cognitive core through Hypothesis-generated interleavings of valid/
    invalid/clear field operations; after every transition the verdict must equal
    the independent model's prediction (admissible iff non-empty and unviolated).
    """
    run_state_machine_as_test(
        CognitiveCoreModel,
        settings=settings(max_examples=120, stateful_step_count=24, deadline=None),
    )


def test_stateful_model_oracle_is_discriminating() -> None:
    """Negative control: the model oracle distinguishes admissible from inadmissible states.

    Proves the stateful invariant is non-vacuous: a clean snapshot is admissible
    while the same snapshot with one out-of-manifold field is not — so the
    machine's invariant would FAIL if the core ever conflated the two.
    """
    core = CognitiveCore()
    clean = {"order_parameter_R": 0.5, "kelly_fraction": 0.25}
    dirty = {"order_parameter_R": 1.5, "kelly_fraction": 0.25}
    assert core.admissibility(clean).admissible is True
    assert core.admissibility(dirty).admissible is False, (
        "STATEFUL-CORE CONTROL BROKEN: an out-of-manifold field did not flip admissibility; "
        "the model oracle would be unable to discriminate states."
    )
