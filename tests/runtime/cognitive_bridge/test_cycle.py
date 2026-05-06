# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""15-stage cycle state-machine tests."""

from __future__ import annotations

import pytest

from runtime.cognitive_bridge.cycle import (
    Cycle,
    CycleContractError,
    Stage,
    is_legal,
)
from runtime.cognitive_bridge.knowledge_node import KnowledgeStatus
from runtime.cognitive_bridge.value_function import ValueComponents


def _components(*, signal: float = 0.9, noise: float = 0.05) -> ValueComponents:
    return ValueComponents(
        invariance=signal,
        falsifiability=signal,
        stability=signal,
        cross_domain=signal,
        actionability=signal,
        reproducibility=signal,
        productivity=signal,
        noise=noise,
        hallucination=noise,
        cognitive_cost=noise,
    )


_PROGRESSION: tuple[Stage, ...] = (
    Stage.CLAIM,
    Stage.CLEAN_CLAIM,
    Stage.INVARIANT,
    Stage.EROSION,
    Stage.FALSIFICATION_CONTRACT,
    Stage.STABILITY,
    Stage.CROSS_DOMAIN,
    Stage.MEANING_FACT,
    Stage.CRYSTALLIZED_FORM,
    Stage.EXECUTABLE_PROTOCOL,
    Stage.RESULT_ARTIFACT,
    Stage.VERIFICATION,
    Stage.AUDIT,
)


def _drive_to_audit(cycle: Cycle, *, components: ValueComponents) -> None:
    for stage in _PROGRESSION:
        if stage is Stage.STABILITY:
            cycle.set_falsification_contract(
                "if OFI/spread divergence > 3σ on out-of-sample, the claim is broken"
            )
        if stage is Stage.AUDIT:
            cycle.set_verification_evidence(
                "tests/research/test_ofi_unity.py shows IC=0.11 OOS over 12 windows"
            )
        cycle.advance(stage, summary=f"stage {stage.value}", components=components)


def test_legal_transitions_only_allow_canonical_path() -> None:
    assert is_legal(Stage.RAW_SIGNAL, Stage.CLAIM)
    assert not is_legal(Stage.RAW_SIGNAL, Stage.AUDIT)
    assert is_legal(Stage.CLAIM, Stage.ABORT)
    assert not is_legal(Stage.KNOWLEDGE_NODE, Stage.AUDIT)


def test_cycle_seed_summary_required() -> None:
    with pytest.raises(CycleContractError):
        Cycle(seed_summary="   ")


def test_cycle_id_is_deterministic_for_seed() -> None:
    cycle = Cycle(seed_summary="bid/ask spread vs Ricci on hourly EURUSD")
    assert len(cycle.cycle_id) == 64


def test_full_progression_emits_core_fact() -> None:
    cycle = Cycle(seed_summary="GeoSync OFI/Ricci alignment hypothesis")
    _drive_to_audit(cycle, components=_components(signal=1.0, noise=0.0))
    node = cycle.commit_to_memory(
        summary="OFI/Ricci alignment yields signed PLV in critical regime"
    )
    assert node.status is KnowledgeStatus.CORE_FACT
    assert node.value_score > 0.6
    assert node.cycle_id == cycle.cycle_id


def test_skipping_stage_raises_contract_error() -> None:
    cycle = Cycle(seed_summary="x")
    with pytest.raises(CycleContractError):
        cycle.advance(Stage.INVARIANT, summary="skip", components=_components())


def test_advancing_to_stability_without_falsifier_raises() -> None:
    cycle = Cycle(seed_summary="x")
    for stage in (
        Stage.CLAIM,
        Stage.CLEAN_CLAIM,
        Stage.INVARIANT,
        Stage.EROSION,
        Stage.FALSIFICATION_CONTRACT,
    ):
        cycle.advance(stage, summary=stage.value, components=_components())
    with pytest.raises(CycleContractError):
        cycle.advance(Stage.STABILITY, summary="stab", components=_components())


def test_commit_without_audit_raises() -> None:
    cycle = Cycle(seed_summary="x")
    cycle.advance(Stage.CLAIM, summary="c", components=_components())
    with pytest.raises(CycleContractError):
        cycle.commit_to_memory(summary="premature")


def test_abort_freezes_cycle() -> None:
    cycle = Cycle(seed_summary="x")
    cycle.advance(Stage.CLAIM, summary="c", components=_components())
    cycle.abort("contradiction with INV-K2")
    assert cycle.stage is Stage.ABORT
    assert cycle.aborted_reason == "contradiction with INV-K2"
    with pytest.raises(CycleContractError):
        cycle.advance(Stage.CLEAN_CLAIM, summary="x", components=_components())


def test_noisy_artifact_falls_below_core_fact_threshold() -> None:
    cycle = Cycle(seed_summary="loose hypothesis")
    weak = ValueComponents(
        invariance=0.30,
        falsifiability=0.25,
        stability=0.20,
        cross_domain=0.10,
        actionability=0.30,
        reproducibility=0.15,
        productivity=0.20,
        noise=0.40,
        hallucination=0.25,
        cognitive_cost=0.30,
    )
    _drive_to_audit(cycle, components=weak)
    node = cycle.commit_to_memory(summary="weak signal")
    assert node.status in {
        KnowledgeStatus.CONTEXTUAL_TOOL,
        KnowledgeStatus.WORKING_HYPOTHESIS,
        KnowledgeStatus.ARCHIVED_ERROR,
        KnowledgeStatus.NOISE,
    }


def test_history_records_every_stage() -> None:
    cycle = Cycle(seed_summary="trace")
    _drive_to_audit(cycle, components=_components())
    stages = [record.stage for record in cycle.history]
    assert stages[0] is Stage.RAW_SIGNAL
    assert stages[-1] is Stage.AUDIT
    assert Stage.FALSIFICATION_CONTRACT in stages
