# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Regression guard for transition-entry dead-zone behavior."""

from core.indicators.kuramoto_ricci_composite import KuramotoRicciComposite, MarketPhase


def test_transition_entry_positive_when_conditions_are_valid() -> None:
    model = KuramotoRicciComposite(min_confidence=0.5)

    entry = model._entry(  # noqa: SLF001 - intentional regression check on core rule
        MarketPhase.TRANSITION,
        R=model.Rp + 0.2,
        kt=-0.3,
        conf=model.min_conf,
    )

    assert entry > 0.0


def test_transition_entry_fail_closed_when_confidence_is_low() -> None:
    model = KuramotoRicciComposite(min_confidence=0.5)

    entry = model._entry(  # noqa: SLF001 - intentional regression check on core rule
        MarketPhase.TRANSITION,
        R=model.Rp + 0.2,
        kt=-0.3,
        conf=model.min_conf - 0.01,
    )

    assert entry == 0.0
