# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Systemic-risk-as-phase-transition research module.

A pre-registered falsification of the hypothesis that interbank
phase-locking precedes banking-crisis events. All claims under
``CLAIMS.md`` are ``HYPOTHESIS`` tier until the battery returns
``HARD_PASS`` on >= 2 independent crises.

Layer in the maintenance hierarchy: this module is a *Sustainer*
diagnostic — it reports approach to the Kuramoto bifurcation Φ → 0
without taking any execution action.
"""

from __future__ import annotations

from .early_warning import (
    EarlyWarningConfig,
    EarlyWarningResult,
    compute_early_warning,
    kuramoto_order_parameter,
)
from .event_ledger import (
    DEFAULT_LEDGER,
    BankingCrisisEvent,
    BankingCrisisLedger,
)
from .falsification import (
    CrisisOutcome,
    FalsificationConfig,
    FalsificationReport,
    auc_mann_whitney,
    benjamini_hochberg,
    run_falsification,
)
from .phase_extraction import (
    INTERBANK_DEFAULT_BAND,
    interbank_phase_extract,
)
from .topology import (
    InterbankTopology,
    barabasi_albert_null,
    from_exposure_matrix,
)

__all__ = [
    "BankingCrisisEvent",
    "BankingCrisisLedger",
    "CrisisOutcome",
    "DEFAULT_LEDGER",
    "EarlyWarningConfig",
    "EarlyWarningResult",
    "FalsificationConfig",
    "FalsificationReport",
    "INTERBANK_DEFAULT_BAND",
    "InterbankTopology",
    "auc_mann_whitney",
    "barabasi_albert_null",
    "benjamini_hochberg",
    "compute_early_warning",
    "from_exposure_matrix",
    "interbank_phase_extract",
    "kuramoto_order_parameter",
    "run_falsification",
]
