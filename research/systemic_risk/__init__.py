# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Systemic-risk-as-phase-transition research module — v2.

Pre-registered falsification of the hypothesis that interbank
phase-locking precedes banking-crisis events. All claims under
``CLAIMS.md`` are ``HYPOTHESIS`` tier until the v2 battery returns
``HARD_PASS`` on >= 2 independent crises with real interbank
exposure data and a bootstrap-CI lower bound clearing 0.70.

Layer in the maintenance hierarchy: this module is a *Sustainer*
diagnostic — it reports approach to the Kuramoto bifurcation
without taking any execution action.
"""

from __future__ import annotations

from .coupling import (
    coupling_from_exposures,
    omega_from_volatility,
    sakaguchi_alpha_zero,
)
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
    auc_bootstrap_ci,
    auc_mann_whitney,
    bonferroni_correction,
    run_falsification,
)
from .network_fitting import (
    MIN_RELATIVE_SE_VALIDATION,
    MIN_TAIL_SIZE_VALIDATION,
    ExponentialFit,
    ModelComparison,
    PowerLawFit,
    compare_power_law_vs_exponential,
    fit_barabasi_albert,
    fit_barabasi_albert_from_topology,
    fit_exponential,
    fit_power_law,
    fit_power_law_validation,
)
from .null_models import (
    NullSurrogate,
    degree_preserving_randomization,
    linear_correlation_surrogate,
    permuted_crisis_dates,
    random_exposure_weights,
    shuffled_time_labels,
    static_topology_baseline,
)
from .phase_extraction import (
    INTERBANK_DEFAULT_BAND,
    interbank_phase_extract,
)
from .replication import (
    RunManifest,
    build_run_manifest,
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
    "ExponentialFit",
    "FalsificationConfig",
    "FalsificationReport",
    "INTERBANK_DEFAULT_BAND",
    "InterbankTopology",
    "MIN_RELATIVE_SE_VALIDATION",
    "MIN_TAIL_SIZE_VALIDATION",
    "ModelComparison",
    "NullSurrogate",
    "PowerLawFit",
    "RunManifest",
    "auc_bootstrap_ci",
    "auc_mann_whitney",
    "barabasi_albert_null",
    "bonferroni_correction",
    "build_run_manifest",
    "compare_power_law_vs_exponential",
    "compute_early_warning",
    "coupling_from_exposures",
    "degree_preserving_randomization",
    "fit_barabasi_albert",
    "fit_barabasi_albert_from_topology",
    "fit_exponential",
    "fit_power_law",
    "fit_power_law_validation",
    "from_exposure_matrix",
    "interbank_phase_extract",
    "kuramoto_order_parameter",
    "linear_correlation_surrogate",
    "omega_from_volatility",
    "permuted_crisis_dates",
    "random_exposure_weights",
    "run_falsification",
    "sakaguchi_alpha_zero",
    "shuffled_time_labels",
    "static_topology_baseline",
]
