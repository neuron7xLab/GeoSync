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

from .adversarial_ladder import (
    LADDER_RUNGS,
    LadderConfig,
    LadderReport,
    ParameterFragilityReport,
    ProsecutorOutcome,
    ProsecutorScore,
    parameter_fragility_audit,
    run_adversarial_ladder,
    run_null_audit,
)
from .baselines import (
    edge_density_score,
    rolling_volatility_score,
)
from .coupling import (
    coupling_from_exposures,
    omega_from_volatility,
    sakaguchi_alpha_zero,
)
from .critical_slowing_down import (
    CSDConfig,
    CSDIndicators,
    compute_csd_indicators,
)
from .early_warning import (
    EarlyWarningConfig,
    EarlyWarningResult,
    compute_early_warning,
    kuramoto_order_parameter,
)
from .errors import (
    InvalidExposureMatrixError,
    InvalidNodeLabelsError,
    InvalidTemporalPanelError,
    SystemicRiskInputError,
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
    run_end_to_end_falsification,
    run_falsification,
    run_score_level_falsification,
)
from .governance import (
    FORBIDDEN_OVERCLAIM_TERMS,
    PremergeGateReport,
    ValidationReadinessReport,
    assert_claim_tier,
    build_validation_readiness_report,
    run_premerge_science_gate,
)
from .metrics import (
    ClassificationMetrics,
    LeadTimeConfig,
    LeadTimeMetrics,
    compute_classification_metrics,
    compute_lead_time_metrics,
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
    fit_barabasi_albert_validation_from_topology,
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
from .temporal_panel import (
    validate_temporal_exposure_panel,
)
from .topology import (
    InterbankTopology,
    barabasi_albert_null,
    from_exposure_matrix,
)

__all__ = [
    "BankingCrisisEvent",
    "BankingCrisisLedger",
    "CSDConfig",
    "CSDIndicators",
    "ClassificationMetrics",
    "CrisisOutcome",
    "DEFAULT_LEDGER",
    "EarlyWarningConfig",
    "EarlyWarningResult",
    "ExponentialFit",
    "FORBIDDEN_OVERCLAIM_TERMS",
    "FalsificationConfig",
    "FalsificationReport",
    "INTERBANK_DEFAULT_BAND",
    "InterbankTopology",
    "InvalidExposureMatrixError",
    "InvalidNodeLabelsError",
    "InvalidTemporalPanelError",
    "LADDER_RUNGS",
    "LadderConfig",
    "LadderReport",
    "LeadTimeConfig",
    "LeadTimeMetrics",
    "MIN_RELATIVE_SE_VALIDATION",
    "MIN_TAIL_SIZE_VALIDATION",
    "ModelComparison",
    "NullSurrogate",
    "ParameterFragilityReport",
    "PowerLawFit",
    "PremergeGateReport",
    "ProsecutorOutcome",
    "ProsecutorScore",
    "RunManifest",
    "SystemicRiskInputError",
    "ValidationReadinessReport",
    "assert_claim_tier",
    "auc_bootstrap_ci",
    "auc_mann_whitney",
    "barabasi_albert_null",
    "bonferroni_correction",
    "build_run_manifest",
    "build_validation_readiness_report",
    "compare_power_law_vs_exponential",
    "compute_classification_metrics",
    "compute_csd_indicators",
    "compute_early_warning",
    "compute_lead_time_metrics",
    "coupling_from_exposures",
    "degree_preserving_randomization",
    "edge_density_score",
    "fit_barabasi_albert",
    "fit_barabasi_albert_from_topology",
    "fit_barabasi_albert_validation_from_topology",
    "fit_exponential",
    "fit_power_law",
    "fit_power_law_validation",
    "from_exposure_matrix",
    "interbank_phase_extract",
    "kuramoto_order_parameter",
    "linear_correlation_surrogate",
    "omega_from_volatility",
    "parameter_fragility_audit",
    "permuted_crisis_dates",
    "random_exposure_weights",
    "rolling_volatility_score",
    "run_adversarial_ladder",
    "run_end_to_end_falsification",
    "run_falsification",
    "run_null_audit",
    "run_premerge_science_gate",
    "run_score_level_falsification",
    "sakaguchi_alpha_zero",
    "shuffled_time_labels",
    "static_topology_baseline",
    "validate_temporal_exposure_panel",
]
