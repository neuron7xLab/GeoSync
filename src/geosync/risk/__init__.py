# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Deprecated shim — the canonical risk package is :mod:`geosync.risk`.

``automated_testing`` and ``risk_core`` were moved out of this ``src.geosync.risk``
shadow into the canonical top-level ``geosync.risk`` package as part of the
single-canonical-package import-architecture ratchet. This module now re-exports
the canonical symbols so any legacy ``from src.geosync.risk import ...`` caller
keeps working, while introducing no ``src.*`` imports of its own.

Prefer importing from ``geosync.risk`` directly.
"""

from geosync.risk import (
    AutomatedRiskTester,
    KillSwitch,
    LimitViolation,
    MonteCarloConfig,
    OrderRateExceeded,
    RiskConfig,
    RiskError,
    RiskLimits,
    RiskManager,
    RiskScenario,
    ScenarioType,
    StressTestResult,
    check_risk_breach,
    compute_final_size,
    generate_flash_crash_scenarios,
    generate_liquidity_crisis_scenarios,
    generate_market_stress_scenarios,
    kelly_shrink,
    portfolio_heat,
    validate_risk_metrics,
    var_es,
)

__all__ = [
    "var_es",
    "kelly_shrink",
    "compute_final_size",
    "check_risk_breach",
    "RiskConfig",
    "RiskManager",
    "RiskLimits",
    "RiskError",
    "LimitViolation",
    "OrderRateExceeded",
    "KillSwitch",
    "portfolio_heat",
    "AutomatedRiskTester",
    "RiskScenario",
    "ScenarioType",
    "StressTestResult",
    "MonteCarloConfig",
    "generate_market_stress_scenarios",
    "generate_liquidity_crisis_scenarios",
    "generate_flash_crash_scenarios",
    "validate_risk_metrics",
]
