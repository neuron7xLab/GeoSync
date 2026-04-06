# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Physics-inspired market modeling framework.

This module provides fundamental physical laws and constants for physics-inspired
algorithmic trading. By grounding market models in physical principles, we aim to:

1. Reduce noise in predictions through deterministic constraints
2. Improve stability via conservation laws
3. Enhance interpretability through physical analogies
4. Provide falsifiable hypotheses for market behavior

The seven fundamental laws integrated are:
- Newton's Laws of Motion (momentum, force, inertia)
- Universal Gravitation (market entity attraction)
- Conservation Laws (energy, momentum)
- Thermodynamics (entropy, free energy, equilibrium)
- Maxwell's Equations (field theory, wave propagation)
- Relativity (reference frames, time dilation)
- Heisenberg's Uncertainty Principle (position-momentum tradeoff)
"""

from .conservation import (
    check_energy_conservation,
    check_momentum_conservation,
    compute_market_energy,
    compute_market_momentum,
)
from .constants import PhysicsConstants

# Physics Engine v2 — T1-T7 modules
from .coulomb import CoulombInteraction

# Research modules — T1-T7 v2
from .diffusion_predictor import BacktestResult as DiffusionBacktestResult
from .diffusion_predictor import DiffusionVolatilityPredictor, VolatilityFrontPrediction
from .engine import GeoSyncPhysicsEngine, PhysicsEngineResult
from .explosive_sync import ESCircuitBreaker, ESProximityResult, ExplosiveSyncDetector
from .forman_ricci import DualTrackRicciMonitor, FormanRicciCurvature, FormanRicciResult
from .free_energy_trading_gate import FreeEnergyTradeDecision, FreeEnergyTradingGate, GateStatistics
from .gravitational_coupling import GravitationalCouplingMatrix
from .gravity import (
    compute_market_gravity,
    gravitational_force,
    gravitational_potential,
    market_gravity_center,
)
from .higher_order_kuramoto import HigherOrderKuramotoEngine, HigherOrderKuramotoResult
from .landauer import (
    K_BOLTZMANN,
    LANDAUER_ENERGY,
    ROOM_TEMPERATURE,
    LandauerInferenceProfiler,
)
from .liquidity_coupling import CouplingBenchmarkResult, LiquidityCouplingMatrix
from .maxwell import (
    compute_market_field_curl,
    compute_market_field_divergence,
    propagate_price_wave,
    wave_energy,
)
from .newton import (
    compute_acceleration,
    compute_force,
    compute_momentum,
    compute_price_acceleration,
    compute_price_velocity,
)
from .newtonian_dynamics import FreeEnergyGate, NewtonianPriceDynamics
from .portfolio_conservation import PortfolioEnergyConservation
from .relativity import (
    compute_relative_time,
    lorentz_factor,
    lorentz_transform,
    relativistic_momentum,
    time_dilation_factor,
    velocity_addition,
)
from .thermodynamic_risk import ThermodynamicRiskGate
from .thermodynamics import (
    boltzmann_entropy,
    compute_free_energy,
    compute_market_temperature,
    gibbs_free_energy,
    is_thermodynamic_equilibrium,
    thermal_equilibrium_distance,
)
from .tsallis_gate import TsallisGateResult, TsallisRegime, TsallisRiskGate
from .uncertainty import (
    check_uncertainty_principle,
    heisenberg_uncertainty,
    information_limit,
    minimum_uncertainty_product,
    optimal_measurement_tradeoff,
    position_momentum_uncertainty,
)
from .wave_propagation import GraphDiffusionEngine

__all__ = [
    # Constants
    "PhysicsConstants",
    # Newton's Laws
    "compute_momentum",
    "compute_force",
    "compute_acceleration",
    "compute_price_velocity",
    "compute_price_acceleration",
    # Gravitation
    "gravitational_force",
    "gravitational_potential",
    "compute_market_gravity",
    "market_gravity_center",
    # Conservation
    "compute_market_energy",
    "compute_market_momentum",
    "check_energy_conservation",
    "check_momentum_conservation",
    # Thermodynamics
    "boltzmann_entropy",
    "compute_market_temperature",
    "compute_free_energy",
    "gibbs_free_energy",
    "thermal_equilibrium_distance",
    "is_thermodynamic_equilibrium",
    # Maxwell's Equations
    "compute_market_field_divergence",
    "compute_market_field_curl",
    "propagate_price_wave",
    "wave_energy",
    # Relativity
    "lorentz_factor",
    "lorentz_transform",
    "relativistic_momentum",
    "compute_relative_time",
    "velocity_addition",
    "time_dilation_factor",
    # Heisenberg Uncertainty
    "heisenberg_uncertainty",
    "minimum_uncertainty_product",
    "position_momentum_uncertainty",
    "check_uncertainty_principle",
    "optimal_measurement_tradeoff",
    "information_limit",
    # Physics Engine v2 — T1-T7
    "GravitationalCouplingMatrix",
    "NewtonianPriceDynamics",
    "FreeEnergyGate",
    "PortfolioEnergyConservation",
    "ThermodynamicRiskGate",
    "CoulombInteraction",
    "GraphDiffusionEngine",
    "LandauerInferenceProfiler",
    "K_BOLTZMANN",
    "ROOM_TEMPERATURE",
    "LANDAUER_ENERGY",
    "GeoSyncPhysicsEngine",
    "PhysicsEngineResult",
    # Research modules — T1-T7 v2
    "FormanRicciCurvature",
    "FormanRicciResult",
    "DualTrackRicciMonitor",
    "LiquidityCouplingMatrix",
    "CouplingBenchmarkResult",
    "ExplosiveSyncDetector",
    "ESProximityResult",
    "ESCircuitBreaker",
    "TsallisRiskGate",
    "TsallisGateResult",
    "TsallisRegime",
    "FreeEnergyTradingGate",
    "FreeEnergyTradeDecision",
    "GateStatistics",
    "HigherOrderKuramotoEngine",
    "HigherOrderKuramotoResult",
    "DiffusionVolatilityPredictor",
    "VolatilityFrontPrediction",
    "DiffusionBacktestResult",
]
