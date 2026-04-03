# SPDX-License-Identifier: MIT
"""Public API for the Kuramoto simulation subsystem.

Core engine:
    :class:`KuramotoConfig` — validated inputs
    :class:`KuramotoEngine` — deterministic RK4 integrator
    :class:`KuramotoResult` — typed outputs
    :func:`run_simulation` — one-shot convenience

Extended engines (Google/DeepMind-grade):
    :class:`JaxKuramotoEngine` — XLA-compiled GPU/TPU acceleration (jax.jit + vmap)
    :class:`SparseKuramotoEngine` — O(E) sparse coupling for million-node networks
    :class:`AdaptiveKuramotoEngine` — Dormand-Prince / LSODA adaptive step control
    :class:`DelayedKuramotoEngine` — DDE with time-delayed coupling τ_ij
    :class:`SecondOrderKuramotoEngine` — inertia + damping (swing equation)
    :class:`EarlyStoppingEngine` — convergence-based early termination

Analysis:
    :class:`PhaseTransitionAnalyzer` — automatic K_c bifurcation detection
"""

from __future__ import annotations

from .adaptive import AdaptiveKuramotoEngine
from .config import KuramotoConfig
from .delayed import DelayedKuramotoEngine
from .early_stopping import EarlyStoppingEngine
from .engine import KuramotoEngine, KuramotoResult, run_simulation
from .phase_transition import PhaseTransitionAnalyzer, PhaseTransitionReport
from .ricci_flow_engine import KuramotoRicciFlowEngine, KuramotoRicciFlowResult
from .second_order import SecondOrderKuramotoEngine, SecondOrderResult
from .sparse import SparseKuramotoEngine

__all__ = [
    # Core
    "KuramotoConfig",
    "KuramotoEngine",
    "KuramotoResult",
    "run_simulation",
    # Extended engines
    "AdaptiveKuramotoEngine",
    "SparseKuramotoEngine",
    "DelayedKuramotoEngine",
    "SecondOrderKuramotoEngine",
    "SecondOrderResult",
    "EarlyStoppingEngine",
    "KuramotoRicciFlowEngine",
    "KuramotoRicciFlowResult",
    # Analysis
    "PhaseTransitionAnalyzer",
    "PhaseTransitionReport",
]

# JAX engine is optional (requires jax + jaxlib)
try:
    from .jax_engine import JAX_AVAILABLE, JaxKuramotoEngine

    __all__ += ["JaxKuramotoEngine", "JAX_AVAILABLE"]
except ImportError:
    JAX_AVAILABLE = False
