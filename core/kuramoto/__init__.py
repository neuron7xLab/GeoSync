# SPDX-License-Identifier: LicenseRef-TradePulse-Proprietary
"""Kuramoto coupled-oscillator simulation package.

This package provides a production-ready implementation of the Kuramoto model —
a canonical model of coupled phase oscillators widely used in neuroscience,
physics, and synchrony analysis.

The model integrates the ODE:

    dθᵢ/dt = ωᵢ + (K/N) · Σⱼ sin(θⱼ − θᵢ)          [global coupling]

    dθᵢ/dt = ωᵢ + K · Σⱼ Aᵢⱼ sin(θⱼ − θᵢ)           [adjacency-matrix coupling]

Key exports:
    KuramotoConfig     — validated parameter container (Pydantic v2)
    KuramotoEngine     — deterministic ODE solver (RK4)
    KuramotoResult     — typed result with phase trajectories & summary stats
    run_simulation     — convenience one-shot function
"""

from __future__ import annotations

from .config import KuramotoConfig
from .engine import KuramotoEngine, KuramotoResult, run_simulation

__all__ = [
    "KuramotoConfig",
    "KuramotoEngine",
    "KuramotoResult",
    "run_simulation",
]
