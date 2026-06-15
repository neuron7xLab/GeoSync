# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Mathematical witnesses for the thermodynamics laws in the physics catalog.

Each binds a pytest function to a first-principles catalog law via ``@law`` and
proves it holds in the real ``core.physics`` code, deriving every tolerance from
the law. Two previously-unwitnessed laws gain witnesses:

* ``thermo.second_law_closed`` — entropy of a closed subsystem is
  non-decreasing under coarse-graining / mixing (INV-TH2; any doubly-stochastic
  relaxation toward the uniform distribution cannot lower Boltzmann entropy).
* ``thermo.landauer_bound``    — erasing one bit costs ≥ k_B·T·ln 2 of free
  energy (Landauer floor; anchored by INV-LANDAUER-PROXY).

Nothing here is a market claim. These are thermodynamics facts about the code.
"""

from __future__ import annotations

import math

import numpy as np

from core.physics.landauer import K_BOLTZMANN, LandauerInferenceProfiler
from core.physics.thermodynamics import boltzmann_entropy
from physics_contracts.law import law

# Catalog ``thermo.second_law_closed`` slack: S(t+1) − S(t) ≥ −1e-12.
ENTROPY_SLACK = 1e-12
# Landauer is an exact algebraic identity; only float round-off is allowed.
LANDAUER_REL_TOL = 1e-15


@law("thermo.second_law_closed", n_distributions=40, mixing_fractions=(0.1, 0.3, 0.6))
def test_entropy_non_decreasing_under_mixing() -> None:
    """INV-TH2: a closed-subsystem mixing step never lowers Boltzmann entropy.

    Relaxation toward the uniform distribution, p ← (1−λ)p + λ·u, is a
    doubly-stochastic map; by concavity of S = −k_B Σ p ln p it cannot decrease
    entropy. The witness sweeps random distributions and mixing fractions and
    tracks the worst-case ΔS, which must stay non-negative within bit-level slack.
    """
    n_distributions = 40
    mixing_fractions = (0.1, 0.3, 0.6)
    rng = np.random.default_rng(seed=0)
    worst_delta_s = math.inf
    for _ in range(n_distributions):
        size = int(rng.integers(4, 10))
        p = rng.dirichlet(np.ones(size))
        uniform = np.ones(size) / size
        s_before = boltzmann_entropy(p)
        for lam in mixing_fractions:
            p_mixed = (1.0 - lam) * p + lam * uniform
            delta_s = boltzmann_entropy(p_mixed) - s_before
            worst_delta_s = min(worst_delta_s, delta_s)
    assert worst_delta_s >= -ENTROPY_SLACK, (
        "INV-TH2 violated: observed worst ΔS = "
        f"{worst_delta_s:.3e} must be ≥ −{ENTROPY_SLACK:.0e} under mixing over "
        f"n_distributions={n_distributions} with seed=0; entropy cannot decrease "
        "under a doubly-stochastic relaxation of a closed subsystem"
    )


@law("thermo.landauer_bound", n_temperatures=5, bits=8)
def test_landauer_erasure_floor_is_kT_ln2() -> None:
    """INV-LANDAUER-PROXY: per-bit erasure energy equals the Landauer floor.

    The minimum free energy to erase one bit is exactly k_B·T·ln 2, and the cost
    of erasing N bits is N times that floor — a hard physical lower bound. The
    witness checks the identity across a range of temperatures and asserts the
    N-bit cost never falls below N·k_B·T·ln 2.
    """
    n_temperatures = 5
    bits = 8
    worst_rel_error = 0.0
    floor_respected = True
    for temperature in np.geomspace(1e-3, 1e3, n_temperatures):
        profiler = LandauerInferenceProfiler(T=float(temperature))
        expected_floor = K_BOLTZMANN * float(temperature) * math.log(2)
        rel_error = abs(profiler.landauer_per_bit - expected_floor) / expected_floor
        worst_rel_error = max(worst_rel_error, rel_error)
        if profiler.minimum_energy(bits) < bits * expected_floor - 1e-30:
            floor_respected = False
    assert worst_rel_error <= LANDAUER_REL_TOL, (
        "thermo.landauer_bound / INV-LANDAUER-PROXY violated: observed worst "
        f"relative error = {worst_rel_error:.2e} must be ≤ {LANDAUER_REL_TOL:.0e} "
        f"with n_temperatures={n_temperatures}; per-bit energy must equal k_B·T·ln 2"
    )
    assert floor_respected, (
        "thermo.landauer_bound / INV-LANDAUER-PROXY violated: observed N-bit "
        f"erasure cost fell below N·k_B·T·ln 2 with bits={bits} and seed=0; the "
        "Landauer floor is a hard physical lower bound and must be respected"
    )
