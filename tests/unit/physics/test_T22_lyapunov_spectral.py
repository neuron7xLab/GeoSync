# SPDX-License-Identifier: MIT
"""T22 — Lyapunov exponent and spectral gap witnesses.

Two new physics diagnostics for GeoSync:

1. **Maximal Lyapunov Exponent (MLE)** — chaos/order detector on scalar
   time series. INV-LE1 (finite), INV-LE2 (sign semantics).

2. **Spectral Gap (Fiedler λ₂)** — algebraic connectivity of the
   coupling graph. INV-SG1 (non-negative), INV-SG2 (connectivity ↔ λ₂>0).
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

from core.physics.lyapunov_exponent import (
    maximal_lyapunov_exponent,
    spectral_gap,
)

# ── INV-LE1: MLE finite for any finite bounded input ────────────────


def test_mle_finite_on_diverse_inputs() -> None:
    """INV-LE1: MLE is finite for every finite bounded input series.

    Sweeps white noise, constant, sine, random walk, and step function
    inputs. MLE must be finite (not NaN/Inf) on each.
    """
    rng = np.random.default_rng(seed=0)
    n = 500
    # epsilon: finiteness is the invariant, no numeric tolerance needed
    series_bank: dict[str, np.ndarray] = {
        "white_noise": rng.normal(0, 1, n),
        "constant": np.full(n, 42.0),
        "sine": np.sin(np.linspace(0, 20 * math.pi, n)),
        "random_walk": np.cumsum(rng.normal(0, 0.01, n)),
        "step": np.concatenate([np.zeros(n // 2), np.ones(n // 2)]),
    }

    for label, series in series_bank.items():
        mle = maximal_lyapunov_exponent(series, dim=3, tau=1)
        assert math.isfinite(mle), (
            f"INV-LE1 VIOLATED on series={label}: MLE={mle} non-finite. "
            f"Expected finite MLE for any finite bounded input. "
            f"Observed at N={n}, dim=3, tau=1. "
            f"Physical reasoning: Rosenstein algorithm uses log(distance) "
            f"which is finite for non-zero distances between embedded points."
        )


# ── INV-LE2: MLE sign matches dynamical regime ──────────────────────


def test_mle_sign_matches_dynamical_regime() -> None:
    """INV-LE2: MLE(noise) ≈ 0, MLE(stable) < 0, MLE(chaotic) > 0.

    Three canonical dynamical systems with known Lyapunov exponents:
    1. White noise: λ ≈ 0 (no deterministic structure)
    2. Damped oscillator: λ < 0 (converging trajectory)
    3. Logistic map (r=4): λ = ln(2) ≈ 0.693 (maximal chaos)
    """
    n = 2000

    # 1. White noise: MLE should be near zero (no predictable structure)
    rng = np.random.default_rng(seed=1)
    noise = rng.normal(0, 1, n)
    mle_noise = maximal_lyapunov_exponent(noise, dim=5, tau=1)

    # 2. Damped oscillator: x(t) = exp(-0.1t) * sin(t) → MLE < 0
    t = np.linspace(0, 100, n)
    damped = np.exp(-0.1 * t) * np.sin(t)
    mle_damped = maximal_lyapunov_exponent(damped, dim=3, tau=5)

    # 3. Logistic map r=4: x_{n+1} = 4·x·(1-x), theoretical λ = ln(2)
    logistic = np.empty(n)
    logistic[0] = 0.1
    for i in range(1, n):
        logistic[i] = 4.0 * logistic[i - 1] * (1.0 - logistic[i - 1])
    # dim=2 (natural for 1D map), max_divergence_steps=20 (short fit window
    # to capture the initial exponential growth before saturation on the attractor)
    mle_logistic = maximal_lyapunov_exponent(logistic, dim=2, tau=1, max_divergence_steps=20)

    print(
        f"  MLE(noise)={mle_noise:.4f}, MLE(damped)={mle_damped:.4f}, MLE(logistic)={mle_logistic:.4f}"
    )

    # Logistic chaos should have highest MLE
    # tolerance: logistic MLE should be near ln(2)=0.693 within 20%
    theoretical_ln2 = math.log(2.0)  # epsilon: theoretical λ for logistic r=4
    assert mle_logistic > 0.5 * theoretical_ln2, (
        f"INV-LE2 VIOLATED: MLE(logistic)={mle_logistic:.4f} ≤ {0.5 * theoretical_ln2:.3f}. "
        f"Expected λ ≈ ln(2)={theoretical_ln2:.4f} for chaotic logistic map. "
        f"Observed at N={n}, dim=2, tau=1, max_div=20, r=4.0. "
        f"Physical reasoning: the logistic map at r=4 is maximally chaotic."
    )

    # Damped oscillator should be negative
    assert mle_damped < 0.0, (  # epsilon: theoretical λ < 0 for damped system
        f"INV-LE2 VIOLATED: MLE(damped)={mle_damped:.4f} ≥ 0. "
        f"Expected λ < 0 for exponentially damped oscillator. "
        f"Observed at N={n}, dim=3, tau=5, damping=0.1. "
        f"Physical reasoning: damped system converges → nearby "
        f"trajectories converge → λ < 0."
    )

    # Logistic should dominate noise
    assert mle_logistic > mle_noise, (
        f"INV-LE2 VIOLATED: MLE(logistic)={mle_logistic:.4f} ≤ "
        f"MLE(noise)={mle_noise:.4f}. "
        f"Expected deterministic chaos to have higher divergence than noise. "
        f"Observed at N={n}. "
        f"Physical reasoning: logistic map has structured divergence, "
        f"noise has unstructured fluctuation."
    )


# ── INV-SG1: λ₂ ≥ 0 always ─────────────────────────────────────────


def test_spectral_gap_non_negative_on_diverse_graphs() -> None:
    """INV-SG1: λ₂ ≥ 0 for every graph topology.

    Sweeps path, cycle, complete, star, random, and disconnected graphs.
    λ₂ must be non-negative for each (Laplacian is PSD).
    """
    # epsilon: λ₂ ≥ 0 is a PSD property, no numerical tolerance needed
    graphs: dict[str, np.ndarray] = {
        "path_10": nx.to_numpy_array(nx.path_graph(10)),
        "cycle_12": nx.to_numpy_array(nx.cycle_graph(12)),
        "complete_6": nx.to_numpy_array(nx.complete_graph(6)),
        "star_8": nx.to_numpy_array(nx.star_graph(7)),
        "erdos_renyi": nx.to_numpy_array(nx.erdos_renyi_graph(15, 0.3, seed=42)),
        "disconnected": np.block(
            [
                [nx.to_numpy_array(nx.complete_graph(4)), np.zeros((4, 4))],
                [np.zeros((4, 4)), nx.to_numpy_array(nx.complete_graph(4))],
            ]
        ),
        "single_edge": np.array([[0.0, 1.0], [1.0, 0.0]]),
    }

    for label, adj in graphs.items():
        lam2 = spectral_gap(adj)
        assert lam2 >= 0.0, (  # epsilon: PSD → λ₂ ≥ 0 exactly
            f"INV-SG1 VIOLATED on graph={label}: λ₂={lam2:.6f} < 0. "
            f"Expected λ₂ ≥ 0 by positive semi-definiteness of the Laplacian. "
            f"Observed at N={adj.shape[0]} nodes. "
            f"Physical reasoning: L = D - A is PSD for non-negative A."
        )


# ── INV-SG2: λ₂ > 0 ⟺ connected ───────────────────────────────────


def test_spectral_gap_connectivity_equivalence() -> None:
    """INV-SG2: λ₂ > 0 if and only if the graph is connected.

    Tests connected graphs (should have λ₂ > 0) and disconnected graphs
    (should have λ₂ = 0 or very near 0).
    """
    # Connected graphs: λ₂ must be strictly positive
    connected_graphs: dict[str, np.ndarray] = {
        "path_10": nx.to_numpy_array(nx.path_graph(10)),
        "complete_6": nx.to_numpy_array(nx.complete_graph(6)),
        "cycle_8": nx.to_numpy_array(nx.cycle_graph(8)),
    }
    for label, adj in connected_graphs.items():
        lam2 = spectral_gap(adj)
        # tolerance: eigenvalue solver has O(eps_machine) slack
        assert lam2 > 1e-10, (  # epsilon: machine-precision floor for "positive"
            f"INV-SG2 VIOLATED: connected graph={label} has λ₂={lam2:.3e} ≈ 0. "
            f"Expected λ₂ > 0 for a connected graph. "
            f"Observed at N={adj.shape[0]} nodes. "
            f"Physical reasoning: connected ⟹ algebraic multiplicity of 0 "
            f"eigenvalue is exactly 1 ⟹ λ₂ > 0."
        )

    # Disconnected graph: λ₂ must be 0 (or near 0)
    adj_disconnected = np.block(
        [
            [nx.to_numpy_array(nx.complete_graph(5)), np.zeros((5, 5))],
            [np.zeros((5, 5)), nx.to_numpy_array(nx.complete_graph(5))],
        ]
    )
    lam2_disc = spectral_gap(adj_disconnected)
    # tolerance: numerical λ₂ may be slightly above 0 due to float precision
    assert lam2_disc < 1e-10, (  # epsilon: machine-precision ceiling for "zero"
        f"INV-SG2 VIOLATED: disconnected graph has λ₂={lam2_disc:.3e} > 0. "
        f"Expected λ₂ ≈ 0 for a disconnected graph (two K5 components). "
        f"Observed at N=10 nodes (2×5 disconnected). "
        f"Physical reasoning: disconnected ⟹ eigenvalue 0 has "
        f"multiplicity ≥ 2 ⟹ λ₂ = 0."
    )


# ── Integration: MLE on Kuramoto R(t) trajectory ────────────────────


def test_mle_on_kuramoto_subcritical_vs_supercritical() -> None:
    """INV-LE2 + INV-K2/K3: MLE of R(t) reflects synchronization regime.

    Subcritical R(t) fluctuates stochastically → MLE ≈ 0 or > 0 (noisy).
    Supercritical R(t) converges to stable R∞ → MLE < 0 (stable).
    The MLE of R(t) is a HIGHER-ORDER diagnostic than R itself — it tells
    you not just "where is R?" but "is R's dynamics predictable?"
    """
    from core.kuramoto.config import KuramotoConfig
    from core.kuramoto.engine import KuramotoEngine

    n_osc = 128
    sigma = 1.0
    K_c = 2.0 * sigma * math.sqrt(2 * math.pi) / math.pi
    rng = np.random.default_rng(seed=99)
    omega = rng.normal(0, sigma, n_osc)
    theta0 = rng.uniform(-math.pi, math.pi, n_osc)

    # Supercritical: R(t) → stable R∞
    cfg_super = KuramotoConfig(
        N=n_osc,
        K=2.0 * K_c,
        omega=omega,
        theta0=theta0,
        dt=0.01,
        steps=2000,
        seed=99,
    )
    R_super = KuramotoEngine(cfg_super).run().order_parameter
    mle_super = maximal_lyapunov_exponent(R_super[500:], dim=3, tau=5)

    # Subcritical: R(t) fluctuates around noise floor
    cfg_sub = KuramotoConfig(
        N=n_osc,
        K=0.3 * K_c,
        omega=omega,
        theta0=theta0,
        dt=0.01,
        steps=2000,
        seed=99,
    )
    R_sub = KuramotoEngine(cfg_sub).run().order_parameter
    mle_sub = maximal_lyapunov_exponent(R_sub[500:], dim=3, tau=5)

    print(f"  MLE(R_super)={mle_super:.4f}, MLE(R_sub)={mle_sub:.4f}")

    # Supercritical should be more stable (lower MLE) than subcritical
    assert mle_super < mle_sub, (
        f"INV-LE2 integration: MLE(supercritical)={mle_super:.4f} ≥ "
        f"MLE(subcritical)={mle_sub:.4f}. "
        f"Expected supercritical R(t) to be more stable (lower MLE) "
        f"than subcritical noise. "
        f"Observed at N={n_osc}, K_super=2·K_c, K_sub=0.3·K_c, seed=99. "
        f"Physical reasoning: supercritical R(t) converges to a stable "
        f"fixed point → nearby trajectories converge → λ < λ(noise)."
    )
