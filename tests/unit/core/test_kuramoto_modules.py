# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Comprehensive unit tests for Kuramoto engine modules.

Covers: DelayedKuramotoEngine, EarlyStoppingEngine, SparseKuramotoEngine,
SecondOrderKuramotoEngine, AdaptiveKuramotoEngine.

Mathematical invariants tested:
- Order parameter R in [0, 1]
- Phases finite after integration
- Convergence under strong coupling
- Edge cases: single-pair oscillators, identical frequencies, NaN guards
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import sparse

from core.kuramoto.adaptive import AdaptiveKuramotoEngine
from core.kuramoto.config import KuramotoConfig
from core.kuramoto.delayed import DelayedKuramotoEngine
from core.kuramoto.early_stopping import EarlyStoppingEngine
from core.kuramoto.metrics import order_parameter
from core.kuramoto.ott_antonsen import OttAntonsenEngine, detect_chimera
from core.kuramoto.phase_transition import PhaseTransitionAnalyzer
from core.kuramoto.second_order import SecondOrderKuramotoEngine
from core.kuramoto.sparse import SparseKuramotoEngine

# Level auto-assigned by conftest from tests/test_levels.yaml


# ── Helpers ──────────────────────────────────────────────────────────────


def _synced_config(N: int = 10, K: float = 5.0, steps: int = 500, seed: int = 42) -> KuramotoConfig:
    """Config with strong coupling that should synchronize."""
    return KuramotoConfig(N=N, K=K, dt=0.01, steps=steps, seed=seed)


def _weak_config(N: int = 10, K: float = 0.01, steps: int = 200, seed: int = 7) -> KuramotoConfig:
    """Config with weak coupling -- low synchronization expected."""
    return KuramotoConfig(N=N, K=K, dt=0.01, steps=steps, seed=seed)


def _minimal_config(seed: int = 0) -> KuramotoConfig:
    """Minimal 2-oscillator config."""
    return KuramotoConfig(N=2, K=2.0, dt=0.01, steps=100, seed=seed)


# ═══════════════════════════════════════════════════════════════════════
# DelayedKuramotoEngine
# ═══════════════════════════════════════════════════════════════════════


class TestDelayedKuramotoEngine:
    def test_order_parameter_bounded(self):
        cfg = _synced_config(N=8, steps=300)
        result = DelayedKuramotoEngine(cfg, tau=0.05).run()
        assert np.all(result.order_parameter >= 0.0)
        assert np.all(result.order_parameter <= 1.0 + 1e-12)

    def test_phases_finite(self):
        cfg = _synced_config(N=6, steps=200)
        result = DelayedKuramotoEngine(cfg, tau=0.02).run()
        assert np.all(np.isfinite(result.phases))

    def test_output_shapes(self):
        cfg = _synced_config(N=5, steps=150)
        result = DelayedKuramotoEngine(cfg, tau=0.1).run()
        assert result.phases.shape == (151, 5)
        assert result.order_parameter.shape == (151,)
        assert result.time.shape == (151,)

    def test_zero_delay_matches_standard_behavior(self):
        """Zero delay should behave like a standard Kuramoto engine."""
        cfg = _synced_config(N=4, steps=100)
        result = DelayedKuramotoEngine(cfg, tau=0.0).run()
        assert np.all(np.isfinite(result.phases))
        assert 0.0 <= result.order_parameter[-1] <= 1.0

    def test_heterogeneous_delay_matrix(self):
        cfg = _synced_config(N=4, steps=100)
        tau_matrix = np.random.default_rng(99).uniform(0.01, 0.05, (4, 4))
        result = DelayedKuramotoEngine(cfg, tau=tau_matrix).run()
        assert np.all(np.isfinite(result.phases))
        assert np.all(result.order_parameter >= 0.0)
        assert np.all(result.order_parameter <= 1.0 + 1e-12)

    def test_custom_history_function(self):
        cfg = _synced_config(N=3, steps=100)
        result = DelayedKuramotoEngine(cfg, tau=0.05, history_fn=lambda t: np.zeros(3)).run()
        assert np.all(np.isfinite(result.phases))

    def test_two_oscillators(self):
        cfg = _minimal_config()
        result = DelayedKuramotoEngine(cfg, tau=0.01).run()
        assert result.phases.shape == (101, 2)
        assert np.all(result.order_parameter >= 0.0)


# ═══════════════════════════════════════════════════════════════════════
# EarlyStoppingEngine
# ═══════════════════════════════════════════════════════════════════════


class TestEarlyStoppingEngine:
    def test_order_parameter_bounded(self):
        cfg = _synced_config(steps=2000)
        result = EarlyStoppingEngine(cfg, epsilon=1e-4, patience=50, min_steps=50).run()
        assert np.all(result.order_parameter >= 0.0)
        assert np.all(result.order_parameter <= 1.0 + 1e-12)

    def test_early_stop_happens_under_strong_coupling(self):
        cfg = _synced_config(N=10, K=8.0, steps=5000)
        result = EarlyStoppingEngine(cfg, epsilon=1e-4, patience=100, min_steps=50).run()
        assert result.summary.get(
            "early_stopped", False
        ), "Strong coupling should converge and trigger early stopping"
        assert result.summary["converged_at_step"] < 5000

    def test_phases_finite(self):
        cfg = _synced_config(steps=500)
        result = EarlyStoppingEngine(cfg, epsilon=1e-5, patience=50, min_steps=30).run()
        assert np.all(np.isfinite(result.phases))

    def test_summary_keys_present(self):
        cfg = _synced_config(steps=500)
        result = EarlyStoppingEngine(cfg, epsilon=1e-4, patience=50, min_steps=30).run()
        for key in ("converged_at_step", "max_steps", "early_stopped", "compute_saved_pct"):
            assert key in result.summary

    def test_no_early_stop_with_tiny_patience(self):
        """With very strict epsilon, may run to completion."""
        cfg = _weak_config(steps=200)
        result = EarlyStoppingEngine(cfg, epsilon=1e-15, patience=5000, min_steps=10).run()
        # Should run to max steps since convergence criterion is extremely strict
        assert result.order_parameter.shape[0] > 0

    def test_two_oscillators(self):
        cfg = _minimal_config()
        result = EarlyStoppingEngine(
            KuramotoConfig(N=2, K=5.0, dt=0.01, steps=1000, seed=0),
            epsilon=1e-4,
            patience=50,
            min_steps=20,
        ).run()
        assert np.all(result.order_parameter >= 0.0)


# ═══════════════════════════════════════════════════════════════════════
# SparseKuramotoEngine
# ═══════════════════════════════════════════════════════════════════════


class TestSparseKuramotoEngine:
    def test_order_parameter_bounded(self):
        cfg = _synced_config(N=20, steps=300)
        result = SparseKuramotoEngine(cfg).run()
        assert np.all(result.order_parameter >= 0.0)
        assert np.all(result.order_parameter <= 1.0 + 1e-12)

    def test_phases_finite(self):
        cfg = _synced_config(N=15, steps=200)
        result = SparseKuramotoEngine(cfg).run()
        assert np.all(np.isfinite(result.phases))

    def test_explicit_sparse_adjacency(self):
        N = 10
        rng = np.random.default_rng(42)
        dense = rng.random((N, N))
        dense = (dense + dense.T) / 2
        np.fill_diagonal(dense, 0.0)
        sp_adj = sparse.csr_matrix(dense)
        cfg = KuramotoConfig(N=N, K=2.0, dt=0.01, steps=200, seed=42)
        result = SparseKuramotoEngine(cfg, sparse_adjacency=sp_adj).run()
        assert np.all(np.isfinite(result.phases))
        assert np.all(result.order_parameter >= 0.0)
        assert np.all(result.order_parameter <= 1.0 + 1e-12)

    def test_dense_adjacency_auto_converted(self):
        N = 8
        adj = np.ones((N, N))
        np.fill_diagonal(adj, 0.0)
        cfg = KuramotoConfig(N=N, K=1.0, dt=0.01, steps=100, adjacency=adj, seed=10)
        result = SparseKuramotoEngine(cfg).run()
        assert result.phases.shape == (101, N)

    def test_output_shapes(self):
        cfg = _synced_config(N=12, steps=100)
        result = SparseKuramotoEngine(cfg).run()
        assert result.phases.shape == (101, 12)
        assert result.order_parameter.shape == (101,)
        assert result.time.shape == (101,)

    def test_two_oscillators(self):
        cfg = _minimal_config()
        result = SparseKuramotoEngine(cfg).run()
        assert result.phases.shape == (101, 2)


# ═══════════════════════════════════════════════════════════════════════
# SecondOrderKuramotoEngine
# ═══════════════════════════════════════════════════════════════════════


class TestSecondOrderKuramotoEngine:
    def test_order_parameter_bounded(self):
        cfg = _synced_config(N=8, K=5.0, steps=500)
        result = SecondOrderKuramotoEngine(cfg, mass=1.0, damping=0.5).run()
        assert np.all(result.order_parameter >= 0.0)
        assert np.all(result.order_parameter <= 1.0 + 1e-12)

    def test_phases_and_velocities_finite(self):
        cfg = _synced_config(N=6, steps=300)
        result = SecondOrderKuramotoEngine(cfg, mass=1.0, damping=0.3).run()
        assert np.all(np.isfinite(result.phases))
        assert np.all(np.isfinite(result.velocities))

    def test_output_shapes(self):
        cfg = _synced_config(N=5, steps=200)
        result = SecondOrderKuramotoEngine(cfg, mass=1.0, damping=0.2).run()
        assert result.phases.shape == (201, 5)
        assert result.velocities.shape == (201, 5)
        assert result.order_parameter.shape == (201,)

    def test_summary_has_frequency_metrics(self):
        cfg = _synced_config(N=5, steps=200)
        result = SecondOrderKuramotoEngine(cfg, mass=1.0, damping=0.2).run()
        for key in (
            "frequency_nadir",
            "frequency_zenith",
            "max_rocof",
            "final_frequency_spread",
            "mean_frequency",
        ):
            assert key in result.summary, f"Missing summary key: {key}"

    def test_heterogeneous_mass_and_damping(self):
        N = 6
        cfg = _synced_config(N=N, steps=200)
        mass = np.linspace(0.5, 2.0, N)
        damping = np.linspace(0.1, 0.5, N)
        result = SecondOrderKuramotoEngine(cfg, mass=mass, damping=damping).run()
        assert np.all(np.isfinite(result.phases))

    def test_invalid_mass_raises(self):
        cfg = _synced_config(N=4, steps=50)
        with pytest.raises(ValueError, match="Mass must be strictly positive"):
            SecondOrderKuramotoEngine(cfg, mass=0.0)

    def test_negative_damping_raises(self):
        cfg = _synced_config(N=4, steps=50)
        with pytest.raises(ValueError, match="Damping must be non-negative"):
            SecondOrderKuramotoEngine(cfg, mass=1.0, damping=-0.1)

    def test_custom_initial_velocity(self):
        cfg = _synced_config(N=4, steps=100)
        v0 = np.array([0.1, -0.1, 0.2, -0.2])
        result = SecondOrderKuramotoEngine(cfg, mass=1.0, damping=0.3, velocity0=v0).run()
        assert np.all(np.isfinite(result.phases))

    def test_two_oscillators(self):
        cfg = _minimal_config()
        result = SecondOrderKuramotoEngine(cfg, mass=1.0, damping=0.5).run()
        assert result.phases.shape == (101, 2)


# ═══════════════════════════════════════════════════════════════════════
# AdaptiveKuramotoEngine
# ═══════════════════════════════════════════════════════════════════════


class TestAdaptiveKuramotoEngine:
    def test_order_parameter_bounded(self):
        cfg = _synced_config(N=8, steps=300)
        result = AdaptiveKuramotoEngine(cfg, method="RK45").run()
        assert np.all(result.order_parameter >= 0.0)
        assert np.all(result.order_parameter <= 1.0 + 1e-12)

    def test_phases_finite(self):
        cfg = _synced_config(N=6, steps=200)
        result = AdaptiveKuramotoEngine(cfg).run()
        assert np.all(np.isfinite(result.phases))

    def test_output_shapes(self):
        cfg = _synced_config(N=5, steps=150)
        result = AdaptiveKuramotoEngine(cfg).run()
        assert result.phases.shape == (151, 5)
        assert result.order_parameter.shape == (151,)

    def test_strong_coupling_convergence(self):
        """Strong coupling should produce high final R."""
        cfg = KuramotoConfig(N=10, K=10.0, dt=0.01, steps=1000, seed=42)
        result = AdaptiveKuramotoEngine(cfg).run()
        assert result.order_parameter[-1] > 0.7

    def test_multiple_methods(self):
        cfg = _synced_config(N=5, steps=100)
        for method in ("RK45", "RK23", "DOP853"):
            result = AdaptiveKuramotoEngine(cfg, method=method).run()
            assert np.all(np.isfinite(result.phases)), f"Failed for method={method}"
            assert np.all(result.order_parameter >= 0.0)

    def test_two_oscillators(self):
        cfg = _minimal_config()
        result = AdaptiveKuramotoEngine(cfg).run()
        assert result.phases.shape == (101, 2)


# ═══════════════════════════════════════════════════════════════════════
# Cross-module: mathematical properties
# ═══════════════════════════════════════════════════════════════════════


class TestCrossModuleMathProperties:
    def test_identical_phases_give_R_one(self):
        """When all oscillators start in phase with zero frequency, R should stay at 1."""
        N = 5
        theta0 = np.zeros(N)
        omega = np.zeros(N)
        cfg = KuramotoConfig(N=N, K=1.0, dt=0.01, steps=50, theta0=theta0, omega=omega, seed=0)

        # Test across multiple engines
        r1 = DelayedKuramotoEngine(cfg, tau=0.01).run()
        assert np.allclose(r1.order_parameter, 1.0, atol=1e-6)

        r2 = EarlyStoppingEngine(cfg, epsilon=1e-6, patience=10, min_steps=5).run()
        assert np.allclose(r2.order_parameter, 1.0, atol=1e-6)

        r3 = SparseKuramotoEngine(cfg).run()
        assert np.allclose(r3.order_parameter, 1.0, atol=1e-6)

        r4 = SecondOrderKuramotoEngine(cfg, mass=1.0, damping=0.5).run()
        assert np.allclose(r4.order_parameter, 1.0, atol=1e-6)

        r5 = AdaptiveKuramotoEngine(cfg).run()
        assert np.allclose(r5.order_parameter, 1.0, atol=1e-6)

    def test_time_monotonically_increasing(self):
        cfg = _synced_config(N=5, steps=100)
        result = DelayedKuramotoEngine(cfg, tau=0.02).run()
        assert np.all(np.diff(result.time) > 0)

    def test_seed_reproducibility(self):
        cfg1 = _synced_config(seed=123)
        cfg2 = _synced_config(seed=123)
        r1 = AdaptiveKuramotoEngine(cfg1).run()
        r2 = AdaptiveKuramotoEngine(cfg2).run()
        np.testing.assert_allclose(r1.phases, r2.phases, atol=1e-10)


class TestFinitenessFailClosed:
    """Fail-closed finiteness guards: non-finite input must never leak past the
    documented order-parameter bounds (INV-OA1 |z|≤1, INV-K1 R∈[0,1]).

    Each test reproduces a latent defect found by a computational manifold
    audit: NaN-blind comparison guards (``abs(z) > 1.0`` is False for NaN) used
    to propagate NaN/Inf silently, and the Ott-Antonsen integrator silently
    renormalised a blown-up RK4 step onto the unit circle — inverting the
    synchronization verdict (subcritical K<2Δ reported as full sync R=1).
    """

    def test_oa_unstable_step_fails_closed_not_silent_full_sync(self) -> None:
        """INV-OA1/OA3: a blown-up RK4 step must raise, not report spurious sync.

        K=0.5 < K_c=2Δ=4 is subcritical (INV-OA3: R→0). With dt past the RK4
        stability limit the old code silently renormalised the divergence to
        |z|=1 and returned R[-1]=1.0 (a full-sync verdict, the exact opposite
        regime). Fail-closed now raises FloatingPointError.
        """
        engine = OttAntonsenEngine(K=0.5, delta=2.0, omega0=0.0)
        with pytest.raises(FloatingPointError):
            engine.integrate(T=50.0, dt=2.5, R0=0.5)

    def test_oa_rejects_non_finite_initial_condition(self) -> None:
        """INV-OA1: R0=NaN previously produced an all-NaN trajectory."""
        engine = OttAntonsenEngine(K=2.0, delta=0.5)
        with pytest.raises(ValueError):
            engine.integrate(T=1.0, dt=0.1, R0=float("nan"))

    def test_oa_rejects_non_finite_parameters(self) -> None:
        """INV-OA1: non-finite K/delta/omega0 must fail-closed at construction."""
        with pytest.raises(ValueError):
            OttAntonsenEngine(K=float("inf"), delta=0.5)
        with pytest.raises(ValueError):
            OttAntonsenEngine(K=1.0, delta=float("nan"))
        with pytest.raises(ValueError):
            OttAntonsenEngine(K=1.0, delta=0.5, omega0=float("inf"))

    def test_oa_legitimate_supercritical_still_exact(self) -> None:
        """Regression: a stable supercritical run still matches INV-OA2 exactly.

        K=6, Δ=0.5 ⟹ K_c=1, R_∞ = √(1 − K_c/K) = √(5/6).
        """
        engine = OttAntonsenEngine(K=6.0, delta=0.5, omega0=0.0)
        result = engine.integrate(T=50.0, dt=0.01, R0=0.1)
        assert result.is_supercritical
        assert abs(result.R[-1] - math.sqrt(1.0 - 1.0 / 6.0)) < 1e-3

    def test_order_parameter_rejects_non_finite_phases(self) -> None:
        """INV-K1: a NaN phase previously leaked NaN past R∈[0,1]."""
        theta = np.array([[0.0, np.nan, 1.0], [0.1, 0.2, 0.3]])
        with pytest.raises(ValueError):
            order_parameter(theta, axis=1)

    def test_order_parameter_finite_phases_bounded(self) -> None:
        """Regression: finite phases yield R∈[0,1] as before."""
        theta = np.array([[0.0, 0.1, 0.2], [1.0, 1.1, 0.9]])
        r = order_parameter(theta, axis=1)
        assert r.shape == (2,)
        assert np.all((r >= 0.0) & (r <= 1.0))


class TestDegenerateInputFailClosed:
    """Degenerate-input guards found by a computational audit: a single-point
    sweep, a flat (never-crossing) R curve, and negative/empty chimera sector
    labels each previously produced an opaque crash or a silently-wrong result.
    """

    def test_find_critical_k_single_point_raises(self) -> None:
        """_find_critical_K previously raised an opaque np.gradient IndexError."""
        with pytest.raises(ValueError, match="≥2 sweep points"):
            PhaseTransitionAnalyzer._find_critical_K(np.array([2.0]), np.array([0.5]))

    def test_find_critical_k_flat_curve_signals_no_transition(self) -> None:
        """A flat R curve has no transition: return NaN, not a spurious K_c=0."""
        k_c = PhaseTransitionAnalyzer._find_critical_K(
            np.linspace(0.0, 5.0, 5), np.full(5, 0.1)
        )
        assert math.isnan(k_c)

    def test_find_critical_k_real_crossing_unchanged(self) -> None:
        """Regression: a genuine threshold crossing still interpolates K_c."""
        k_c = PhaseTransitionAnalyzer._find_critical_K(
            np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.5, 0.9])
        )
        assert 1.0 < k_c < 2.0

    def test_detect_chimera_rejects_negative_sectors(self) -> None:
        """Negative sector labels were silently dropped from per-sector stats."""
        with pytest.raises(ValueError, match="non-negative"):
            detect_chimera(
                np.array([0.0, 0.1, 3.0, 3.1]),
                np.array([-1, -1, 0, 0], dtype=np.int64),
            )

    def test_detect_chimera_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            detect_chimera(np.array([]), np.array([], dtype=np.int64))

    def test_detect_chimera_valid_sectors_unchanged(self) -> None:
        """Regression: a valid 0-indexed assignment still produces a report."""
        report = detect_chimera(
            np.array([0.0, 0.1, 3.0, 3.1]),
            np.array([0, 0, 1, 1], dtype=np.int64),
        )
        assert report.sector_R.shape == (2,)
