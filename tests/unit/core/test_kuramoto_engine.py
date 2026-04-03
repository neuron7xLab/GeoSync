# SPDX-License-Identifier: LicenseRef-TradePulse-Proprietary
"""Unit tests for the Kuramoto ODE simulation engine.

Covers:
    - Basic correctness (phase shapes, R ∈ [0, 1])
    - Deterministic behaviour with fixed seed
    - Shape / dimension validation
    - Edge cases: small N, invalid params, zero/negative dt
    - Regression: synchronisation in a known high-coupling scenario
    - Adjacency matrix coupling
    - Summary statistics completeness
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.kuramoto import KuramotoConfig, KuramotoEngine, KuramotoResult, run_simulation
from core.kuramoto.engine import _dtheta_dt, _order_parameter, _rk4_step


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def default_cfg() -> KuramotoConfig:
    """Default small configuration for fast unit tests."""
    return KuramotoConfig(N=10, K=1.0, dt=0.01, steps=200, seed=0)


@pytest.fixture()
def result_default(default_cfg: KuramotoConfig) -> KuramotoResult:
    return run_simulation(default_cfg)


# ──────────────────────────────────────────────────────────────────────────────
# Shape / dimension tests
# ──────────────────────────────────────────────────────────────────────────────


class TestOutputShapes:
    def test_phases_shape(self, default_cfg: KuramotoConfig, result_default: KuramotoResult) -> None:
        expected = (default_cfg.steps + 1, default_cfg.N)
        assert result_default.phases.shape == expected, (
            f"phases.shape should be {expected}, got {result_default.phases.shape}"
        )

    def test_order_parameter_shape(self, default_cfg: KuramotoConfig, result_default: KuramotoResult) -> None:
        assert result_default.order_parameter.shape == (default_cfg.steps + 1,)

    def test_time_axis_shape(self, default_cfg: KuramotoConfig, result_default: KuramotoResult) -> None:
        assert result_default.time.shape == (default_cfg.steps + 1,)

    def test_time_axis_values(self, default_cfg: KuramotoConfig, result_default: KuramotoResult) -> None:
        expected_last = default_cfg.steps * default_cfg.dt
        assert result_default.time[0] == pytest.approx(0.0)
        assert result_default.time[-1] == pytest.approx(expected_last)

    def test_initial_phases_preserved(self, result_default: KuramotoResult) -> None:
        """Row 0 must equal the initial condition, not a derivative."""
        cfg = result_default.config
        # Re-resolve initial condition using the same seed path
        engine = KuramotoEngine(cfg)
        np.testing.assert_array_equal(result_default.phases[0], engine._theta0)


# ──────────────────────────────────────────────────────────────────────────────
# Correctness tests
# ──────────────────────────────────────────────────────────────────────────────


class TestCorrectnessBasic:
    def test_order_parameter_in_unit_interval(self, result_default: KuramotoResult) -> None:
        R = result_default.order_parameter
        assert np.all(R >= 0.0), "R must be ≥ 0"
        assert np.all(R <= 1.0), "R must be ≤ 1"

    def test_perfectly_aligned_phases_give_R_one(self) -> None:
        """All oscillators starting in phase → R(0) = 1."""
        N = 8
        cfg = KuramotoConfig(
            N=N,
            K=1.0,
            dt=0.01,
            steps=10,
            theta0=np.zeros(N),
            omega=np.zeros(N),
        )
        result = run_simulation(cfg)
        # When omega=0 and phases=0, coupling term is also 0 → no movement
        assert result.order_parameter[0] == pytest.approx(1.0, abs=1e-9)
        assert result.order_parameter[-1] == pytest.approx(1.0, abs=1e-9)

    def test_uniformly_distributed_phases_give_R_near_zero(self) -> None:
        """Uniformly distributed phases → R ≈ 0 (for large N)."""
        N = 200
        theta0 = np.linspace(0, 2 * np.pi, N, endpoint=False)
        cfg = KuramotoConfig(
            N=N,
            K=0.0,
            dt=0.01,
            steps=1,
            theta0=theta0,
            omega=np.zeros(N),
        )
        result = run_simulation(cfg)
        assert result.order_parameter[0] < 0.05, (
            f"Uniform phases should give R ≈ 0; got {result.order_parameter[0]:.4f}"
        )

    def test_zero_coupling_preserves_phases(self) -> None:
        """K=0 → each oscillator evolves independently at its natural frequency."""
        N = 5
        omega = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        theta0 = np.zeros(N)
        dt = 0.1
        steps = 10
        cfg = KuramotoConfig(N=N, K=0.0, dt=dt, steps=steps, omega=omega, theta0=theta0)
        result = run_simulation(cfg)

        # Expected: θᵢ(t) ≈ ωᵢ · t  (Euler would be exact; RK4 is also near-exact for linear)
        t_final = steps * dt
        expected_final = theta0 + omega * t_final
        np.testing.assert_allclose(
            result.phases[-1], expected_final, rtol=1e-5, atol=1e-5
        )

    def test_phases_are_finite(self, result_default: KuramotoResult) -> None:
        assert np.isfinite(result_default.phases).all()

    def test_order_parameter_is_finite(self, result_default: KuramotoResult) -> None:
        assert np.isfinite(result_default.order_parameter).all()


# ──────────────────────────────────────────────────────────────────────────────
# Determinism tests
# ──────────────────────────────────────────────────────────────────────────────


class TestDeterminism:
    def test_same_seed_produces_identical_results(self) -> None:
        cfg = KuramotoConfig(N=15, K=1.2, dt=0.01, steps=300, seed=99)
        r1 = run_simulation(cfg)
        r2 = run_simulation(cfg)
        np.testing.assert_array_equal(r1.phases, r2.phases)
        np.testing.assert_array_equal(r1.order_parameter, r2.order_parameter)

    def test_different_seeds_produce_different_results(self) -> None:
        cfg_a = KuramotoConfig(N=10, K=1.0, dt=0.01, steps=100, seed=1)
        cfg_b = KuramotoConfig(N=10, K=1.0, dt=0.01, steps=100, seed=2)
        r_a = run_simulation(cfg_a)
        r_b = run_simulation(cfg_b)
        assert not np.array_equal(r_a.phases, r_b.phases), (
            "Different seeds should produce different phase trajectories"
        )

    def test_explicit_arrays_ignore_seed(self) -> None:
        N = 6
        omega = np.linspace(-1.0, 1.0, N)
        theta0 = np.linspace(0, np.pi, N)
        cfg_s0 = KuramotoConfig(N=N, K=1.0, dt=0.01, steps=50, seed=0, omega=omega, theta0=theta0)
        cfg_s1 = KuramotoConfig(N=N, K=1.0, dt=0.01, steps=50, seed=1, omega=omega, theta0=theta0)
        r0 = run_simulation(cfg_s0)
        r1 = run_simulation(cfg_s1)
        # Seed should have no effect when arrays are explicitly supplied
        np.testing.assert_array_equal(r0.phases, r1.phases)


# ──────────────────────────────────────────────────────────────────────────────
# Validation / error handling tests
# ──────────────────────────────────────────────────────────────────────────────


class TestValidation:
    def test_N_less_than_2_raises(self) -> None:
        with pytest.raises(Exception):  # pydantic ValidationError
            KuramotoConfig(N=1, K=1.0, dt=0.01, steps=10)

    def test_negative_dt_raises(self) -> None:
        with pytest.raises(Exception):
            KuramotoConfig(N=5, K=1.0, dt=-0.01, steps=10)

    def test_zero_dt_raises(self) -> None:
        with pytest.raises(Exception):
            KuramotoConfig(N=5, K=1.0, dt=0.0, steps=10)

    def test_zero_steps_raises(self) -> None:
        with pytest.raises(Exception):
            KuramotoConfig(N=5, K=1.0, dt=0.01, steps=0)

    def test_omega_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError, match="omega"):
            KuramotoConfig(N=5, K=1.0, dt=0.01, steps=10, omega=np.ones(3))

    def test_omega_wrong_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="omega"):
            KuramotoConfig(N=4, K=1.0, dt=0.01, steps=10, omega=np.ones((2, 2)))

    def test_omega_with_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="omega"):
            KuramotoConfig(N=3, K=1.0, dt=0.01, steps=10, omega=np.array([1.0, np.nan, 0.5]))

    def test_theta0_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError, match="theta0"):
            KuramotoConfig(N=5, K=1.0, dt=0.01, steps=10, theta0=np.zeros(7))

    def test_adjacency_wrong_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="adjacency"):
            KuramotoConfig(N=4, K=1.0, dt=0.01, steps=10, adjacency=np.ones((3, 3)))

    def test_adjacency_with_inf_raises(self) -> None:
        adj = np.ones((3, 3))
        adj[0, 1] = np.inf
        with pytest.raises(ValueError, match="adjacency"):
            KuramotoConfig(N=3, K=1.0, dt=0.01, steps=10, adjacency=adj)

    def test_theta0_with_inf_raises(self) -> None:
        with pytest.raises(ValueError, match="theta0"):
            KuramotoConfig(N=3, K=1.0, dt=0.01, steps=10, theta0=np.array([0.0, np.inf, 1.0]))

    def test_N_equals_2_is_valid(self) -> None:
        """Minimum valid N is 2."""
        cfg = KuramotoConfig(N=2, K=1.0, dt=0.01, steps=5, seed=0)
        result = run_simulation(cfg)
        assert result.phases.shape == (6, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Adjacency matrix coupling
# ──────────────────────────────────────────────────────────────────────────────


class TestAdjacencyMatrix:
    def test_all_ones_adjacency_matches_global_coupling(self) -> None:
        """A = ones(N,N) with diagonal zeroed should approximate K/N global coupling."""
        N = 6
        omega = np.zeros(N)
        theta0 = np.linspace(0, np.pi, N)
        dt = 0.01
        steps = 50

        # Global coupling
        cfg_global = KuramotoConfig(N=N, K=1.0, dt=dt, steps=steps, omega=omega, theta0=theta0)

        # Adjacency = full matrix / N  → effective weight per edge is 1/N (same as global)
        adj = np.ones((N, N))
        np.fill_diagonal(adj, 0.0)
        adj /= N
        cfg_adj = KuramotoConfig(N=N, K=1.0, dt=dt, steps=steps, omega=omega, theta0=theta0, adjacency=adj)

        r_global = run_simulation(cfg_global)
        r_adj = run_simulation(cfg_adj)

        np.testing.assert_allclose(r_global.phases, r_adj.phases, atol=1e-10)

    def test_disconnected_adjacency_halts_coupling(self) -> None:
        """Zero adjacency matrix → no coupling → oscillators evolve freely."""
        N = 4
        omega = np.array([1.0, 2.0, 3.0, 4.0])
        theta0 = np.zeros(N)
        dt = 0.1
        steps = 5
        adj = np.zeros((N, N))
        cfg = KuramotoConfig(N=N, K=5.0, dt=dt, steps=steps, omega=omega, theta0=theta0, adjacency=adj)
        result = run_simulation(cfg)

        expected = theta0 + omega * (steps * dt)
        np.testing.assert_allclose(result.phases[-1], expected, rtol=1e-5, atol=1e-5)


# ──────────────────────────────────────────────────────────────────────────────
# Regression test — known synchronisation scenario
# ──────────────────────────────────────────────────────────────────────────────


class TestSynchronizationRegression:
    def test_strong_coupling_leads_to_high_R(self) -> None:
        """Above the critical coupling K_c = 2·std(ω), oscillators should synchronise.

        For N=50 oscillators with ω ~ N(0,1), K_c ≈ 2.  Using K=6 (well above)
        and running for 1000 steps should yield final R > 0.9.
        """
        cfg = KuramotoConfig(N=50, K=6.0, dt=0.05, steps=1000, seed=42)
        result = run_simulation(cfg)
        assert result.order_parameter[-1] > 0.9, (
            f"Expected R > 0.9 for strong coupling; got {result.order_parameter[-1]:.4f}"
        )

    def test_zero_coupling_prevents_synchronisation(self) -> None:
        """With K=0 and uniformly distributed phases, R should stay low."""
        N = 100
        theta0 = np.linspace(0, 2 * np.pi, N, endpoint=False)
        omega = np.zeros(N)
        cfg = KuramotoConfig(N=N, K=0.0, dt=0.01, steps=500, theta0=theta0, omega=omega)
        result = run_simulation(cfg)
        assert result.order_parameter[-1] < 0.05, (
            f"Expected R ≈ 0 without coupling; got {result.order_parameter[-1]:.4f}"
        )

    def test_R_increases_monotonically_under_strong_coupling(self) -> None:
        """Under very strong coupling starting near-uniform, R should trend upward."""
        N = 30
        rng = np.random.default_rng(7)
        theta0 = rng.uniform(0, 2 * np.pi, N)
        omega = rng.standard_normal(N) * 0.1  # very narrow frequency spread

        cfg = KuramotoConfig(N=N, K=10.0, dt=0.01, steps=500, theta0=theta0, omega=omega)
        result = run_simulation(cfg)

        R_early = result.order_parameter[:100].mean()
        R_late = result.order_parameter[400:].mean()
        assert R_late > R_early, (
            f"R should increase under strong coupling; early={R_early:.4f}, late={R_late:.4f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────────────────────────────────────


class TestSummaryStatistics:
    def test_summary_keys_present(self, result_default: KuramotoResult) -> None:
        required = {"final_R", "mean_R", "max_R", "min_R", "std_R", "N", "K", "dt", "steps", "total_time"}
        assert required.issubset(result_default.summary.keys())

    def test_summary_final_R_matches_trajectory(self, result_default: KuramotoResult) -> None:
        assert result_default.summary["final_R"] == pytest.approx(
            result_default.order_parameter[-1], abs=1e-12
        )

    def test_summary_total_time_matches_config(self, result_default: KuramotoResult) -> None:
        expected = result_default.config.steps * result_default.config.dt
        assert result_default.summary["total_time"] == pytest.approx(expected)

    def test_summary_N_matches_config(self, result_default: KuramotoResult) -> None:
        assert result_default.summary["N"] == result_default.config.N

    def test_config_to_dict_roundtrip(self) -> None:
        cfg = KuramotoConfig(N=5, K=2.0, dt=0.05, steps=100, seed=3)
        d = cfg.to_dict()
        assert d["N"] == 5
        assert d["K"] == pytest.approx(2.0)
        assert d["seed"] == 3
        assert d["adjacency"] is None
        assert d["omega"] is None


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper unit tests
# ──────────────────────────────────────────────────────────────────────────────


class TestInternalHelpers:
    def test_order_parameter_all_same_phase(self) -> None:
        theta = np.zeros(20)
        assert _order_parameter(theta) == pytest.approx(1.0, abs=1e-12)

    def test_order_parameter_opposite_phases(self) -> None:
        """N/2 at 0, N/2 at π → R ≈ 0."""
        theta = np.array([0.0] * 50 + [np.pi] * 50)
        assert _order_parameter(theta) < 1e-9

    def test_dtheta_dt_no_coupling(self) -> None:
        """With all-zero adjacency, coupling term vanishes → dθ/dt = ω."""
        N = 5
        theta = np.ones(N)
        omega = np.arange(1, N + 1, dtype=float)
        adj = np.zeros((N, N))
        result = _dtheta_dt(theta, omega, adj)
        np.testing.assert_array_equal(result, omega)

    def test_rk4_step_zero_derivative_stays_constant(self) -> None:
        """θ does not change when dθ/dt = 0 everywhere (ω=0, adj=0)."""
        N = 4
        theta = np.array([0.1, 0.5, 1.2, 2.3])
        omega = np.zeros(N)
        adj = np.zeros((N, N))
        result = _rk4_step(theta, omega, adj, dt=0.1)
        np.testing.assert_array_almost_equal(result, theta)
