# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Unit tests for the Kuramoto ODE simulation engine."""

from __future__ import annotations

import numpy as np
import pytest

from core.kuramoto import KuramotoConfig, KuramotoEngine, KuramotoResult, run_simulation
from core.kuramoto.engine import _dtheta_dt, _order_parameter, _rk4_step


@pytest.fixture()
def default_cfg() -> KuramotoConfig:
    return KuramotoConfig(N=10, K=1.0, dt=0.01, steps=200, seed=0)


@pytest.fixture()
def result_default(default_cfg: KuramotoConfig) -> KuramotoResult:
    return run_simulation(default_cfg)


class TestOutputShapes:
    def test_shapes(self, default_cfg: KuramotoConfig, result_default: KuramotoResult) -> None:
        assert result_default.phases.shape == (default_cfg.steps + 1, default_cfg.N)
        assert result_default.order_parameter.shape == (default_cfg.steps + 1,)
        assert result_default.time.shape == (default_cfg.steps + 1,)

    def test_time_axis_values(
        self, default_cfg: KuramotoConfig, result_default: KuramotoResult
    ) -> None:
        assert result_default.time[0] == pytest.approx(0.0)
        assert result_default.time[-1] == pytest.approx(default_cfg.steps * default_cfg.dt)

    def test_initial_phases_match_engine_state(self, result_default: KuramotoResult) -> None:
        engine = KuramotoEngine(result_default.config)
        np.testing.assert_array_equal(result_default.phases[0], engine._theta0)


class TestCorrectness:
    def test_order_parameter_bounds(self, result_default: KuramotoResult) -> None:
        assert np.all(
            (0.0 <= result_default.order_parameter) & (result_default.order_parameter <= 1.0)
        )

    def test_finite_outputs(self, result_default: KuramotoResult) -> None:
        assert np.isfinite(result_default.phases).all()
        assert np.isfinite(result_default.order_parameter).all()

    def test_perfect_alignment_stays_synchronised(self) -> None:
        N = 8
        cfg = KuramotoConfig(N=N, K=1.0, dt=0.01, steps=10, theta0=np.zeros(N), omega=np.zeros(N))
        result = run_simulation(cfg)
        assert result.order_parameter[0] == pytest.approx(1.0)
        assert result.order_parameter[-1] == pytest.approx(1.0)

    def test_uniform_phases_are_desynchronised(self) -> None:
        N = 200
        theta0 = np.linspace(0, 2 * np.pi, N, endpoint=False)
        cfg = KuramotoConfig(N=N, K=0.0, dt=0.01, steps=1, theta0=theta0, omega=np.zeros(N))
        result = run_simulation(cfg)
        assert result.order_parameter[0] < 0.05

    def test_k_zero_matches_analytic_trajectory_all_steps(self) -> None:
        N = 4
        dt = 0.05
        steps = 20
        omega = np.array([0.1, -0.2, 0.5, 1.0])
        theta0 = np.array([0.0, 1.0, -1.0, 0.4])
        cfg = KuramotoConfig(N=N, K=0.0, dt=dt, steps=steps, omega=omega, theta0=theta0)
        result = run_simulation(cfg)
        expected = (
            theta0[np.newaxis, :] + np.arange(steps + 1)[:, np.newaxis] * dt * omega[np.newaxis, :]
        )
        np.testing.assert_allclose(result.phases, expected, atol=1e-10, rtol=1e-10)


class TestDeterminism:
    def test_same_seed_identical(self) -> None:
        cfg = KuramotoConfig(N=15, K=1.2, dt=0.01, steps=300, seed=99)
        r1 = run_simulation(cfg)
        r2 = run_simulation(cfg)
        np.testing.assert_array_equal(r1.phases, r2.phases)
        np.testing.assert_array_equal(r1.order_parameter, r2.order_parameter)

    def test_different_seed_differs(self) -> None:
        r_a = run_simulation(KuramotoConfig(N=10, K=1.0, dt=0.01, steps=100, seed=1))
        r_b = run_simulation(KuramotoConfig(N=10, K=1.0, dt=0.01, steps=100, seed=2))
        assert not np.array_equal(r_a.phases, r_b.phases)

    def test_explicit_arrays_ignore_seed(self) -> None:
        N = 6
        omega = np.linspace(-1.0, 1.0, N)
        theta0 = np.linspace(0, np.pi, N)
        r0 = run_simulation(
            KuramotoConfig(N=N, K=1.0, dt=0.01, steps=50, seed=0, omega=omega, theta0=theta0)
        )
        r1 = run_simulation(
            KuramotoConfig(N=N, K=1.0, dt=0.01, steps=50, seed=1, omega=omega, theta0=theta0)
        )
        np.testing.assert_array_equal(r0.phases, r1.phases)

    def test_explicit_omega_alone_ignores_seed(self) -> None:
        N = 5
        omega = np.linspace(-0.5, 0.5, N)
        engine_a = KuramotoEngine(
            KuramotoConfig(N=N, K=1.0, dt=0.01, steps=20, seed=1, omega=omega)
        )
        engine_b = KuramotoEngine(
            KuramotoConfig(N=N, K=1.0, dt=0.01, steps=20, seed=9, omega=omega)
        )
        np.testing.assert_array_equal(engine_a._omega, engine_b._omega)
        assert not np.array_equal(engine_a._theta0, engine_b._theta0)

    def test_explicit_theta0_alone_ignores_seed(self) -> None:
        N = 5
        theta0 = np.linspace(0.0, 1.0, N)
        a = run_simulation(KuramotoConfig(N=N, K=1.0, dt=0.01, steps=20, seed=1, theta0=theta0))
        b = run_simulation(KuramotoConfig(N=N, K=1.0, dt=0.01, steps=20, seed=9, theta0=theta0))
        np.testing.assert_array_equal(a.phases[0], b.phases[0])


class TestValidation:
    def test_invalid_core_params_raise(self) -> None:
        with pytest.raises(Exception):
            KuramotoConfig(N=1, K=1.0, dt=0.01, steps=10)
        with pytest.raises(Exception):
            KuramotoConfig(N=5, K=1.0, dt=-0.01, steps=10)
        with pytest.raises(Exception):
            KuramotoConfig(N=5, K=1.0, dt=0.0, steps=10)
        with pytest.raises(Exception):
            KuramotoConfig(N=5, K=1.0, dt=0.01, steps=0)

    def test_array_shape_and_finite_validation(self) -> None:
        with pytest.raises(ValueError, match="omega"):
            KuramotoConfig(N=5, K=1.0, dt=0.01, steps=10, omega=np.ones(3))
        with pytest.raises(ValueError, match="omega"):
            KuramotoConfig(N=4, K=1.0, dt=0.01, steps=10, omega=np.ones((2, 2)))
        with pytest.raises(ValueError, match="omega"):
            KuramotoConfig(N=3, K=1.0, dt=0.01, steps=10, omega=np.array([1.0, np.nan, 0.5]))
        with pytest.raises(ValueError, match="theta0"):
            KuramotoConfig(N=3, K=1.0, dt=0.01, steps=10, theta0=np.array([0.0, np.inf, 1.0]))
        with pytest.raises(ValueError, match="adjacency"):
            KuramotoConfig(N=4, K=1.0, dt=0.01, steps=10, adjacency=np.ones((3, 3)))
        with pytest.raises(ValueError, match="adjacency"):
            KuramotoConfig(
                N=3, K=1.0, dt=0.01, steps=10, adjacency=np.array([[0.0, 1.0, np.nan]] * 3)
            )

    def test_seed_and_coupling_validation(self) -> None:
        with pytest.raises(ValueError, match="seed"):
            KuramotoConfig(N=5, K=1.0, dt=0.01, steps=10, seed=-1)
        with pytest.raises(ValueError, match="K"):
            KuramotoConfig(N=5, K=np.inf, dt=0.01, steps=10)

    def test_config_to_dict_contract(self) -> None:
        cfg = KuramotoConfig(N=3, K=2.5, dt=0.1, steps=4, seed=7)
        payload = cfg.to_dict()
        assert payload["coupling_mode"] == "global"
        assert payload["seed"] == 7
        assert payload["adjacency"] is None

    def test_config_to_dict_adjacency_serialization_contract(self) -> None:
        adj = np.array([[0.0, 0.2], [0.3, 0.0]])
        cfg = KuramotoConfig(
            N=2,
            K=1.5,
            dt=0.1,
            steps=3,
            seed=5,
            omega=np.array([0.1, -0.2]),
            theta0=np.array([0.0, 1.0]),
            adjacency=adj,
            adjacency_kind="normalized_topology",
        )
        payload = cfg.to_dict()
        assert payload["coupling_mode"] == "adjacency"
        assert payload["adjacency"] == adj.tolist()
        assert payload["adjacency_kind"] == "normalized_topology"
        assert payload["claim_boundary"] == "physical_attractive_kuramoto_only"
        assert payload["omega"] == [0.1, -0.2]
        assert payload["theta0"] == [0.0, 1.0]


class TestAdjacencySemantics:
    def test_global_and_equivalent_adjacency_match(self) -> None:
        N = 6
        omega = np.zeros(N)
        theta0 = np.linspace(0, np.pi, N)
        adj = np.ones((N, N))
        np.fill_diagonal(adj, 0.0)
        adj /= N
        global_result = run_simulation(
            KuramotoConfig(N=N, K=1.0, dt=0.01, steps=50, omega=omega, theta0=theta0)
        )
        adj_result = run_simulation(
            KuramotoConfig(N=N, K=1.0, dt=0.01, steps=50, omega=omega, theta0=theta0, adjacency=adj)
        )
        np.testing.assert_allclose(global_result.phases, adj_result.phases, atol=1e-10)

    def test_diagonal_entries_are_ignored(self) -> None:
        N = 5
        omega = np.linspace(-0.2, 0.2, N)
        theta0 = np.linspace(0.0, 2.0, N)
        adj = np.ones((N, N)) / N
        np.fill_diagonal(adj, 1000.0)
        adj_zero_diag = adj.copy()
        np.fill_diagonal(adj_zero_diag, 0.0)

        with_diag = run_simulation(
            KuramotoConfig(N=N, K=1.0, dt=0.01, steps=40, omega=omega, theta0=theta0, adjacency=adj)
        )
        zero_diag = run_simulation(
            KuramotoConfig(
                N=N, K=1.0, dt=0.01, steps=40, omega=omega, theta0=theta0, adjacency=adj_zero_diag
            )
        )
        np.testing.assert_allclose(with_diag.phases, zero_diag.phases, atol=1e-12)

    def test_disconnected_adjacency_means_no_coupling(self) -> None:
        N = 4
        omega = np.array([1.0, 2.0, 3.0, 4.0])
        theta0 = np.zeros(N)
        cfg = KuramotoConfig(
            N=N,
            K=5.0,
            dt=0.1,
            steps=5,
            omega=omega,
            theta0=theta0,
            adjacency=np.zeros((N, N)),
            adjacency_kind="normalized_topology",
        )
        result = run_simulation(cfg)
        np.testing.assert_allclose(result.phases[-1], theta0 + omega * 0.5, rtol=1e-5, atol=1e-5)

    def test_permutation_invariance_under_global_coupling(self) -> None:
        N = 7
        omega = np.linspace(-1.0, 1.0, N)
        theta0 = np.linspace(0.2, 1.5, N)
        perm = np.array([2, 5, 1, 6, 0, 4, 3])
        base = run_simulation(
            KuramotoConfig(N=N, K=1.7, dt=0.02, steps=80, omega=omega, theta0=theta0)
        )
        permuted = run_simulation(
            KuramotoConfig(N=N, K=1.7, dt=0.02, steps=80, omega=omega[perm], theta0=theta0[perm])
        )
        np.testing.assert_allclose(base.order_parameter, permuted.order_parameter, atol=1e-12)


class TestSynchronizationRegression:
    def test_strong_coupling_increases_synchrony(self) -> None:
        cfg = KuramotoConfig(N=50, K=6.0, dt=0.05, steps=1000, seed=42)
        result = run_simulation(cfg)
        assert result.order_parameter[-1] > 0.9

    def test_small_controlled_system_trends_up_with_strong_coupling(self) -> None:
        N = 8
        theta0 = np.linspace(0, 2 * np.pi, N, endpoint=False)
        omega = np.linspace(-0.05, 0.05, N)
        cfg = KuramotoConfig(N=N, K=8.0, dt=0.02, steps=300, omega=omega, theta0=theta0)
        result = run_simulation(cfg)
        assert result.order_parameter[-1] > result.order_parameter[0]


class TestSummaryAndHelpers:
    def test_summary_keys_and_consistency(self, result_default: KuramotoResult) -> None:
        required = {
            "final_R",
            "mean_R",
            "max_R",
            "min_R",
            "std_R",
            "N",
            "K",
            "dt",
            "steps",
            "total_time",
            "coupling_mode",
            "seed",
        }
        assert required.issubset(result_default.summary)
        assert result_default.summary["final_R"] == pytest.approx(
            result_default.order_parameter[-1]
        )
        assert result_default.summary["coupling_mode"] == result_default.config.coupling_mode

    def test_result_constructor_validation(self, default_cfg: KuramotoConfig) -> None:
        phases = np.zeros((default_cfg.steps + 1, default_cfg.N))
        order = np.zeros(default_cfg.steps + 1)
        time = np.arange(default_cfg.steps + 1, dtype=float) * default_cfg.dt

        with pytest.raises(ValueError, match="phases"):
            KuramotoResult(phases=phases[:-1], order_parameter=order, time=time, config=default_cfg)
        with pytest.raises(ValueError, match="order_parameter"):
            KuramotoResult(phases=phases, order_parameter=order[:-1], time=time, config=default_cfg)
        with pytest.raises(ValueError, match="time"):
            KuramotoResult(phases=phases, order_parameter=order, time=time[:-1], config=default_cfg)
        phases_bad = phases.copy()
        phases_bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            KuramotoResult(phases=phases_bad, order_parameter=order, time=time, config=default_cfg)
        order_bad = order.copy()
        order_bad[0] = 1.0001
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            KuramotoResult(phases=phases, order_parameter=order_bad, time=time, config=default_cfg)

    def test_result_constructor_order_parameter_tolerance_boundaries(
        self, default_cfg: KuramotoConfig
    ) -> None:
        phases = np.zeros((default_cfg.steps + 1, default_cfg.N))
        time = np.arange(default_cfg.steps + 1, dtype=float) * default_cfg.dt
        base = np.zeros(default_cfg.steps + 1)
        base[0] = 0.0
        base[1] = 1.0

        # Accept boundary values and tiny floating noise (±1e-12).
        order_within_tol = base.copy()
        order_within_tol[2] = 1.0 + 5e-13
        order_within_tol[3] = -5e-13
        KuramotoResult(
            phases=phases, order_parameter=order_within_tol, time=time, config=default_cfg
        )

        # Reject values clearly outside tolerance.
        order_above = base.copy()
        order_above[2] = 1.0 + 2e-12
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            KuramotoResult(
                phases=phases, order_parameter=order_above, time=time, config=default_cfg
            )

        order_below = base.copy()
        order_below[2] = -2e-12
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            KuramotoResult(
                phases=phases, order_parameter=order_below, time=time, config=default_cfg
            )

    def test_order_parameter_helper(self) -> None:
        assert _order_parameter(np.zeros(20)) == pytest.approx(1.0)
        assert _order_parameter(np.array([0.0] * 50 + [np.pi] * 50)) < 1e-9

    def test_dtheta_dt_and_rk4_helpers(self) -> None:
        theta = np.array([0.1, 0.5, 1.2, 2.3])
        omega = np.zeros(4)
        adj = np.zeros((4, 4))
        np.testing.assert_array_equal(_dtheta_dt(theta, omega, adj), omega)
        np.testing.assert_array_almost_equal(_rk4_step(theta, omega, adj, dt=0.1), theta)


class TestFP1SingleCouplingOwner:
    """FP-1 (Issue #1096): ``K`` is the single owner of coupling scale.

    ``A_norm`` resolved by the engine must be topology-only (no hidden ``K``),
    ``K`` must be applied exactly once in the RHS, and the canonical
    ``KuramotoEngine`` output must be preserved within numerical tolerance.
    """

    @staticmethod
    def _symmetric_topology(n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        a = rng.random((n, n))
        a = (a + a.T) / 2.0
        np.fill_diagonal(a, 0.0)
        return a

    def test_K_not_embedded_in_adjacency(self) -> None:
        # Explicit mode: resolved A_norm must equal the caller topology and be
        # independent of K (no K folded into the stored matrix).
        a = self._symmetric_topology(8, seed=1)
        eng_lo = KuramotoEngine(
            KuramotoConfig(
                N=8, K=0.3, dt=0.01, steps=1, seed=0,
                adjacency=a, adjacency_kind="normalized_topology",
            )
        )
        eng_hi = KuramotoEngine(
            KuramotoConfig(
                N=8, K=9.0, dt=0.01, steps=1, seed=0,
                adjacency=a, adjacency_kind="normalized_topology",
            )
        )
        expected = a.copy()
        np.fill_diagonal(expected, 0.0)
        np.testing.assert_array_equal(eng_lo._adj, expected)
        np.testing.assert_array_equal(eng_hi._adj, expected)
        # Same topology, different K ⇒ identical A_norm (K is NOT embedded).
        np.testing.assert_array_equal(eng_lo._adj, eng_hi._adj)

    def test_global_mode_adjacency_is_topology_only(self) -> None:
        # Global mode: off-diagonal weight is 1/N, never K/N.
        n = 10
        for k in (0.5, 4.0):
            eng = KuramotoEngine(KuramotoConfig(N=n, K=k, dt=0.01, steps=1, seed=0))
            off = eng._adj[~np.eye(n, dtype=bool)]
            np.testing.assert_allclose(off, 1.0 / n, rtol=0.0, atol=0.0)

    def test_zero_diagonal_enforced(self) -> None:
        a = self._symmetric_topology(6, seed=2)
        np.fill_diagonal(a, 0.7)  # caller diagonal must be stripped
        eng_expl = KuramotoEngine(
            KuramotoConfig(
                N=6, K=1.0, dt=0.01, steps=1, seed=0,
                adjacency=a, adjacency_kind="normalized_topology",
            )
        )
        eng_glob = KuramotoEngine(KuramotoConfig(N=6, K=2.0, dt=0.01, steps=1, seed=0))
        np.testing.assert_array_equal(np.diag(eng_expl._adj), np.zeros(6))
        np.testing.assert_array_equal(np.diag(eng_glob._adj), np.zeros(6))

    def test_explicit_adjacency_rhs_uses_single_K(self) -> None:
        # K applied once over topology == K folded into the matrix once (K=1).
        a = self._symmetric_topology(7, seed=3)
        theta = np.linspace(0.0, 2.0, 7)
        omega = np.zeros(7)
        k = 3.1
        single_owner = _dtheta_dt(theta, omega, a, K=k)
        baked_once = _dtheta_dt(theta, omega, k * a, K=1.0)
        np.testing.assert_allclose(single_owner, baked_once, rtol=1e-12, atol=1e-14)

    def test_K_not_double_scaled(self) -> None:
        # Doubling K must scale the coupling term exactly linearly (no K²).
        a = self._symmetric_topology(7, seed=4)
        theta = np.linspace(0.1, 3.0, 7)
        omega = np.zeros(7)
        base = _dtheta_dt(theta, omega, a, K=1.0)
        doubled = _dtheta_dt(theta, omega, a, K=2.0)
        np.testing.assert_allclose(doubled, 2.0 * base, rtol=1e-12, atol=1e-14)

    def test_boundary_rhs_consistency(self) -> None:
        # The boundary spectral scale K·λ_max(A_norm) must equal λ_max of the
        # effective coupling K·A_norm that the RHS integrates.
        a = self._symmetric_topology(9, seed=5)
        eng = KuramotoEngine(
            KuramotoConfig(
                N=9, K=2.7, dt=0.01, steps=1, seed=0,
                adjacency=a, adjacency_kind="normalized_topology",
            )
        )
        lam_anorm = float(np.max(np.linalg.eigvalsh(eng._adj)))
        lam_effective = float(np.max(np.linalg.eigvalsh(eng._K * eng._adj)))
        np.testing.assert_allclose(eng._K * lam_anorm, lam_effective, rtol=1e-12, atol=1e-14)

    def test_default_K_identity_preserves_prescaled_callers(self) -> None:
        # Callers passing an already-effective matrix (early_stopping / ricci_flow)
        # rely on the K=1.0 default leaving dynamics unchanged.
        a = self._symmetric_topology(5, seed=6)
        effective = 2.5 * a
        theta = np.linspace(0.2, 1.7, 5)
        omega = np.zeros(5)
        np.testing.assert_array_equal(
            _dtheta_dt(theta, omega, effective),
            _dtheta_dt(theta, omega, effective, K=1.0),
        )

    def test_golden_equivalence_output_preserved(self) -> None:
        # Output-preserving: refactoring K into the RHS keeps the canonical
        # trajectory invariant within float reassociation noise (~1e-13).
        # Golden constants captured from pre-refactor origin/main engine.
        g = run_simulation(KuramotoConfig(N=12, K=2.5, dt=0.01, steps=300, seed=7))
        np.testing.assert_allclose(g.summary["final_R"], 0.9623362223, rtol=0.0, atol=1e-9)
        a = self._symmetric_topology(8, seed=3)
        e = run_simulation(
            KuramotoConfig(
                N=8, K=1.7, dt=0.01, steps=250, seed=4,
                adjacency=a, adjacency_kind="normalized_topology",
            )
        )
        np.testing.assert_allclose(e.summary["final_R"], 0.9852126793, rtol=0.0, atol=1e-9)
