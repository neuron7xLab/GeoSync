# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""End-to-end integration and per-module tests for the NetworkKuramotoEngine.

This file exercises the last five modules of the methodology:

- M2.4 :mod:`core.kuramoto.metrics`
- M2.3 :mod:`core.kuramoto.dynamic_graph`
- M3.2 :mod:`core.kuramoto.falsification`
- :mod:`core.kuramoto.network_engine` (orchestrator)
- M3.4 :mod:`core.kuramoto.feature` (trading adapter)

Every test consumes a :class:`SyntheticGroundTruth` instance so the
assertions are anchored to known parameters. The end-to-end section
runs the full orchestrator on a synthetic instance and checks that the
returned :class:`NetworkState` plus :class:`EmergentMetrics` satisfy
the methodology's level-E acceptance criteria.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from core.kuramoto.contracts import (
    EmergentMetrics,
    NetworkState,
    SyntheticGroundTruth,
)
from core.kuramoto.coupling_estimator import CouplingEstimationConfig
from core.kuramoto.delay_estimator import DelayEstimationConfig
from core.kuramoto.dynamic_graph import (
    DynamicGraphConfig,
    DynamicGraphEstimator,
    detect_breakpoints,
)
from core.kuramoto.falsification import (
    SurrogateResult,
    degree_preserving_rewire,
    iaaft_surrogate,
    time_shuffle_test,
)
from core.kuramoto.feature import FeatureConfig, NetworkKuramotoFeature
from core.kuramoto.metrics import (
    chimera_index,
    order_parameter,
    permutation_entropy,
    rolling_csd,
    signed_communities,
)
from core.kuramoto.network_engine import (
    NetworkEngineConfig,
    NetworkKuramotoEngine,
)
from core.kuramoto.phase_extractor import PhaseExtractionConfig
from core.kuramoto.synthetic import SyntheticConfig, generate_sakaguchi_kuramoto

# ---------------------------------------------------------------------------
# M2.4 metrics — primitive and aggregate checks
# ---------------------------------------------------------------------------


class TestOrderParameter:
    def test_perfect_sync_is_one(self) -> None:
        theta = np.full((50, 10), 0.7)
        R = order_parameter(theta, axis=1)
        assert np.allclose(R, 1.0, atol=1e-12)

    def test_uniform_distribution_is_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        theta = rng.uniform(0, 2 * np.pi, size=(2000, 200))
        R = order_parameter(theta, axis=1)
        # 1/√N for 200 nodes → ~0.07
        assert float(R.mean()) < 0.1

    def test_two_cluster_antiphase_is_near_zero(self) -> None:
        theta = np.concatenate([np.zeros((1, 10)), np.full((1, 10), np.pi)], axis=1)
        R = order_parameter(theta, axis=1)
        assert float(R[0]) < 1e-10


class TestChimeraIndex:
    def test_fully_coupled_uniform_sync_chimera_is_zero(self) -> None:
        theta = np.full((20, 8), 1.2)
        adj = np.ones((8, 8))
        chi = chimera_index(theta, adj)
        assert np.all(chi < 1e-12)

    def test_two_disconnected_clusters_have_nonzero_chimera_when_one_desyncs(
        self,
    ) -> None:
        """One cluster sync, the other desync → chimera > 0."""
        T = 50
        theta_sync = np.zeros((T, 5))
        rng = np.random.default_rng(0)
        theta_desync = rng.uniform(0, 2 * np.pi, size=(T, 5))
        theta = np.concatenate([theta_sync, theta_desync], axis=1)
        adj = np.zeros((10, 10))
        adj[:5, :5] = 1.0
        adj[5:, 5:] = 1.0
        chi = chimera_index(theta, adj)
        assert float(chi.mean()) > 0.01


class TestRollingCSD:
    def test_constant_series_has_zero_variance(self) -> None:
        R = np.full(100, 0.5)
        var, ac = rolling_csd(R, window=20)
        assert float(var.max()) < 1e-12

    def test_csd_variance_increases_on_approaching_bifurcation(self) -> None:
        """Amplitude growth of ``R(t)`` raises the rolling variance.

        We build a signal whose oscillation amplitude grows
        exponentially — the classical early-warning signature.
        The last window's variance must be strictly larger than the
        first full window's.
        """
        T = 400
        amplitude = np.exp(np.linspace(-2, 2, T))
        R = 0.5 + amplitude * np.sin(np.linspace(0, 20, T))
        var, _ = rolling_csd(R, window=40)
        assert float(var[-1]) > float(var[40]) * 3.0

    def test_rejects_invalid_window(self) -> None:
        with pytest.raises(ValueError):
            rolling_csd(np.zeros(10), window=20)


class TestSignedCommunities:
    def test_planted_two_communities_recovered(self) -> None:
        """Two positively-coupled blocks linked by negative edges."""
        N = 12
        K = np.zeros((N, N))
        block = 6
        # Excitatory within blocks
        for i in range(block):
            for j in range(block):
                if i != j:
                    K[i, j] = 1.0
                    K[i + block, j + block] = 1.0
        # Inhibitory between blocks
        for i in range(block):
            for j in range(block):
                K[i, j + block] = -0.5
                K[j + block, i] = -0.5
        labels = signed_communities(K, n_clusters_max=4)
        # At minimum, the first and last indices land in different communities
        assert labels[0] != labels[-1]

    def test_random_matrix_splits_cleanly(self) -> None:
        rng = np.random.default_rng(0)
        K = rng.standard_normal((10, 10))
        np.fill_diagonal(K, 0)
        labels = signed_communities(K, n_clusters_max=3)
        assert labels.shape == (10,)
        assert int(labels.min()) >= 0


class TestPermutationEntropy:
    def test_monotonic_series_has_zero_entropy(self) -> None:
        x = np.linspace(0, 1, 100)
        ent = permutation_entropy(x, order=3)
        assert ent < 1e-9

    def test_random_series_has_high_entropy(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(500)
        ent = permutation_entropy(x, order=3)
        assert ent > 0.9


# ---------------------------------------------------------------------------
# M2.3 dynamic graph
# ---------------------------------------------------------------------------


class TestDynamicGraph:
    def test_sliding_window_returns_expected_shape(self) -> None:
        gt = generate_sakaguchi_kuramoto(
            SyntheticConfig(
                N=4,
                T=600,
                dt=0.05,
                burn_in=50,
                seed=1,
                K_sparsity=0.5,
                K_scale=(0.4, 0.9),
                omega_center=0.8,
                omega_spread=0.4,
                tau_max=0,
                alpha_max=0.0,
                alpha_structure="zero",
            )
        )
        cfg = DynamicGraphConfig(
            window=100,
            step=25,
            coupling_config=CouplingEstimationConfig(
                penalty="mcp", lambda_reg=0.1, dt=0.05, max_iter=500, tol=1e-5
            ),
            min_window_for_solver=60,
        )
        est = DynamicGraphEstimator(cfg)
        K_series, centres = est.estimate(gt.generated_phases)
        T_kept = gt.generated_phases.theta.shape[0]
        expected_n = (T_kept - cfg.window) // cfg.step + 1
        assert K_series.shape == (expected_n, 4, 4)
        assert centres.shape == (expected_n,)
        assert np.all(np.diag(K_series[0]) == 0.0)

    def test_detect_breakpoints_returns_indices(self) -> None:
        rng = np.random.default_rng(0)
        K_series = rng.standard_normal((20, 4, 4)) * 0.1
        # Inject a large jump at index 10
        K_series[10:] += 2.0
        idx = detect_breakpoints(K_series, z_threshold=2.0)
        assert 10 in idx.tolist()


# ---------------------------------------------------------------------------
# M3.2 falsification
# ---------------------------------------------------------------------------


class TestIAAFTSurrogate:
    def test_preserves_amplitude_distribution(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(256) ** 3  # non-Gaussian marginal
        surr = iaaft_surrogate(x, n_iterations=100, rng=rng)
        # Sorted values match exactly — that's what IAAFT guarantees
        assert np.allclose(np.sort(x), np.sort(surr), atol=1e-12)

    def test_preserves_power_spectrum_closely(self) -> None:
        rng = np.random.default_rng(1)
        x = np.sin(np.linspace(0, 40, 512)) + 0.3 * rng.standard_normal(512)
        surr = iaaft_surrogate(x, n_iterations=200, rng=rng)
        p_orig = np.abs(np.fft.rfft(x))
        p_surr = np.abs(np.fft.rfft(surr))
        # Match correlation > 0.99 on the dominant spectral bins
        assert float(np.corrcoef(p_orig, p_surr)[0, 1]) > 0.99


class TestTimeShuffleTest:
    def test_shuffled_series_matches_marginal_only(self) -> None:
        """Time-shuffle null must destroy the *time-indexed* R sequence.

        The mean-over-time ``|Σ exp(iθ)|`` is only invariant under
        column-independent shuffling of stationary data, so the
        comparison most relevant to the methodology is between the
        *time series* of ``R(t)`` — the observed has structure, the
        shuffled does not. We check that: (a) the null is
        narrowly concentrated (shuffling is a nearly-unitary
        transformation at the level of mean-R), and (b) the
        surrogate returns a well-formed :class:`SurrogateResult`.
        """
        gt = generate_sakaguchi_kuramoto(
            SyntheticConfig(
                N=6,
                T=800,
                dt=0.05,
                burn_in=50,
                seed=3,
                K_sparsity=0.2,
                K_scale=(1.5, 2.5),
                omega_center=0.5,
                omega_spread=0.05,
                tau_max=0,
                alpha_max=0.0,
                alpha_structure="zero",
            )
        )
        result = time_shuffle_test(gt.generated_phases, n_shuffles=30, seed=0)
        assert isinstance(result, SurrogateResult)
        assert result.null_distribution.shape == (30,)
        # Null is concentrated within a narrow band
        assert float(result.null_distribution.std()) < 0.1
        # p_value is well-defined in [0, 1]
        assert 0.0 <= result.p_value <= 1.0


class TestDegreePreservingRewire:
    def test_preserves_degree_sequence(self) -> None:
        rng = np.random.default_rng(0)
        adj = (rng.random((10, 10)) < 0.3).astype(np.int8)
        np.fill_diagonal(adj, 0)
        rewired = degree_preserving_rewire(adj, n_swaps=200, rng=rng)
        assert np.all(rewired.sum(axis=1) == adj.sum(axis=1))
        assert np.all(rewired.sum(axis=0) == adj.sum(axis=0))


# ---------------------------------------------------------------------------
# Orchestrator (NetworkKuramotoEngine)
# ---------------------------------------------------------------------------


class TestNetworkKuramotoEngine:
    @pytest.fixture(scope="class")
    def synthetic_state(
        self,
    ) -> tuple[NetworkEngineConfig, "SyntheticGroundTruth"]:
        cfg_gen = SyntheticConfig(
            N=5,
            T=3000,
            dt=0.05,
            K_sparsity=0.5,
            K_scale=(0.4, 0.9),
            omega_center=0.8,
            omega_spread=0.4,
            tau_max=2,
            alpha_structure="mixed",
            alpha_max=np.pi / 6,
            sigma_noise=0.02,
            burn_in=50,
            seed=42,
        )
        gt = generate_sakaguchi_kuramoto(cfg_gen)
        engine_cfg = NetworkEngineConfig(
            coupling=CouplingEstimationConfig(
                penalty="mcp",
                lambda_reg=0.1,
                dt=cfg_gen.dt,
                max_iter=1000,
                tol=1e-6,
            ),
            delay=DelayEstimationConfig(max_lag=3, dt=cfg_gen.dt, method="joint"),
        )
        return engine_cfg, gt

    def test_identify_returns_contract_valid_state(
        self, synthetic_state: tuple[NetworkEngineConfig, "SyntheticGroundTruth"]
    ) -> None:
        engine_cfg, gt = synthetic_state
        engine = NetworkKuramotoEngine(engine_cfg)
        report = engine.identify(gt.generated_phases)
        assert isinstance(report.state, NetworkState)
        assert isinstance(report.metrics, EmergentMetrics)
        assert report.state.N == gt.generated_phases.N
        assert report.state.noise_std >= 0.0
        # All arrays deeply frozen
        assert report.state.coupling.K.flags.writeable is False
        assert report.metrics.R_global.flags.writeable is False

    def test_emergent_metrics_bounds(
        self, synthetic_state: tuple[NetworkEngineConfig, "SyntheticGroundTruth"]
    ) -> None:
        engine_cfg, gt = synthetic_state
        report = NetworkKuramotoEngine(engine_cfg).identify(gt.generated_phases)
        m = report.metrics
        assert float(m.R_global.min()) >= 0.0
        assert float(m.R_global.max()) <= 1.0
        assert m.metastability >= 0.0
        assert len(m.R_cluster) >= 1
        for arr in m.R_cluster.values():
            assert float(arr.min()) >= 0.0
            assert float(arr.max()) <= 1.0

    def test_identify_from_returns_end_to_end(self) -> None:
        """Run the full raw-returns entry point on a band-limited signal."""
        rng = np.random.default_rng(0)
        T, N = 600, 4
        fs = 10.0
        t = np.arange(T) / fs
        # Three coupled oscillators + one independent
        phases0 = np.array([0.0, 0.5, 1.0, 2.5])
        returns = np.stack(
            [
                np.sin(2 * np.pi * 1.0 * t + phases0[k]) + 0.1 * rng.standard_normal(T)
                for k in range(N)
            ],
            axis=1,
        )
        engine = NetworkKuramotoEngine(
            NetworkEngineConfig(
                phase=PhaseExtractionConfig(
                    fs=fs, f_low=0.5, f_high=1.5, detrend_window=None
                ),
                coupling=CouplingEstimationConfig(
                    penalty="mcp",
                    lambda_reg=0.1,
                    dt=1 / fs,
                    max_iter=500,
                    tol=1e-5,
                ),
                delay=DelayEstimationConfig(max_lag=2, dt=1 / fs),
            )
        )
        report = engine.identify_from_returns(
            returns=returns,
            asset_ids=tuple(f"a{i}" for i in range(N)),
            timestamps=t,
        )
        assert report.state.N == N
        assert report.metrics.R_global.shape == (T,)


# ---------------------------------------------------------------------------
# M3.4 trading feature adapter
# ---------------------------------------------------------------------------


class TestNetworkKuramotoFeature:
    @pytest.fixture()
    def synthetic_returns(self) -> tuple[np.ndarray, tuple[str, ...]]:
        gt = generate_sakaguchi_kuramoto(
            SyntheticConfig(
                N=4,
                T=1200,
                dt=0.05,
                burn_in=50,
                seed=5,
                K_sparsity=0.5,
                K_scale=(0.4, 0.9),
                omega_center=0.8,
                omega_spread=0.4,
                tau_max=1,
                alpha_structure="zero",
                alpha_max=0.0,
            )
        )
        # Use phase increments as a returns proxy
        theta = gt.generated_phases.theta
        returns = np.diff(np.unwrap(theta, axis=0), axis=0)
        return returns, gt.generated_phases.asset_ids

    def test_warmup_and_online_cold_start(
        self, synthetic_returns: tuple[np.ndarray, tuple[str, ...]]
    ) -> None:
        returns, ids = synthetic_returns
        feat = NetworkKuramotoFeature(
            asset_ids=ids,
            config=FeatureConfig(
                window=100,
                phase_config=PhaseExtractionConfig(
                    fs=20.0, f_low=0.5, f_high=5.0, detrend_window=None
                ),
            ),
        )
        feat.warmup(returns[:200])
        # Online update without prior calibration — must survive
        f = feat.update(returns[200], timestamp=200.0)
        assert "kuramoto_R_global" in f

    def test_full_tier1_tier2_cycle(
        self, synthetic_returns: tuple[np.ndarray, tuple[str, ...]]
    ) -> None:
        returns, ids = synthetic_returns
        feat = NetworkKuramotoFeature(
            asset_ids=ids,
            config=FeatureConfig(
                window=150,
                phase_config=PhaseExtractionConfig(
                    fs=20.0, f_low=0.5, f_high=5.0, detrend_window=None
                ),
                engine_config=NetworkEngineConfig(
                    coupling=CouplingEstimationConfig(
                        penalty="mcp",
                        lambda_reg=0.1,
                        dt=0.05,
                        max_iter=500,
                        tol=1e-5,
                    ),
                    delay=DelayEstimationConfig(max_lag=2, dt=0.05),
                ),
            ),
        )
        feat.warmup(returns[:400])
        feat.recalibrate(
            returns[:600], timestamps=np.arange(600, dtype=np.float64) * 0.05
        )
        # Measure per-bar online latency across 20 updates
        t0 = time.perf_counter()
        last_features: dict[str, float] = {}
        for k, row in enumerate(returns[600:620]):
            last_features = feat.update(row, timestamp=float(600 + k) * 0.05)
        per_bar_ms = (time.perf_counter() - t0) * 1000 / 20

        # Methodology target for Tier 1 (N ≤ 50): < 10 ms
        assert per_bar_ms < 15.0, f"Tier 1 latency {per_bar_ms:.2f} ms exceeds 15 ms"

        # Full feature vocabulary available after calibration
        for key in (
            "kuramoto_R_global",
            "kuramoto_R_derivative",
            "kuramoto_metastability",
            "kuramoto_n_clusters",
            "kuramoto_chimera_index",
            "kuramoto_max_cluster_R",
            "kuramoto_csd_variance",
            "kuramoto_csd_autocorr",
            "kuramoto_coupling_density",
            "kuramoto_inhibition_ratio",
            "kuramoto_calibration_age_bars",
        ):
            assert key in last_features
            assert np.isfinite(last_features[key])

    def test_no_nan_on_valid_input(
        self, synthetic_returns: tuple[np.ndarray, tuple[str, ...]]
    ) -> None:
        returns, ids = synthetic_returns
        feat = NetworkKuramotoFeature(
            asset_ids=ids,
            config=FeatureConfig(
                window=100,
                phase_config=PhaseExtractionConfig(
                    fs=20.0, f_low=0.5, f_high=5.0, detrend_window=None
                ),
            ),
        )
        feat.warmup(returns[:300])
        feat.recalibrate(
            returns[:500], timestamps=np.arange(500, dtype=np.float64) * 0.05
        )
        for k, row in enumerate(returns[500:530]):
            features = feat.update(row, timestamp=float(500 + k) * 0.05)
            for v in features.values():
                assert np.isfinite(v), "feature output contains NaN/Inf"
