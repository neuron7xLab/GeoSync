# SPDX-License-Identifier: MIT
"""T2 — Explosive Synchronization Proximity tests."""

import numpy as np
import pytest

from core.physics.explosive_sync import (
    ESCircuitBreaker,
    ESProximityResult,
    ExplosiveSyncDetector,
)


@pytest.fixture
def detector() -> ExplosiveSyncDetector:
    return ExplosiveSyncDetector(
        K_range=(0.5, 4.0), n_K_steps=10, kuramoto_steps=100,
        R_threshold=0.5, hysteresis_threshold=0.3,
    )


class TestHysteresisDetection:
    """Forward and backward sweeps should differ for ES-prone networks."""

    def test_returns_valid_result(self, detector):
        result = detector.measure_proximity(N=5, seed=42)
        assert isinstance(result, ESProximityResult)
        assert result.R_forward.shape == (10,)
        assert result.R_backward.shape == (10,)
        assert result.K_values.shape == (10,)
        assert 0 <= result.proximity <= 1

    def test_R_bounded(self, detector):
        """INV-K1: R ∈ [0, 1] on both forward and backward K-sweeps.

        Explosive-sync detection depends on comparing two R trajectories
        (forward K↑ and backward K↓). If either leaves the definitional
        range, hysteresis measurement is meaningless.
        """
        result = detector.measure_proximity(N=5, seed=42)
        r_fwd_min = float(np.min(result.R_forward))
        r_fwd_max = float(np.max(result.R_forward))
        r_bwd_min = float(np.min(result.R_backward))
        r_bwd_max = float(np.max(result.R_backward))

        assert np.all(result.R_forward >= 0), (
            f"INV-K1 VIOLATED: R_forward min = {r_fwd_min:.6f} < 0. "
            f"Expected R ∈ [0, 1] by definition. "
            f"Observed at N=5, seed=42 on forward K-sweep."
        )
        assert np.all(result.R_forward <= 1), (
            f"INV-K1 VIOLATED: R_forward max = {r_fwd_max:.6f} > 1. "
            f"Expected R ≤ 1 from |mean(e^{{iθ}})|. "
            f"Observed at N=5, seed=42 on forward K-sweep."
        )
        assert np.all(result.R_backward >= 0), (
            f"INV-K1 VIOLATED: R_backward min = {r_bwd_min:.6f} < 0. "
            f"Expected R ∈ [0, 1] by definition. "
            f"Observed at N=5, seed=42 on backward K-sweep."
        )
        assert np.all(result.R_backward <= 1), (
            f"INV-K1 VIOLATED: R_backward max = {r_bwd_max:.6f} > 1. "
            f"Expected R ≤ 1 from Cauchy-Schwarz. "
            f"Observed at N=5, seed=42 on backward K-sweep."
        )

    def test_hysteresis_non_negative(self, detector):
        """INV-ES1: K_c^↑ − K_c^↓ ≥ 0 across independent seed realisations.

        A single seed can accidentally produce non-negative width even
        when the detector is broken; to enforce universal-invariant
        semantics we sweep multiple seeds and demand the bound on every
        realisation. A negative width would mean the backward sweep
        synchronises at a higher K than the forward sweep — an acausal
        ordering that contradicts the irreversibility of explosive
        transitions.
        """
        widths: list[float] = []
        for seed in range(5):
            result = detector.measure_proximity(N=5, seed=seed)
            width = float(result.hysteresis_width)
            widths.append(width)
            assert width >= 0, (
                f"INV-ES1 VIOLATED: hysteresis width = {width:.6f} < 0 at seed={seed}. "
                f"Expected K_c^↑ ≥ K_c^↓ for any ES-prone topology. "
                f"Observed at N=5, K_range=(0.5, 4.0), 10 K-steps, 100 kuramoto_steps. "
                f"Physical reasoning: negative width would mean backward sweep "
                f"synchronises at higher K than forward, violating time-arrow."
            )

    def test_deterministic(self, detector):
        """INV-HPC1: repeated measurement with identical seed is bit-identical.

        The ES detector runs two Kuramoto sweeps internally. A determinism
        leak in either sweep (unseeded RNG branch, hash ordering) would
        produce run-to-run drift in hysteresis width and break the
        reproducibility contract of the circuit breaker.
        """
        n_runs = 3
        runs = [detector.measure_proximity(N=5, seed=42).R_forward for _ in range(n_runs)]
        baseline = runs[0]
        for run_idx, other in enumerate(runs[1:], start=1):
            max_diff = float(np.max(np.abs(other - baseline)))
            assert np.array_equal(other, baseline), (
                f"INV-HPC1 VIOLATED: run {run_idx} vs run 0 diff = {max_diff:.3e}. "
                f"Expected bit-identical R_forward under seed=42. "
                f"Observed at N=5, K_range=(0.5, 4.0), 10 K-steps, 100 kuramoto_steps. "
                f"Physical reasoning: seeded ODE + seeded RNG must replay identically."
            )

    def test_with_adjacency(self, detector):
        """Custom adjacency should work."""
        adj = np.array([
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0],
        ], dtype=float)
        result = detector.measure_proximity(adjacency=adj, N=5, seed=42)
        assert isinstance(result, ESProximityResult)


class TestCrisisSignal:
    def test_from_prices(self, detector):
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.normal(0, 1, (80, 5)), axis=0)
        result = detector.crisis_signal(prices, window=30)
        assert isinstance(result, ESProximityResult)
        assert np.isfinite(result.proximity)

    def test_insufficient_data_raises(self, detector):
        with pytest.raises(ValueError):
            detector.crisis_signal(np.ones((5, 3)), window=60)


class TestCircuitBreaker:
    def test_triggers_on_high_proximity(self):
        cb = ESCircuitBreaker(proximity_threshold=0.1, cooldown_steps=3)
        assert not cb.is_triggered
        assert cb.check(0.05) is False
        assert cb.check(0.15) is True
        assert cb.is_triggered
        assert cb.trigger_count == 1

    def test_cooldown(self):
        cb = ESCircuitBreaker(proximity_threshold=0.1, cooldown_steps=2)
        cb.check(0.2)  # trigger
        assert cb.is_triggered
        cb.check(0.01)  # cooldown step 1
        assert cb.is_triggered
        cb.check(0.01)  # cooldown step 2 → released
        assert not cb.is_triggered

    def test_reset(self):
        cb = ESCircuitBreaker(proximity_threshold=0.1, cooldown_steps=5)
        cb.check(0.2)
        assert cb.is_triggered
        cb.reset()
        assert not cb.is_triggered

    def test_multiple_triggers(self):
        cb = ESCircuitBreaker(proximity_threshold=0.1, cooldown_steps=1)
        cb.check(0.2)  # trigger 1
        cb.check(0.01)  # cooldown ends
        cb.check(0.2)  # trigger 2
        assert cb.trigger_count == 2


class TestInputValidation:
    def test_bad_K_range(self):
        with pytest.raises(ValueError):
            ExplosiveSyncDetector(K_range=(5.0, 1.0))

    def test_bad_n_steps(self):
        with pytest.raises(ValueError):
            ExplosiveSyncDetector(n_K_steps=1)

    def test_bad_cb_threshold(self):
        with pytest.raises(ValueError):
            ESCircuitBreaker(proximity_threshold=0.0)
