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
        result = detector.measure_proximity(N=5, seed=42)
        assert np.all(result.R_forward >= 0)
        assert np.all(result.R_forward <= 1)
        assert np.all(result.R_backward >= 0)
        assert np.all(result.R_backward <= 1)

    def test_hysteresis_non_negative(self, detector):
        result = detector.measure_proximity(N=5, seed=42)
        assert result.hysteresis_width >= 0

    def test_deterministic(self, detector):
        r1 = detector.measure_proximity(N=5, seed=42)
        r2 = detector.measure_proximity(N=5, seed=42)
        np.testing.assert_array_equal(r1.R_forward, r2.R_forward)

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
