# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""T5 — Tsallis entropy risk gate tests."""

import numpy as np
import pytest

from core.physics.tsallis_gate import TsallisGateResult, TsallisRegime, TsallisRiskGate


@pytest.fixture
def gate() -> TsallisRiskGate:
    return TsallisRiskGate(window=30, q_normal=1.35, q_crisis=1.55)


class TestQEstimation:
    """q estimated from kurtosis: q = (5+3κ)/(3+κ)."""

    def test_gaussian_returns_near_5_3(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 10000)
        q = TsallisRiskGate.estimate_q(returns)
        # Gaussian: κ≈0 → q≈5/3≈1.67
        assert 1.4 < q < 2.0

    def test_heavy_tails_higher_q(self):
        rng = np.random.default_rng(42)
        normal = rng.normal(0, 0.01, 1000)
        heavy = rng.standard_t(3, 1000) * 0.01  # t-distribution, heavy tails

        q_normal = TsallisRiskGate.estimate_q(normal)
        q_heavy = TsallisRiskGate.estimate_q(heavy)
        assert q_heavy > q_normal, "Heavy tails should have higher q"

    def test_constant_returns_q_1(self):
        """No variance → q = 1.0 (Gaussian limit)."""
        returns = np.zeros(100)
        q = TsallisRiskGate.estimate_q(returns)
        assert q == 1.0

    def test_insufficient_data(self):
        q = TsallisRiskGate.estimate_q(np.array([0.01, 0.02]))
        assert q == 1.0


class TestRegimeClassification:
    def test_normal_regime(self, gate):
        assert gate.classify_regime(1.2) == TsallisRegime.NORMAL

    def test_elevated_regime(self, gate):
        assert gate.classify_regime(1.45) == TsallisRegime.ELEVATED

    def test_crisis_regime(self, gate):
        assert gate.classify_regime(1.60) == TsallisRegime.CRISIS


class TestPositionMultiplier:
    """f(q) = max(0, 1 - (q - 1.35) / 0.20)."""

    def test_full_position_in_normal(self, gate):
        assert gate.position_multiplier(1.20) == 1.0

    def test_zero_position_in_crisis(self, gate):
        assert gate.position_multiplier(1.55) == 0.0
        assert gate.position_multiplier(2.0) == 0.0

    def test_half_position_midway(self, gate):
        mult = gate.position_multiplier(1.45)
        assert abs(mult - 0.5) < 1e-10

    def test_monotone_decreasing(self, gate):
        q_values = np.linspace(1.0, 2.0, 20)
        mults = [gate.position_multiplier(q) for q in q_values]
        for i in range(1, len(mults)):
            assert mults[i] <= mults[i - 1] + 1e-12


class TestGateEvaluation:
    def test_normal_market_full_position(self, gate):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.005, 100)  # low vol → normal
        result = gate.evaluate(returns)
        assert isinstance(result, TsallisGateResult)
        assert result.position_multiplier > 0

    def test_history_recorded(self, gate):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 60)
        gate.evaluate(returns)
        gate.evaluate(returns)
        assert len(gate.history) == 2

    def test_2d_returns(self, gate):
        """Cross-sectional returns should work."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, (60, 5))
        result = gate.evaluate(returns)
        assert isinstance(result, TsallisGateResult)

    def test_evaluate_prices(self, gate):
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.normal(0, 1, 100))
        result = gate.evaluate_prices(prices)
        assert isinstance(result, TsallisGateResult)


class TestDrawdownReduction:
    """Gate should reduce exposure during crisis-like returns."""

    def test_crisis_returns_reduce_position(self, gate):
        rng = np.random.default_rng(42)
        # Simulate crisis: heavy-tailed returns with high kurtosis
        crisis = rng.standard_t(2.5, 200) * 0.05
        result = gate.evaluate(crisis)
        assert result.position_multiplier < 1.0, "Crisis should reduce position"


class TestInputValidation:
    def test_bad_window(self):
        with pytest.raises(ValueError):
            TsallisRiskGate(window=1)

    def test_bad_q_order(self):
        with pytest.raises(ValueError):
            TsallisRiskGate(q_normal=1.55, q_crisis=1.35)

    def test_insufficient_prices(self, gate):
        with pytest.raises(ValueError):
            gate.evaluate_prices(np.array([100.0]))
