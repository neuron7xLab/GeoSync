# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Tests for :mod:`research.systemic_risk.network_fitting`."""

from __future__ import annotations

import numpy as np
import pytest

from research.systemic_risk.network_fitting import (
    compare_power_law_vs_exponential,
    fit_barabasi_albert,
    fit_exponential,
    fit_power_law,
)


def _draw_power_law(n: int, alpha: float, k_min: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    raw = (k_min - 0.5) * (1.0 - u) ** (1.0 / (1.0 - alpha)) + 0.5
    return np.maximum(np.rint(raw).astype(np.int64), k_min)


class TestFitPowerLaw:
    def test_recovers_alpha_on_synthetic(self) -> None:
        # Statistical: ensemble of 30 fits on synthetic α=2.5 samples.
        # The continuous-approximation MLE has known small-sample bias
        # of order 1/n for discrete data (Clauset 2009 §B). At n=2000,
        # k_min=6 the bias is ≤ 0.20 in expectation — used here as the
        # tolerance bound. Standard error of the ensemble mean is
        # σ_α/√30 ≈ 0.05.
        true_alpha = 2.5
        estimates = []
        for seed in range(30):
            sample = _draw_power_law(2000, true_alpha, 6, seed)
            estimates.append(fit_power_law(sample, k_min=6).alpha)
        mean_alpha = float(np.mean(estimates))
        assert abs(mean_alpha - true_alpha) < 0.20, (
            f"MLE-BIAS VIOLATED: mean(α̂)={mean_alpha:.4f} vs α={true_alpha} "
            f"over 30 samples of size 2000, k_min=6 (tolerance 0.20)"
        )

    def test_alpha_se_decreases_with_n(self) -> None:
        small = _draw_power_law(200, 2.5, 6, 0)
        large = _draw_power_law(2000, 2.5, 6, 0)
        small_se = fit_power_law(small, k_min=6).alpha_se
        large_se = fit_power_law(large, k_min=6).alpha_se
        assert large_se < small_se, (
            f"SE-MONOTONE VIOLATED: SE(n=2000)={large_se:.4f} >= "
            f"SE(n=200)={small_se:.4f}; expected SE ∝ 1/√n at α=2.5"
        )

    def test_ks_p_value_bracket(self) -> None:
        sample = _draw_power_law(500, 2.5, 6, 7)
        fit = fit_power_law(sample, k_min=6, n_bootstrap=200, seed=11)
        assert fit.ks_p_value is not None
        assert 0.0 < fit.ks_p_value <= 1.0

    def test_invalid_input_rejected(self) -> None:
        with pytest.raises(ValueError):
            fit_power_law(np.array([1, 2], dtype=np.int64))  # n<4
        with pytest.raises(ValueError):
            fit_power_law(np.array([-1, 2, 3, 4], dtype=np.int64))


class TestFitExponential:
    def test_recovers_lambda(self) -> None:
        rng = np.random.default_rng(3)
        true_lambda = 0.4
        # Discrete shifted exponential: k = k_min + Geom(p=1-exp(-λ)).
        p = 1.0 - np.exp(-true_lambda)
        k_min = 1
        sample = (k_min + rng.geometric(p, size=2000) - 1).astype(np.int64)
        fit = fit_exponential(sample, k_min=k_min)
        assert abs(fit.lambda_ - true_lambda) < 0.1, (
            f"EXP-BIAS VIOLATED: λ̂={fit.lambda_:.4f} vs λ={true_lambda} "
            f"on synthetic n=2000, k_min={k_min}"
        )


class TestCompare:
    def test_power_law_preferred_on_power_law_sample(self) -> None:
        sample = _draw_power_law(1500, 2.4, 4, 0)
        cmp = compare_power_law_vs_exponential(sample, k_min=4)
        assert cmp.preferred == "power_law"
        assert cmp.aic_power_law < cmp.aic_exponential
        assert cmp.aic_delta > 0

    def test_exponential_preferred_on_exponential_sample(self) -> None:
        rng = np.random.default_rng(5)
        sample = (1 + rng.geometric(0.3, size=1500) - 1).astype(np.int64)
        cmp = compare_power_law_vs_exponential(sample, k_min=1)
        assert cmp.preferred == "exponential"


class TestFitBarabasiAlbert:
    def test_returns_positive_m(self) -> None:
        sample = _draw_power_law(1000, 2.5, 4, 0)
        m, fit = fit_barabasi_albert(sample)
        assert m >= 1
        assert 1.5 < fit.alpha < 4.5

    def test_seed_determinism(self) -> None:
        # Same input → same fit (no hidden randomness when n_bootstrap=0).
        sample = _draw_power_law(500, 2.5, 4, 0)
        a, fit_a = fit_barabasi_albert(sample)
        b, fit_b = fit_barabasi_albert(sample)
        assert a == b
        assert fit_a.alpha == fit_b.alpha
