# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Invariant and falsifier tests for the expectile-ensemble dopamine learner.

The estimator is grounded in Dabney et al. (2020), Nature 577:671. These tests
encode the algebraic identities, the canonical TD(0) reduction, and the strong
falsifier: the online learner must converge to the offline analytic expectile.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from geosync.core.neuro.dopamine.expectile_value import (
    ExpectileChannel,
    ExpectileEnsembleValue,
    expectile_of_samples,
)


# ----------------------------------------------------------------------
# INV-EXP1 (algebraic): tau = alpha_pos / (alpha_pos + alpha_neg) exactly
# ----------------------------------------------------------------------
@pytest.mark.parametrize("tau", [0.05, 0.25, 0.5, 0.75, 0.95])
def test_tau_recovered_from_asymmetric_rates(tau: float) -> None:
    ch = ExpectileChannel(tau=tau, learning_rate=0.1)
    assert abs(ch.recovered_tau() - tau) < 1e-12, (
        f"INV-EXP1 VIOLATED: recovered tau {ch.recovered_tau()} != {tau}; "
        f"alpha_pos={ch.alpha_pos}, alpha_neg={ch.alpha_neg}. "
        f"Dabney 2020 identity tau=alpha+/(alpha+ + alpha-) broken."
    )


# ----------------------------------------------------------------------
# INV-EXP2 (algebraic): tau=0.5 expectile IS the mean
# ----------------------------------------------------------------------
def test_half_expectile_equals_mean() -> None:
    rng = np.random.default_rng(0)
    samples = rng.normal(2.0, 3.0, size=5000)
    assert abs(expectile_of_samples(samples, 0.5) - float(samples.mean())) < 1e-9


# ----------------------------------------------------------------------
# INV-EXP3 (asymptotic): symmetric channel recovers canonical TD(0) mean
# ----------------------------------------------------------------------
def test_symmetric_channel_recovers_canonical_mean() -> None:
    ens = ExpectileEnsembleValue.symmetric(n_channels=1, learning_rate=0.02)
    rng = np.random.default_rng(1)
    samples = rng.normal(1.0, 0.5, size=20000)
    for s in samples:
        ens.observe_return(float(s))
    # The tau=0.5 channel converges to the sample mean: the canonical anchor
    # is a special case of the ensemble, not a thing it replaces.
    assert abs(ens.canonical_value() - float(samples.mean())) < 0.05


def test_td_target_channel_tracks_scalar_anchor() -> None:
    # With a fixed reward and next_value, the bootstrap target is constant and
    # the tau=0.5 channel must converge on it: reward + gamma * next_value.
    ens = ExpectileEnsembleValue.symmetric(n_channels=1, learning_rate=0.05)
    reward, next_value, gamma = 0.7, 0.4, 0.9
    for _ in range(3000):
        ens.observe_td_target(reward, next_value, gamma)
    assert abs(ens.canonical_value() - (reward + gamma * next_value)) < 1e-6


# ----------------------------------------------------------------------
# INV-EXP4 (monotonic): expectile estimates are non-decreasing in tau
# ----------------------------------------------------------------------
def test_expectile_curve_is_monotone() -> None:
    ens = ExpectileEnsembleValue.spread(n_channels=9, learning_rate=0.02)
    rng = np.random.default_rng(2)
    # Skewed target stream so the curve has genuine spread.
    samples = rng.gamma(shape=2.0, scale=1.5, size=20000)
    for s in samples:
        ens.observe_return(float(s))
    taus, values = ens.expectile_curve()
    assert np.all(np.diff(taus) > 0), "tau levels must be strictly ascending"
    diffs = np.diff(values)
    assert np.all(diffs >= -1e-9), (
        f"INV-EXP4 VIOLATED: expectile curve not monotone in tau; "
        f"min adjacent diff={float(diffs.min())}. "
        f"tau_a < tau_b must imply V_a <= V_b at convergence."
    )
    assert ens.dispersion() > 0.0


# ----------------------------------------------------------------------
# INV-EXP5 (universal): values stay within the convex hull of targets
# ----------------------------------------------------------------------
def test_values_bounded_by_target_hull() -> None:
    ens = ExpectileEnsembleValue.spread(n_channels=7, learning_rate=0.3)
    lo, hi = -2.0, 5.0
    rng = np.random.default_rng(3)
    samples = rng.uniform(lo, hi, size=5000)
    for s in samples:
        ens.observe_return(float(s))
        vals = ens.values
        assert np.all(vals >= lo - 1e-9) and np.all(vals <= hi + 1e-9), (
            f"INV-EXP5 VIOLATED: value left [{lo}, {hi}] hull: {vals}"
        )


# ----------------------------------------------------------------------
# FALSIFIER part A (algebraic, exact): the asymmetric-weighted residual is
# zero AT the analytic expectile. This is the *definition* of expectile
# regression — the fixed point the online update targets. Deterministic teeth.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("tau", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_analytic_expectile_is_a_true_fixed_point(tau: float) -> None:
    rng = np.random.default_rng(int(tau * 100))
    x = rng.normal(0.0, 1.0, size=400)
    e = expectile_of_samples(x, tau)
    # sum_i w_i (x_i - e) == 0, w_i = tau if x_i > e else (1 - tau).
    w = np.where(x > e, tau, 1.0 - tau)
    residual = float(np.sum(w * (x - e)))
    assert abs(residual) < 1e-9, (
        f"FALSIFIER FIRED: tau={tau} weighted residual {residual:.2e} != 0; "
        f"expectile_of_samples is not a true expectile, so the online update "
        f"has no valid target. Revert, do not loosen."
    )


# ----------------------------------------------------------------------
# FALSIFIER part B (dynamics): the online learner converges into the O(sqrt(eta))
# stationary neighbourhood of the analytic expectile. The 0.05 tolerance is the
# genuine constant-rate fluctuation band, NOT a slackened physics bound: it
# still cleanly separates an expectile from a quantile (for tau=0.9 on N(0,1)
# the quantile is ~1.28 vs expectile ~0.94, a 0.35 gap). tau=0.5 is excluded —
# it is the canonical mean, tested exactly above, and does not discriminate
# expectile from quantile. Seeds are fixed, so this is deterministic.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("tau", [0.1, 0.3, 0.7, 0.9])
def test_online_learner_converges_to_analytic_expectile(tau: float) -> None:
    rng = np.random.default_rng(int(tau * 100))
    targets = rng.normal(0.0, 1.0, size=400)
    truth = expectile_of_samples(targets, tau)

    ens = ExpectileEnsembleValue.from_levels([tau], learning_rate=0.02)
    polyak: list[float] = []
    for p in range(300):
        for t in targets:
            ens.observe_return(float(t))
        if p >= 200:  # Polyak-Ruppert averaging over the stationary tail.
            polyak.append(float(ens.values[0]))
    learned = float(np.mean(polyak))
    assert abs(learned - truth) < 0.05, (
        f"FALSIFIER FIRED: online tau={tau} expectile {learned:.5f} != "
        f"analytic {truth:.5f}. Estimator is not expectile regression."
    )


def test_risk_adjusted_value_orders_by_tau() -> None:
    ens = ExpectileEnsembleValue.spread(n_channels=9, learning_rate=0.02)
    rng = np.random.default_rng(7)
    samples = rng.normal(0.0, 1.0, size=15000)
    for s in samples:
        ens.observe_return(float(s))
    risk_averse = ens.risk_adjusted_value(0.1)
    neutral = ens.canonical_value()
    risk_seeking = ens.risk_adjusted_value(0.9)
    assert risk_averse < neutral < risk_seeking


# ----------------------------------------------------------------------
# Fail-closed contract guards
# ----------------------------------------------------------------------
@pytest.mark.parametrize("bad_tau", [0.0, 1.0, -0.1, 1.5, math.nan, math.inf])
def test_channel_rejects_bad_tau(bad_tau: float) -> None:
    with pytest.raises(ValueError):
        ExpectileChannel(tau=bad_tau, learning_rate=0.1)


@pytest.mark.parametrize("bad_eta", [0.0, -0.1, 1.5, math.nan])
def test_channel_rejects_bad_learning_rate(bad_eta: float) -> None:
    with pytest.raises(ValueError):
        ExpectileChannel(tau=0.5, learning_rate=bad_eta)


def test_ensemble_rejects_empty_and_duplicate_levels() -> None:
    with pytest.raises(ValueError):
        ExpectileEnsembleValue([])
    with pytest.raises(ValueError):
        ExpectileEnsembleValue.from_levels([0.5, 0.5])


def test_observe_rejects_nonfinite_and_bad_gamma() -> None:
    ens = ExpectileEnsembleValue.symmetric()
    with pytest.raises(ValueError):
        ens.observe_return(math.nan)
    with pytest.raises(ValueError):
        ens.observe_td_target(1.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        ens.observe_td_target(1.0, 1.0, 1.5)


def test_expectile_solver_rejects_empty_and_bad_tau() -> None:
    with pytest.raises(ValueError):
        expectile_of_samples([], 0.5)
    with pytest.raises(ValueError):
        expectile_of_samples([1.0, 2.0], 0.0)
    with pytest.raises(ValueError):
        expectile_of_samples([1.0, math.inf], 0.5)


def test_determinism_bit_identical() -> None:
    rng = np.random.default_rng(11)
    samples = rng.normal(0.0, 1.0, size=2000)

    def run() -> np.ndarray:
        ens = ExpectileEnsembleValue.spread(n_channels=5, learning_rate=0.03)
        for s in samples:
            ens.observe_return(float(s))
        return ens.values

    a, b = run(), run()
    assert np.array_equal(a, b), "estimator must be deterministic (bit-identical)"
