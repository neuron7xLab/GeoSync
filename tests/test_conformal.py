# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Unit tests for the conformal calibration module."""

from __future__ import annotations

import numpy as np

from geosync_hpc.conformal import ConformalCQR


def test_cqr_qhat_nonnegative() -> None:
    cqr = ConformalCQR(alpha=0.1, decay=0.01, window=50)
    lower = np.array([-0.01] * 100)
    upper = np.array([0.01] * 100)
    targets = np.concatenate([np.random.normal(0, 0.005, 95), np.array([0.05] * 5)])
    cqr.fit_calibrate(lower, upper, targets)
    assert cqr.qhat is not None and cqr.qhat >= 0.0


def test_cqr_reset_restores_calibrated_baseline_after_online_updates() -> None:
    cqr = ConformalCQR(alpha=0.1, decay=0.01, window=100, online_window=10)
    lower = np.array([-0.01] * 40)
    upper = np.array([0.01] * 40)
    targets = np.random.normal(0.0, 0.005, size=40)
    cqr.fit_calibrate(lower, upper, targets)

    baseline_qhat = cqr.qhat
    baseline_resid = tuple(cqr._resid)

    cqr.update_online(-0.01, 0.01, 0.05)
    cqr.update_online(-0.01, 0.01, -0.05)
    assert cqr.qhat != baseline_qhat or tuple(cqr._resid) != baseline_resid

    cqr.reset()
    assert cqr.alpha == cqr.alpha0
    assert cqr.qhat == baseline_qhat
    assert tuple(cqr._resid) == baseline_resid


def test_cqr_empty_calibration_replaces_stale_baseline() -> None:
    cqr = ConformalCQR(alpha=0.1, decay=0.01, window=100, online_window=10)
    cqr.fit_calibrate(np.array([-0.01, -0.01]), np.array([0.01, 0.01]), np.array([0.0, 0.03]))
    assert cqr.qhat is not None and cqr.qhat >= 0.0

    cqr.fit_calibrate(np.array([]), np.array([]), np.array([]))
    cqr.update_online(-0.01, 0.01, 0.25)
    cqr.reset()

    assert cqr.qhat == 0.0
    assert tuple(cqr._resid) == tuple()
