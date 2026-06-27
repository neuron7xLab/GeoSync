# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Formula checks for the streamed second-order swing-energy evaluator."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
import pytest

from core.kuramoto.second_order import SecondOrderKuramotoEngine


EnergyFn = Callable[
    [
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    NDArray[np.float64],
]


def _energy_fn() -> EnergyFn:
    return getattr(SecondOrderKuramotoEngine, "_swing_energy_trajectory")


def test_streamed_energy_formula_two_node_manual_values() -> None:
    theta = np.array(
        [[0.0, np.pi / 2.0], [0.25, -0.25]],
        dtype=np.float64,
    )
    vel = np.array([[2.0, -1.0], [0.5, 1.5]], dtype=np.float64)
    mass = np.array([2.0, 4.0], dtype=np.float64)
    omega = np.array([0.3, -0.2], dtype=np.float64)
    adj = np.array([[0.0, 1.5], [1.5, 0.0]], dtype=np.float64)

    got = _energy_fn()(theta, vel, mass, omega, adj)
    expected = np.array(
        [6.0 + 0.1 * np.pi, 4.625 - 1.5 * np.cos(0.5)],
        dtype=np.float64,
    )

    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_streamed_energy_rejects_mismatched_trajectory_shape() -> None:
    theta = np.zeros((2, 2), dtype=np.float64)
    vel = np.zeros((2, 3), dtype=np.float64)
    mass = np.ones(2, dtype=np.float64)
    omega = np.zeros(2, dtype=np.float64)
    adj = np.zeros((2, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="equal shape"):
        _energy_fn()(theta, vel, mass, omega, adj)
