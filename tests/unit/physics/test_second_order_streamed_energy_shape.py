# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Shape-contract checks for streamed second-order swing energy."""

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


def test_streamed_energy_requires_two_dimensional_phase_trajectory() -> None:
    theta = np.zeros(2, dtype=np.float64)
    vel = np.zeros(2, dtype=np.float64)
    mass = np.ones(2, dtype=np.float64)
    omega = np.zeros(2, dtype=np.float64)
    adj = np.zeros((2, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="2-D arrays"):
        _energy_fn()(theta, vel, mass, omega, adj)
