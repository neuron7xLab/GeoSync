# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Deterministic expectile-ensemble value learner.

This module is a structural research surface only. It provides an online
expectile learner with deterministic replay and finite-output guards. It does
not modify DopamineController or claim evaluated performance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

_EXPECTILE_MAX_ITERS = 256
_EXPECTILE_TOL = 1e-13


def _require_finite(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _require_finite_array(name: str, values: FloatArray) -> FloatArray:
    if not bool(np.all(np.isfinite(values))):
        raise ValueError(f"{name} must be finite")
    return values


def expectile_of_samples(samples: npt.ArrayLike, tau: float) -> float:
    """Return the offline fixed-point expectile of finite samples."""
    tau = _require_finite("tau", float(tau))
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must be in (0, 1), got {tau!r}")
    arr = cast(FloatArray, np.asarray(samples, dtype=np.float64).ravel())
    if arr.size == 0:
        raise ValueError("samples must be non-empty")
    _require_finite_array("samples", arr)

    e = _require_finite("expectile seed", float(np.median(arr)))
    max_abs = _require_finite("sample scale", float(np.max(np.abs(arr))))
    scale = max(1.0, max_abs)

    for _ in range(_EXPECTILE_MAX_ITERS):
        scaled_arr = cast(FloatArray, arr / scale)
        scaled_e = e / scale
        upper = scaled_arr >= scaled_e
        w_upper = tau * np.count_nonzero(upper)
        w_lower = (1.0 - tau) * np.count_nonzero(~upper)
        denom = w_upper + w_lower
        if denom <= 0.0:  # pragma: no cover - guarded by arr.size > 0
            break

        upper_residual = scaled_arr[upper] - scaled_e
        lower_residual = scaled_arr[~upper] - scaled_e
        weighted_residual = (
            tau * upper_residual.sum(dtype=np.float64)
            + (1.0 - tau) * lower_residual.sum(dtype=np.float64)
        )
        weighted_residual = _require_finite(
            "expectile weighted residual",
            float(weighted_residual),
        )
        new_scaled_e = _require_finite(
            "scaled expectile",
            scaled_e + weighted_residual / denom,
        )
        new_e = _require_finite("expectile", new_scaled_e * scale)
        if abs(new_scaled_e - scaled_e) <= _EXPECTILE_TOL * (
            1.0 + abs(scaled_e)
        ):
            e = new_e
            break
        e = new_e

    return _require_finite("expectile", float(e))


@dataclass(frozen=True)
class ExpectileChannel:
    """One channel learning one expectile level."""

    tau: float
    learning_rate: float

    def __post_init__(self) -> None:
        tau = _require_finite("tau", float(self.tau))
        eta = _require_finite("learning_rate", float(self.learning_rate))
        if not 0.0 < tau < 1.0:
            raise ValueError(f"tau must be in (0, 1), got {tau!r}")
        if not 0.0 < eta <= 1.0:
            raise ValueError(f"learning_rate must be in (0, 1], got {eta!r}")
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "learning_rate", eta)

    @property
    def alpha_pos(self) -> float:
        """Learning rate used for positive errors."""
        return self.learning_rate * self.tau

    @property
    def alpha_neg(self) -> float:
        """Learning rate used for negative errors."""
        return self.learning_rate * (1.0 - self.tau)

    def recovered_tau(self) -> float:
        """Recover tau from the asymmetric learning rates."""
        total = _require_finite("alpha_total", self.alpha_pos + self.alpha_neg)
        return _require_finite("recovered_tau", self.alpha_pos / total)


class ExpectileEnsembleValue:
    """Online ensemble of expectile channels."""

    def __init__(
        self,
        channels: list[ExpectileChannel],
        *,
        init_value: float = 0.0,
    ) -> None:
        if not channels:
            raise ValueError("ensemble requires at least one channel")
        init_value = _require_finite("init_value", float(init_value))
        taus = [c.tau for c in channels]
        if len(set(taus)) != len(taus):
            raise ValueError("channel tau levels must be unique")
        order = sorted(range(len(channels)), key=lambda i: channels[i].tau)
        self._channels: tuple[ExpectileChannel, ...] = tuple(
            channels[i] for i in order
        )
        self._values: FloatArray = cast(
            FloatArray,
            np.full(len(channels), init_value, dtype=np.float64),
        )
        _require_finite_array("initial values", self._values)
        self._n_updates: int = 0

    @classmethod
    def symmetric(
        cls,
        *,
        n_channels: int = 1,
        learning_rate: float = 0.05,
    ) -> ExpectileEnsembleValue:
        """Build the single canonical tau=0.5 channel."""
        if n_channels != 1:
            raise ValueError("symmetric ensemble is the single tau=0.5 channel")
        return cls([ExpectileChannel(tau=0.5, learning_rate=learning_rate)])

    @classmethod
    def from_levels(
        cls,
        taus: npt.ArrayLike,
        *,
        learning_rate: float = 0.05,
        init_value: float = 0.0,
    ) -> ExpectileEnsembleValue:
        """Build an ensemble from explicit expectile levels."""
        levels = cast(FloatArray, np.asarray(taus, dtype=np.float64).ravel())
        channels = [
            ExpectileChannel(tau=float(t), learning_rate=learning_rate)
            for t in levels
        ]
        return cls(channels, init_value=init_value)

    @classmethod
    def spread(
        cls,
        n_channels: int = 9,
        *,
        learning_rate: float = 0.05,
        init_value: float = 0.0,
    ) -> ExpectileEnsembleValue:
        """Build evenly spaced levels across (0, 1)."""
        if n_channels < 1:
            raise ValueError("n_channels must be >= 1")
        levels = np.arange(1, n_channels + 1, dtype=np.float64) / (
            n_channels + 1
        )
        return cls.from_levels(levels, learning_rate=learning_rate, init_value=init_value)

    def observe_return(self, target: float) -> FloatArray:
        """Update every channel toward a realised or bootstrapped target."""
        target = _require_finite("target", float(target))
        with np.errstate(over="ignore", invalid="ignore"):
            raw_deltas = cast(FloatArray, target - self._values)
        deltas = _require_finite_array("distributional deltas", raw_deltas)
        rates = cast(
            FloatArray,
            np.where(
                deltas > 0.0,
                np.array([c.alpha_pos for c in self._channels], dtype=np.float64),
                np.array([c.alpha_neg for c in self._channels], dtype=np.float64),
            ),
        )
        with np.errstate(over="ignore", invalid="ignore"):
            raw_updated = cast(FloatArray, self._values + rates * deltas)
        updated = _require_finite_array("updated values", raw_updated)
        self._values = updated
        self._n_updates += 1
        return cast(FloatArray, deltas.copy())

    def observe_td_target(
        self,
        reward: float,
        next_value: float,
        discount_gamma: float,
    ) -> FloatArray:
        """Update from the TD target reward + gamma * next_value."""
        reward = _require_finite("reward", float(reward))
        next_value = _require_finite("next_value", float(next_value))
        discount_gamma = _require_finite("discount_gamma", float(discount_gamma))
        if not 0.0 < discount_gamma <= 1.0:
            raise ValueError("discount_gamma must be in (0, 1]")
        td_target = _require_finite("td_target", reward + discount_gamma * next_value)
        return self.observe_return(td_target)

    @property
    def taus(self) -> FloatArray:
        """Expectile levels, ascending."""
        return cast(
            FloatArray,
            np.array([c.tau for c in self._channels], dtype=np.float64),
        )

    @property
    def values(self) -> FloatArray:
        """Current per-channel estimates, aligned with taus."""
        return cast(FloatArray, self._values.copy())

    def expectile_curve(self) -> tuple[FloatArray, FloatArray]:
        """Return (taus, values)."""
        return self.taus, self.values

    def canonical_value(self) -> float:
        """Return the tau=0.5 value or its interpolation."""
        taus = self.taus
        exact = np.isclose(taus, 0.5)
        if bool(np.any(exact)):
            return _require_finite("canonical_value", float(self._values[exact][0]))
        return _require_finite(
            "canonical_value",
            float(np.interp(0.5, taus, self._values)),
        )

    def risk_adjusted_value(self, tau: float) -> float:
        """Return the value at expectile level tau."""
        tau = _require_finite("tau", float(tau))
        if not 0.0 < tau < 1.0:
            raise ValueError(f"tau must be in (0, 1), got {tau!r}")
        taus = self.taus
        return _require_finite(
            "risk_adjusted_value",
            float(np.interp(tau, taus, self._values)),
        )

    def dispersion(self) -> float:
        """Return the top-minus-bottom channel spread."""
        return _require_finite("dispersion", float(self._values[-1] - self._values[0]))

    @property
    def n_updates(self) -> int:
        """Number of updates applied so far."""
        return self._n_updates
