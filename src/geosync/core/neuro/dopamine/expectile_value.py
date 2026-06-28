from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _finite(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _finite_arr(name: str, values: FloatArray) -> FloatArray:
    if not bool(np.all(np.isfinite(values))):
        raise ValueError(f"{name} must be finite")
    return values


def expectile_of_samples(samples: npt.ArrayLike, tau: float) -> float:
    tau = _finite("tau", float(tau))
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must be in (0, 1), got {tau!r}")
    arr = np.asarray(samples, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("samples must be non-empty")
    _finite_arr("samples", arr)
    e = _finite("expectile seed", float(np.median(arr)))
    scale = max(1.0, _finite("sample scale", float(np.max(np.abs(arr)))))
    for _ in range(256):
        x = arr / scale
        y = e / scale
        up = x >= y
        denom = tau * float(np.count_nonzero(up))
        denom += (1.0 - tau) * float(np.count_nonzero(~up))
        if denom <= 0.0:
            break
        residual = tau * float((x[up] - y).sum())
        residual += (1.0 - tau) * float((x[~up] - y).sum())
        new_y = _finite("scaled expectile", y + residual / denom)
        new_e = _finite("expectile", new_y * scale)
        if abs(new_y - y) <= 1e-13 * (1.0 + abs(y)):
            e = new_e
            break
        e = new_e
    return _finite("expectile", float(e))


@dataclass(frozen=True)
class ExpectileChannel:
    tau: float
    learning_rate: float

    def __post_init__(self) -> None:
        tau = _finite("tau", float(self.tau))
        rate = _finite("learning_rate", float(self.learning_rate))
        if not 0.0 < tau < 1.0:
            raise ValueError(f"tau must be in (0, 1), got {tau!r}")
        if not 0.0 < rate <= 1.0:
            raise ValueError(f"learning_rate must be in (0, 1], got {rate!r}")
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "learning_rate", rate)

    @property
    def alpha_pos(self) -> float:
        return self.learning_rate * self.tau

    @property
    def alpha_neg(self) -> float:
        return self.learning_rate * (1.0 - self.tau)

    def recovered_tau(self) -> float:
        total = _finite("alpha_total", self.alpha_pos + self.alpha_neg)
        return _finite("recovered_tau", self.alpha_pos / total)


class ExpectileEnsembleValue:
    def __init__(self, channels: list[ExpectileChannel], *, init_value: float = 0.0) -> None:
        if not channels:
            raise ValueError("ensemble requires at least one channel")
        init_value = _finite("init_value", float(init_value))
        ordered = tuple(sorted(channels, key=lambda c: c.tau))
        if len({c.tau for c in ordered}) != len(ordered):
            raise ValueError("channel tau levels must be unique")
        self._taus = np.asarray([c.tau for c in ordered], dtype=np.float64)
        self._alpha_pos = np.asarray([c.alpha_pos for c in ordered], dtype=np.float64)
        self._alpha_neg = np.asarray([c.alpha_neg for c in ordered], dtype=np.float64)
        self._values = np.full(len(ordered), init_value, dtype=np.float64)
        _finite_arr("initial values", self._values)
        self._n_updates = 0

    @classmethod
    def symmetric(cls, *, n_channels: int = 1, learning_rate: float = 0.05) -> ExpectileEnsembleValue:
        if n_channels != 1:
            raise ValueError("symmetric ensemble is the single tau=0.5 channel")
        return cls([ExpectileChannel(0.5, learning_rate)])

    @classmethod
    def from_levels(cls, taus: npt.ArrayLike, *, learning_rate: float = 0.05, init_value: float = 0.0) -> ExpectileEnsembleValue:
        levels = np.asarray(taus, dtype=np.float64).ravel()
        return cls([ExpectileChannel(float(t), learning_rate) for t in levels], init_value=init_value)

    @classmethod
    def spread(cls, n_channels: int = 9, *, learning_rate: float = 0.05, init_value: float = 0.0) -> ExpectileEnsembleValue:
        if n_channels < 1:
            raise ValueError("n_channels must be >= 1")
        levels = np.arange(1, n_channels + 1, dtype=np.float64) / (n_channels + 1)
        return cls.from_levels(levels, learning_rate=learning_rate, init_value=init_value)

    def observe_return(self, target: float) -> FloatArray:
        target = _finite("target", float(target))
        deltas = _finite_arr("distributional deltas", target - self._values)
        rates = np.where(deltas > 0.0, self._alpha_pos, self._alpha_neg)
        self._values = _finite_arr("updated values", self._values + rates * deltas)
        self._n_updates += 1
        return deltas.copy()

    def observe_td_target(self, reward: float, next_value: float, discount_gamma: float) -> FloatArray:
        reward = _finite("reward", float(reward))
        next_value = _finite("next_value", float(next_value))
        gamma = _finite("discount_gamma", float(discount_gamma))
        if not 0.0 < gamma <= 1.0:
            raise ValueError("discount_gamma must be in (0, 1]")
        return self.observe_return(_finite("td_target", reward + gamma * next_value))

    @property
    def taus(self) -> FloatArray:
        return self._taus.copy()

    @property
    def values(self) -> FloatArray:
        return self._values.copy()

    def expectile_curve(self) -> tuple[FloatArray, FloatArray]:
        return self.taus, self.values

    def canonical_value(self) -> float:
        exact = np.isclose(self._taus, 0.5)
        if bool(np.any(exact)):
            return _finite("canonical_value", float(self._values[int(np.flatnonzero(exact)[0])]))
        return _finite("canonical_value", float(np.interp(0.5, self._taus, self._values)))

    def risk_adjusted_value(self, tau: float) -> float:
        tau = _finite("tau", float(tau))
        if not 0.0 < tau < 1.0:
            raise ValueError(f"tau must be in (0, 1), got {tau!r}")
        return _finite("risk_adjusted_value", float(np.interp(tau, self._taus, self._values)))

    def dispersion(self) -> float:
        return _finite("dispersion", float(self._values[-1] - self._values[0]))

    @property
    def n_updates(self) -> int:
        return self._n_updates
