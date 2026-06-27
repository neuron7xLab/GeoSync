# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Distributional dopamine value learner via an expectile ensemble.

This module turns the disconnected helpers in :mod:`experimental_rpe` into one
coherent, science-grounded estimator. It stays non-invasive: it does not touch
:class:`DopamineController`, does not rewrite INV-DA1/INV-DA7, and is not wired
into any production path. Promotion still has to be earned through the gates in
``docs/research/dopamine_rpe_extension_surface.md``.

Scientific anchor
-----------------
Dabney, Kurth-Nelson, Uchida, Starkweather, Hassabis, Munos & Botvinick (2020),
"A distributional code for value in dopamine-based reinforcement learning",
*Nature* 577:671-675. Each dopamine "channel" learns one statistic of the
return distribution. With a linear response function the learnt statistic is an
*expectile*, and the expectile level is fixed exactly by the asymmetry of the
positive/negative learning rates::

    tau_i = alpha_i_pos / (alpha_i_pos + alpha_i_neg)

[established, 3-vote verified] The asymmetry-to-reversal-point correlation
(p = 8.1e-5) and the diversity of reversal points (split-half R = 0.58,
p = 1.8e-5) are the empirical signatures this estimator reproduces.

Canonical reduction
-------------------
A symmetric channel (``alpha_pos == alpha_neg`` so ``tau = 0.5``) learns the
mean of the target stream, because the 0.5-expectile *is* the mean. The scalar
TD(0) anchor ``delta = reward + gamma * next_value - value`` is therefore a
special case of this estimator, not a thing it replaces:

    >>> ens = ExpectileEnsembleValue.symmetric(n_channels=1)
    >>> for _ in range(2000):
    ...     ens.observe_return(1.0)
    >>> abs(ens.canonical_value() - 1.0) < 1e-6
    True

Engineering boundary
--------------------
The neuroscience licenses only the *form* (expectile ensemble, asymmetric
learning, risk-as-low-expectile). Neural ``alpha`` values, reversal points and
R^2 are not market parameters; channel levels here must be configured for the
task and locked through the promotion gates, never copied from the biology.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

# Maximum iterations / tolerance for the offline expectile fixed point.
_EXPECTILE_MAX_ITERS = 256
_EXPECTILE_TOL = 1e-13


def _require_finite(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def expectile_of_samples(samples: npt.ArrayLike, tau: float) -> float:
    """Return the ``tau``-expectile of ``samples`` (offline ground truth).

    The ``tau``-expectile ``e`` is the unique fixed point of the asymmetrically
    weighted mean

    ``e = (tau * sum_{z>=e} z + (1-tau) * sum_{z<e} z)
          / (tau * |{z>=e}| + (1-tau) * |{z<e}|)``

    For ``tau = 0.5`` this collapses to the arithmetic mean. The iteration is
    monotone and converges to machine precision; it is the analytic reference
    that :class:`ExpectileEnsembleValue` must reproduce online (its falsifier).

    Raises:
        ValueError: if ``tau`` is not in ``(0, 1)`` or ``samples`` is empty or
            contains a non-finite value.
    """
    tau = _require_finite("tau", float(tau))
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must be in (0, 1), got {tau!r}")
    arr = np.asarray(samples, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("samples must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("samples must be finite")

    # Seed at the mean and iterate the weighted-mean map to its fixed point.
    e = float(arr.mean())
    for _ in range(_EXPECTILE_MAX_ITERS):
        upper = arr >= e
        w_upper = tau * np.count_nonzero(upper)
        w_lower = (1.0 - tau) * np.count_nonzero(~upper)
        denom = w_upper + w_lower
        if denom <= 0.0:  # pragma: no cover - guarded by arr.size > 0
            break
        new_e = (tau * arr[upper].sum() + (1.0 - tau) * arr[~upper].sum()) / denom
        if abs(new_e - e) <= _EXPECTILE_TOL * (1.0 + abs(e)):
            e = new_e
            break
        e = new_e
    return float(e)


@dataclass(frozen=True)
class ExpectileChannel:
    """One dopamine value channel learning a single expectile level.

    Attributes:
        tau: Expectile level in ``(0, 1)``; ``0.5`` is the mean / canonical
            channel, ``tau < 0.5`` is risk-averse (pessimistic), ``tau > 0.5``
            is risk-seeking (optimistic).
        learning_rate: Base learning rate ``eta > 0``. The asymmetric rates are
            ``alpha_pos = eta * tau`` and ``alpha_neg = eta * (1 - tau)`` so the
            identity ``tau = alpha_pos / (alpha_pos + alpha_neg)`` holds exactly.
    """

    tau: float
    learning_rate: float

    def __post_init__(self) -> None:
        tau = _require_finite("tau", float(self.tau))
        eta = _require_finite("learning_rate", float(self.learning_rate))
        if not 0.0 < tau < 1.0:
            raise ValueError(f"tau must be in (0, 1), got {tau!r}")
        if not 0.0 < eta <= 1.0:
            raise ValueError(f"learning_rate must be in (0, 1], got {eta!r}")

    @property
    def alpha_pos(self) -> float:
        """Learning rate applied to positive (better-than-expected) errors."""
        return self.learning_rate * self.tau

    @property
    def alpha_neg(self) -> float:
        """Learning rate applied to negative (worse-than-expected) errors."""
        return self.learning_rate * (1.0 - self.tau)

    def recovered_tau(self) -> float:
        """Return ``alpha_pos / (alpha_pos + alpha_neg)`` — must equal ``tau``.

        This is the Dabney (2020) identity made checkable: it is the algebraic
        invariant that ties the asymmetric learning rates to the expectile the
        channel converges on.
        """
        total = self.alpha_pos + self.alpha_neg
        return self.alpha_pos / total


class ExpectileEnsembleValue:
    """An ensemble of expectile channels approximating the return distribution.

    Each channel updates its value estimate ``V_i`` with an asymmetric step

    ``V_i <- V_i + (alpha_pos if delta > 0 else alpha_neg) * delta``,
    ``delta = target - V_i``

    so ``V_i`` converges to the ``tau_i``-expectile of the target stream. The
    estimator is fully deterministic: identical inputs produce bit-identical
    state, so no seed is needed.
    """

    def __init__(self, channels: list[ExpectileChannel], *, init_value: float = 0.0) -> None:
        if not channels:
            raise ValueError("ensemble requires at least one channel")
        init_value = _require_finite("init_value", float(init_value))
        taus = [c.tau for c in channels]
        if len(set(taus)) != len(taus):
            raise ValueError("channel tau levels must be unique")
        # Store channels sorted by tau so the expectile curve is monotone by index.
        order = sorted(range(len(channels)), key=lambda i: channels[i].tau)
        self._channels: tuple[ExpectileChannel, ...] = tuple(channels[i] for i in order)
        self._values: FloatArray = np.full(len(channels), init_value, dtype=np.float64)
        self._n_updates: int = 0

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def symmetric(
        cls, *, n_channels: int = 1, learning_rate: float = 0.05
    ) -> ExpectileEnsembleValue:
        """Build an ensemble whose single channel is the canonical mean (tau=0.5).

        ``n_channels`` symmetric copies are not useful (taus must be unique), so
        only ``n_channels == 1`` is accepted here; it exists to make the
        canonical reduction explicit and testable.
        """
        if n_channels != 1:
            raise ValueError("symmetric ensemble is the single tau=0.5 channel")
        return cls([ExpectileChannel(tau=0.5, learning_rate=learning_rate)])

    @classmethod
    def from_levels(
        cls, taus: npt.ArrayLike, *, learning_rate: float = 0.05, init_value: float = 0.0
    ) -> ExpectileEnsembleValue:
        """Build an ensemble from explicit expectile levels.

        Always include ``0.5`` (or a level near it) if you need a canonical mean
        read-out; :meth:`canonical_value` interpolates when it is absent.
        """
        levels = np.asarray(taus, dtype=np.float64).ravel()
        channels = [ExpectileChannel(tau=float(t), learning_rate=learning_rate) for t in levels]
        return cls(channels, init_value=init_value)

    @classmethod
    def spread(
        cls, n_channels: int = 9, *, learning_rate: float = 0.05, init_value: float = 0.0
    ) -> ExpectileEnsembleValue:
        """Build a symmetric spread of expectile levels across ``(0, 1)``.

        Levels are ``i / (n_channels + 1)`` for ``i = 1..n_channels``, which is
        symmetric about ``0.5`` and always includes ``0.5`` when ``n_channels``
        is odd — giving an exact canonical-mean channel plus risk-averse and
        risk-seeking flanks.
        """
        if n_channels < 1:
            raise ValueError("n_channels must be >= 1")
        levels = np.arange(1, n_channels + 1, dtype=np.float64) / (n_channels + 1)
        return cls.from_levels(levels, learning_rate=learning_rate, init_value=init_value)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def observe_return(self, target: float) -> FloatArray:
        """Update every channel toward a realised/bootstrapped return ``target``.

        Returns the per-channel signed prediction errors ``delta_i`` (the
        distributional RPE vector) *before* the update is applied, so callers
        can inspect surprise without recomputing it.
        """
        target = _require_finite("target", float(target))
        deltas = target - self._values
        rates = np.where(
            deltas > 0.0,
            np.array([c.alpha_pos for c in self._channels], dtype=np.float64),
            np.array([c.alpha_neg for c in self._channels], dtype=np.float64),
        )
        self._values = self._values + rates * deltas
        self._n_updates += 1
        return deltas

    def observe_td_target(
        self, reward: float, next_value: float, discount_gamma: float
    ) -> FloatArray:
        """Update from a TD(0) bootstrap target ``reward + gamma * next_value``.

        The bootstrap target is the canonical anchor's right-hand side, so the
        ``tau = 0.5`` channel tracks exactly the scalar TD(0) value while the
        other channels add the distributional spread around it.
        """
        reward = _require_finite("reward", float(reward))
        next_value = _require_finite("next_value", float(next_value))
        discount_gamma = _require_finite("discount_gamma", float(discount_gamma))
        if not 0.0 < discount_gamma <= 1.0:
            raise ValueError("discount_gamma must be in (0, 1]")
        return self.observe_return(reward + discount_gamma * next_value)

    # ------------------------------------------------------------------
    # Read-outs
    # ------------------------------------------------------------------
    @property
    def taus(self) -> FloatArray:
        """Expectile levels, ascending."""
        return np.array([c.tau for c in self._channels], dtype=np.float64)

    @property
    def values(self) -> FloatArray:
        """Current per-channel expectile estimates, aligned with :attr:`taus`."""
        return self._values.copy()

    def expectile_curve(self) -> tuple[FloatArray, FloatArray]:
        """Return ``(taus, values)`` — the learnt, monotone expectile curve."""
        return self.taus, self.values

    def canonical_value(self) -> float:
        """Return the mean / canonical TD(0) value (the ``tau = 0.5`` expectile).

        If no channel sits exactly at ``0.5`` the value is linearly interpolated
        between the two nearest levels, so this always returns a scalar mean
        estimate consistent with the canonical anchor.
        """
        taus = self.taus
        exact = np.isclose(taus, 0.5)
        if bool(np.any(exact)):
            return float(self._values[exact][0])
        return float(np.interp(0.5, taus, self._values))

    def risk_adjusted_value(self, tau: float) -> float:
        """Return the value at expectile level ``tau`` (risk-sensitive read-out).

        ``tau < 0.5`` yields a pessimistic, risk-averse value; ``tau > 0.5`` a
        risk-seeking one. This is the structurally-licensed risk layer: risk
        sensitivity emerges from the expectile ensemble itself, not from an
        ad-hoc penalty bolted onto the reward.
        """
        tau = _require_finite("tau", float(tau))
        if not 0.0 < tau < 1.0:
            raise ValueError(f"tau must be in (0, 1), got {tau!r}")
        taus = self.taus
        return float(np.interp(tau, taus, self._values))

    def dispersion(self) -> float:
        """Return a non-negative spread proxy (top expectile minus bottom).

        Zero when the target stream is deterministic; grows with return
        variability. It is a read-out of the distribution width, not a variance
        estimate, and is guaranteed ``>= 0`` by the channel ordering.
        """
        return float(self._values[-1] - self._values[0])

    @property
    def n_updates(self) -> int:
        """Number of ``observe_*`` calls applied so far."""
        return self._n_updates
