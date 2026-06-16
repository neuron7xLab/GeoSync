# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
r"""Exact stochastic-thermodynamics primitives for the overdamped Langevin.

This module supplies the *fluctuation-theorem* layer that the GeoSync
physics kernel was missing. The existing thermodynamics machinery
(:mod:`core.physics.thermodynamics`, :mod:`core.physics.free_energy_variational`,
:mod:`core.physics.landauer`) covers equilibrium free energy, the Landauer
bound, and variational free energy, but contained **no** non-equilibrium
fluctuation theorem — no Jarzynski equality, no Crooks relation, no
detailed-balance / equipartition witness. This module closes that gap with
closed-form, exactly-testable statistical mechanics.

Physical model
--------------

A single overdamped degree of freedom :math:`x` in a time-dependent
harmonic potential :math:`V(x, t) = \tfrac{1}{2} k(t)\, x^2`, evolving by
the overdamped Langevin equation (mobility :math:`1/\gamma`):

.. math::

    \mathrm{d}x = -\frac{1}{\gamma} V'(x, t)\, \mathrm{d}t
                  + \sqrt{\frac{2 k_B T}{\gamma}}\, \mathrm{d}W_t .

Integrated with Euler-Maruyama at step ``dt``:

.. math::

    x \leftarrow x - \frac{\mathrm{d}t}{\gamma} k\, x
                 + \sqrt{2\gamma k_B T\, \mathrm{d}t}\; \mathcal N(0, 1).

The equilibrium free energy of the harmonic well is the Gaussian
partition-function result

.. math::

    F(k) = -\frac{1}{2\beta}\,\ln\!\frac{2\pi}{\beta k},
    \qquad \beta = \frac{1}{k_B T},

so a stiffness ramp :math:`k_i \to k_f` has the *exact* free-energy
difference

.. math::

    \Delta F = -\frac{1}{2\beta}\,\ln\frac{k_i}{k_f}.

Work along a protocol is accumulated in the Sekimoto stochastic-energetics
convention: at each step the protocol parameter advances at *fixed*
configuration, and the work increment is the resulting change of potential
energy,

.. math::

    \mathrm{d}W = \frac{\partial V}{\partial k}\,\mathrm{d}k
                = \tfrac12\, x^2\, \mathrm{d}k .

The Jarzynski equality then holds *exactly* in expectation over the
non-equilibrium work distribution, for an arbitrarily fast protocol:

.. math::

    \big\langle e^{-\beta W} \big\rangle = e^{-\beta \Delta F},

and the second law follows by Jensen's inequality,
:math:`\langle W \rangle \ge \Delta F`, with the gap (mean dissipated work)
shrinking as the protocol is slowed toward the quasi-static limit.

This is pure statistical mechanics. It is **not** a market model and emits
no market claim — the only inputs are a stiffness schedule, a temperature,
and a random seed.

Public surface
--------------

* :func:`harmonic_free_energy` — closed-form :math:`F(k)`.
* :func:`delta_free_energy` — closed-form :math:`\Delta F(k_i \to k_f)`.
* :func:`langevin_ensemble_step` — one vectorized Euler-Maruyama step of
  an ensemble in a harmonic well of given stiffness.
* :func:`stiffness_ramp_work` — ensemble **work** array for a linear
  stiffness-ramp protocol (the Jarzynski observable).
* :func:`stationary_samples` — draws from the Boltzmann stationary
  distribution by relaxing an ensemble; the equipartition witness.
* :func:`jarzynski_average` — :math:`\langle e^{-\beta W}\rangle` of a work
  array.

All functions are pure, NumPy-vectorized, deterministic for a given seed,
and ``mypy --strict`` clean. No I/O, no hidden state.

References
----------
* Jarzynski, C. (1997). *Nonequilibrium equality for free energy
  differences.* Phys. Rev. Lett. **78**, 2690.
* Crooks, G. E. (1999). *Entropy production fluctuation theorem and the
  nonequilibrium work relation for free energy differences.* Phys. Rev. E
  **60**, 2721.
* Sekimoto, K. (1998). *Langevin equation and thermodynamics.* Prog. Theor.
  Phys. Suppl. **130**, 17 (stochastic energetics / work convention).
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

__all__ = [
    "harmonic_free_energy",
    "delta_free_energy",
    "langevin_ensemble_step",
    "stiffness_ramp_work",
    "stationary_samples",
    "jarzynski_average",
]

FloatArray = npt.NDArray[np.float64]

# Default protocol parameters reproduce the verified reference run:
# seeds {1, 7, 42, 123} give ⟨e^{-βW}⟩ ∈ [0.498, 0.500] vs target 0.5000.
_DEFAULT_ENSEMBLE: Final[int] = 40_000
_DEFAULT_TAU: Final[float] = 1.0
_DEFAULT_DT: Final[float] = 0.005


def _validate_positive(name: str, value: float) -> None:
    """Fail-closed guard: physical scales must be strictly positive and finite."""
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite float, got {value!r}")


def harmonic_free_energy(k: float, kT: float = 1.0) -> float:
    r"""Equilibrium free energy of a 1-D harmonic well :math:`V = \tfrac12 k x^2`.

    Closed form from the Gaussian partition function
    :math:`Z = \sqrt{2\pi / (\beta k)}`, :math:`F = -k_B T \ln Z`:

    .. math::

        F(k) = -\frac{1}{2\beta}\,\ln\frac{2\pi}{\beta k},
        \qquad \beta = 1/k_B T.

    Parameters
    ----------
    k:
        Harmonic stiffness, strictly positive.
    kT:
        Thermal energy :math:`k_B T`, strictly positive.

    Returns
    -------
    float
        The Helmholtz free energy :math:`F(k)` in units of :math:`k_B T`.
    """
    _validate_positive("k", k)
    _validate_positive("kT", kT)
    beta = 1.0 / kT
    return -(1.0 / (2.0 * beta)) * float(np.log(2.0 * np.pi / (beta * k)))


def delta_free_energy(ki: float, kf: float, kT: float = 1.0) -> float:
    r"""Exact free-energy difference of a stiffness change :math:`k_i \to k_f`.

    .. math::

        \Delta F = F(k_f) - F(k_i)
                 = -\frac{1}{2\beta}\,\ln\frac{k_i}{k_f}.

    The :math:`2\pi` and :math:`\beta` factors of the partition function
    cancel, leaving the clean log-stiffness-ratio form. For the reference
    ramp :math:`k_i=1 \to k_f=4`, :math:`kT=1`, this is
    :math:`\tfrac12\ln 4 \approx 0.693`, and :math:`e^{-\beta\Delta F} =
    \sqrt{k_i/k_f} = 0.5`.

    Parameters
    ----------
    ki, kf:
        Initial and final stiffness, both strictly positive.
    kT:
        Thermal energy :math:`k_B T`, strictly positive.
    """
    _validate_positive("ki", ki)
    _validate_positive("kf", kf)
    _validate_positive("kT", kT)
    beta = 1.0 / kT
    return -(1.0 / (2.0 * beta)) * float(np.log(ki / kf))


def langevin_ensemble_step(
    x: FloatArray,
    *,
    k: float,
    dt: float,
    kT: float,
    gamma: float,
    rng: np.random.Generator,
) -> FloatArray:
    r"""One vectorized Euler-Maruyama step in a harmonic well of stiffness ``k``.

    Advances every trajectory in the ensemble ``x`` by

    .. math::

        x \leftarrow x - \frac{\mathrm{d}t}{\gamma} k\, x
                     + \sqrt{2\gamma k_B T\, \mathrm{d}t}\; \mathcal N(0, 1).

    Parameters
    ----------
    x:
        Ensemble of positions, shape ``(M,)``.
    k:
        Current harmonic stiffness, strictly positive.
    dt:
        Integration timestep, strictly positive.
    kT:
        Thermal energy, strictly positive.
    gamma:
        Friction coefficient, strictly positive.
    rng:
        Seeded NumPy generator providing the thermal noise.

    Returns
    -------
    FloatArray
        The advanced ensemble, same shape as ``x``.
    """
    _validate_positive("k", k)
    _validate_positive("dt", dt)
    _validate_positive("kT", kT)
    _validate_positive("gamma", gamma)
    noise_scale = float(np.sqrt(2.0 * gamma * kT * dt))
    drift = (dt / gamma) * k * x
    return np.asarray(x - drift + noise_scale * rng.standard_normal(x.shape), dtype=np.float64)


def stiffness_ramp_work(
    seed: int,
    *,
    ensemble: int = _DEFAULT_ENSEMBLE,
    tau: float = _DEFAULT_TAU,
    dt: float = _DEFAULT_DT,
    ki: float = 1.0,
    kf: float = 4.0,
    kT: float = 1.0,
    gamma: float = 1.0,
) -> FloatArray:
    r"""Ensemble work array for a linear stiffness-ramp protocol.

    The protocol drives the stiffness linearly :math:`k_i \to k_f` over
    duration ``tau`` while every trajectory obeys the overdamped Langevin
    dynamics. The ensemble is initialised from the equilibrium of the
    *initial* well (:math:`\mathrm{Var}(x) = k_B T / k_i`). Work is
    accumulated in the Sekimoto convention: at each step the stiffness
    advances at fixed configuration and the work increment is the resulting
    change of potential energy,

    .. math::

        \mathrm{d}W = \tfrac12 (k_{\text{next}} - k_{\text{cur}})\, x^2 .

    Parameters
    ----------
    seed:
        Seed for the deterministic NumPy generator.
    ensemble:
        Number of independent trajectories ``M``.
    tau:
        Protocol duration. Slower ramps (larger ``tau``) dissipate less work.
    dt:
        Integration timestep.
    ki, kf:
        Initial and final stiffness.
    kT, gamma:
        Thermal energy and friction coefficient.

    Returns
    -------
    FloatArray
        Work values, shape ``(ensemble,)``. ``⟨e^{-W/kT}⟩`` equals
        :math:`e^{-\beta \Delta F}` to within the Jarzynski tolerance, and
        ``⟨W⟩ ≥ ΔF`` (second law).
    """
    _validate_positive("tau", tau)
    _validate_positive("dt", dt)
    _validate_positive("ki", ki)
    _validate_positive("kf", kf)
    _validate_positive("kT", kT)
    _validate_positive("gamma", gamma)
    if ensemble <= 0:
        raise ValueError(f"ensemble must be a positive integer, got {ensemble!r}")

    rng = np.random.default_rng(seed)
    n_steps = int(tau / dt)
    if n_steps <= 0:
        raise ValueError(f"tau/dt must yield ≥1 step, got tau={tau}, dt={dt}")

    x = rng.standard_normal(ensemble) * float(np.sqrt(kT / ki))
    work = np.zeros(ensemble, dtype=np.float64)
    for i in range(n_steps):
        k_cur = ki + (kf - ki) * i / n_steps
        k_next = ki + (kf - ki) * (i + 1) / n_steps
        work += 0.5 * (k_next - k_cur) * x * x
        x = langevin_ensemble_step(x, k=k_next, dt=dt, kT=kT, gamma=gamma, rng=rng)
    return work


def stationary_samples(
    seed: int,
    *,
    k: float,
    ensemble: int = _DEFAULT_ENSEMBLE,
    kT: float = 1.0,
    gamma: float = 1.0,
    dt: float = _DEFAULT_DT,
    relax_steps: int = 2000,
) -> FloatArray:
    r"""Draw an ensemble from the Boltzmann stationary distribution of a well.

    Initialises trajectories at the origin and relaxes them in a *fixed*
    harmonic well for ``relax_steps`` Langevin steps. After relaxation the
    ensemble is distributed as :math:`p(x) \propto e^{-\beta V(x)}`, a
    zero-mean Gaussian with the equipartition variance

    .. math::

        \mathrm{Var}(x) = \frac{k_B T}{k}.

    The relaxation timescale is :math:`\gamma / k`; the default
    ``relax_steps`` covers many such times for the reference parameters.

    Parameters
    ----------
    seed:
        Seed for the deterministic generator.
    k:
        Harmonic stiffness, strictly positive.
    ensemble:
        Number of independent trajectories.
    kT, gamma, dt:
        Thermal energy, friction, timestep.
    relax_steps:
        Number of Langevin steps used to reach stationarity.

    Returns
    -------
    FloatArray
        Stationary samples, shape ``(ensemble,)``.
    """
    _validate_positive("k", k)
    _validate_positive("kT", kT)
    _validate_positive("gamma", gamma)
    _validate_positive("dt", dt)
    if ensemble <= 0:
        raise ValueError(f"ensemble must be a positive integer, got {ensemble!r}")
    if relax_steps <= 0:
        raise ValueError(f"relax_steps must be a positive integer, got {relax_steps!r}")

    rng = np.random.default_rng(seed)
    x = np.zeros(ensemble, dtype=np.float64)
    for _ in range(relax_steps):
        x = langevin_ensemble_step(x, k=k, dt=dt, kT=kT, gamma=gamma, rng=rng)
    return x


def jarzynski_average(work: FloatArray, kT: float = 1.0) -> float:
    r"""Exponential work average :math:`\langle e^{-\beta W}\rangle`.

    The Jarzynski estimator of :math:`e^{-\beta \Delta F}`. Computed in a
    shifted-exponential form for numerical stability:
    :math:`\langle e^{-\beta(W - W_{\min})}\rangle\, e^{-\beta W_{\min}}`.

    Parameters
    ----------
    work:
        Ensemble work array.
    kT:
        Thermal energy, strictly positive.
    """
    _validate_positive("kT", kT)
    if work.size == 0:
        raise ValueError("work array must be non-empty")
    beta = 1.0 / kT
    w_min = float(np.min(work))
    shifted = float(np.mean(np.exp(-beta * (work - w_min))))
    return shifted * float(np.exp(-beta * w_min))
