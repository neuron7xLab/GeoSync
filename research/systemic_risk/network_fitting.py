# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""MLE fitting of interbank degree distributions to candidate generators.

The Barabási–Albert generator at fixed *m* is the topology-null
baseline of the systemic-risk falsification battery. v1 hard-coded
``m=3`` from Boss et al. 2004 / Soramäki et al. 2007; v2 fits *m*
(equivalently the power-law exponent α) directly to the empirical
degree sequence by maximum likelihood per Clauset, Shalizi & Newman
(2009), *SIAM Rev.* **51**: 661.

Two candidate models are compared on every fit:

* **Power law** (BA-like): :math:`P(k) \\propto k^{-\\alpha}` for
  :math:`k \\ge k_{\\min}`. MLE estimator
  :math:`\\hat{\\alpha} = 1 + n / \\sum_i \\ln(k_i / k_{\\min})`.
* **Exponential**: :math:`P(k) \\propto e^{-\\lambda k}` for
  :math:`k \\ge k_{\\min}`. MLE estimator
  :math:`\\hat{\\lambda} = 1 / (\\bar{k} - k_{\\min})`.

The Kolmogorov–Smirnov statistic and a parametric-bootstrap p-value
quantify the goodness-of-fit; the Akaike Information Criterion picks
between the two candidates without overfitting penalty drift.

Pure-function API. No I/O. Determinism via explicit seeds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "PowerLawFit",
    "ExponentialFit",
    "ModelComparison",
    "fit_power_law",
    "fit_exponential",
    "compare_power_law_vs_exponential",
    "fit_barabasi_albert",
]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PowerLawFit:
    """MLE fit of a discrete power-law tail :math:`P(k) \\propto k^{-\\alpha}`.

    Attributes
    ----------
    alpha
        Maximum-likelihood exponent. Standard error
        :math:`\\sigma_\\alpha = (\\alpha - 1) / \\sqrt{n}`
        (Clauset et al. 2009, eq. 3.2).
    alpha_se
        Asymptotic standard error of ``alpha``.
    k_min
        Lower-tail cutoff used by the fit.
    n_tail
        Number of degrees in the fit tail (``k >= k_min``).
    log_likelihood
        Log-likelihood of the tail under the fitted model.
    ks_statistic
        Kolmogorov–Smirnov distance between the empirical tail CDF
        and the fitted CDF.
    ks_p_value
        Parametric-bootstrap p-value: probability that a synthetic
        sample of size ``n_tail`` drawn from the fitted distribution
        produces a KS statistic at least as large as the observed one.
        ``None`` when the bootstrap was not requested.
    """

    alpha: float
    alpha_se: float
    k_min: int
    n_tail: int
    log_likelihood: float
    ks_statistic: float
    ks_p_value: float | None


@dataclass(frozen=True, slots=True)
class ExponentialFit:
    """MLE fit of a discrete shifted exponential :math:`P(k) \\propto e^{-\\lambda k}`."""

    lambda_: float
    k_min: int
    n_tail: int
    log_likelihood: float
    ks_statistic: float


@dataclass(frozen=True, slots=True)
class ModelComparison:
    """AIC-based comparison of power-law vs exponential.

    ``preferred`` is the model with the lower AIC. ``aic_delta`` is
    the absolute AIC gap (smaller is more decisive — Burnham &
    Anderson 2002 conventionally read Δ < 2 as roughly equivalent,
    Δ ∈ (4, 7) as strong, Δ > 10 as decisive).
    """

    power_law: PowerLawFit
    exponential: ExponentialFit
    aic_power_law: float
    aic_exponential: float
    preferred: Literal["power_law", "exponential"]
    aic_delta: float


# ---------------------------------------------------------------------------
# MLE estimators
# ---------------------------------------------------------------------------


def _validate_degrees(degrees: NDArray[np.int64]) -> NDArray[np.int64]:
    d = np.asarray(degrees, dtype=np.int64)
    if d.ndim != 1:
        raise ValueError(f"degrees must be 1-D, got shape={d.shape}")
    if d.size < 4:
        raise ValueError(f"degrees must contain at least 4 entries, got n={d.size}")
    if np.any(d < 0):
        raise ValueError("degrees must be non-negative")
    return d


def fit_power_law(
    degrees: NDArray[np.int64],
    *,
    k_min: int | None = None,
    n_bootstrap: int = 0,
    seed: int = 42,
) -> PowerLawFit:
    """Fit a discrete power-law tail by maximum likelihood.

    Parameters
    ----------
    degrees
        1-D array of non-negative degree observations.
    k_min
        Tail cutoff. ``None`` (default) selects the cutoff that
        minimises the KS statistic over all admissible cutoffs
        (Clauset et al. 2009, §3.3 — discrete grid scan).
    n_bootstrap
        Number of parametric-bootstrap resamples for the p-value.
        ``0`` (default) skips the bootstrap and returns ``ks_p_value=None``.
    seed
        RNG seed for the bootstrap.
    """
    d = _validate_degrees(degrees)
    if k_min is None:
        chosen = _select_k_min(d)
    else:
        if k_min < 1:
            raise ValueError(f"k_min must be >= 1, got {k_min}")
        chosen = int(k_min)
    tail = d[d >= chosen]
    if tail.size < 2:
        raise ValueError(f"only {tail.size} observations >= k_min={chosen}; need >=2 to fit")
    # Continuous-approximation MLE — bias-stable for k_min >= 6 (Clauset 2009).
    log_ratio = float(np.log(tail.astype(np.float64) / (chosen - 0.5)).sum())
    if log_ratio <= 0:
        raise ValueError("non-positive log-ratio sum — degenerate tail")
    alpha = 1.0 + tail.size / log_ratio
    alpha_se = (alpha - 1.0) / math.sqrt(tail.size)
    log_likelihood = (
        tail.size * math.log(alpha - 1.0)
        - tail.size * math.log(chosen - 0.5)
        - alpha * float(np.log(tail.astype(np.float64) / (chosen - 0.5)).sum())
    )
    ks = _ks_statistic_power_law(tail, alpha, chosen)
    p_value: float | None = None
    if n_bootstrap > 0:
        p_value = _ks_p_value_power_law(
            tail.size, alpha, chosen, observed_ks=ks, n_bootstrap=n_bootstrap, seed=seed
        )
    return PowerLawFit(
        alpha=float(alpha),
        alpha_se=float(alpha_se),
        k_min=int(chosen),
        n_tail=int(tail.size),
        log_likelihood=float(log_likelihood),
        ks_statistic=float(ks),
        ks_p_value=p_value,
    )


def fit_exponential(
    degrees: NDArray[np.int64],
    *,
    k_min: int | None = None,
) -> ExponentialFit:
    """Fit a shifted exponential tail by maximum likelihood."""
    d = _validate_degrees(degrees)
    chosen = int(k_min) if k_min is not None else int(d[d > 0].min()) if (d > 0).any() else 1
    if chosen < 1:
        raise ValueError(f"k_min must be >= 1, got {chosen}")
    tail = d[d >= chosen].astype(np.float64)
    if tail.size < 2:
        raise ValueError(f"only {tail.size} observations >= k_min={chosen}; need >=2 to fit")
    mean_excess = float(tail.mean()) - chosen
    if mean_excess <= 0:
        raise ValueError(f"mean degree {tail.mean():.3f} <= k_min={chosen}; cannot fit")
    lam = 1.0 / mean_excess
    log_likelihood = float(tail.size * math.log(lam) - lam * float((tail - chosen).sum()))
    ks = _ks_statistic_exponential(tail, lam, chosen)
    return ExponentialFit(
        lambda_=float(lam),
        k_min=int(chosen),
        n_tail=int(tail.size),
        log_likelihood=float(log_likelihood),
        ks_statistic=float(ks),
    )


def compare_power_law_vs_exponential(
    degrees: NDArray[np.int64],
    *,
    k_min: int | None = None,
    n_bootstrap: int = 0,
    seed: int = 42,
) -> ModelComparison:
    """Fit both candidates and pick by AIC.

    AIC = 2 k − 2 ℓ̂ where *k* is the number of free parameters
    (1 for both models given a fixed ``k_min``) and ℓ̂ is the maximum
    log-likelihood. Lower is better.
    """
    pl = fit_power_law(degrees, k_min=k_min, n_bootstrap=n_bootstrap, seed=seed)
    exp_fit = fit_exponential(degrees, k_min=k_min)
    aic_pl = 2.0 * 1 - 2.0 * pl.log_likelihood
    aic_exp = 2.0 * 1 - 2.0 * exp_fit.log_likelihood
    preferred: Literal["power_law", "exponential"] = (
        "power_law" if aic_pl < aic_exp else "exponential"
    )
    return ModelComparison(
        power_law=pl,
        exponential=exp_fit,
        aic_power_law=float(aic_pl),
        aic_exponential=float(aic_exp),
        preferred=preferred,
        aic_delta=float(abs(aic_pl - aic_exp)),
    )


def fit_barabasi_albert(
    degrees: NDArray[np.int64],
    *,
    n_bootstrap: int = 0,
    seed: int = 42,
) -> tuple[int, PowerLawFit]:
    """Fit a BA-compatible *m* parameter to an empirical degree sequence.

    The Barabási–Albert generator produces a power-law tail with
    exponent α = 3 in the thermodynamic limit; finite-size and
    direction-asymmetry effects shift α typically into [2, 3]
    (Albert & Barabási 2002, *Rev. Mod. Phys.* 74: 47). Once the
    empirical α is fitted, *m* is recovered from the BA mean-degree
    identity ``<k> = 2m``: rounding ``mean(k) / 2`` to the nearest
    positive integer.

    Returns the fitted *m* and the underlying :class:`PowerLawFit`
    so the caller can inspect ``alpha``, ``ks_statistic``, and
    (when ``n_bootstrap > 0``) ``ks_p_value``.
    """
    d = _validate_degrees(degrees)
    pl = fit_power_law(d, n_bootstrap=n_bootstrap, seed=seed)
    mean_k = float(d.mean())
    m = max(1, int(round(mean_k / 2.0)))
    return m, pl


# ---------------------------------------------------------------------------
# Internals: k_min selection + KS statistic + bootstrap p
# ---------------------------------------------------------------------------


def _select_k_min(degrees: NDArray[np.int64]) -> int:
    """Clauset-Shalizi-Newman 2009 §3.3 k_min selection by KS minimisation."""
    candidates = np.unique(degrees[degrees >= 1])
    # The largest few candidates leave too few tail points to fit; cap.
    if candidates.size <= 4:
        return int(candidates[0]) if candidates.size > 0 else 1
    upper_idx = max(1, candidates.size - 4)
    candidates = candidates[:upper_idx]
    best_k_min = int(candidates[0])
    best_ks = math.inf
    for k_min in candidates:
        tail = degrees[degrees >= k_min]
        if tail.size < 4:
            continue
        log_ratio = float(np.log(tail.astype(np.float64) / (int(k_min) - 0.5)).sum())
        if log_ratio <= 0:
            continue
        alpha = 1.0 + tail.size / log_ratio
        ks = _ks_statistic_power_law(tail, alpha, int(k_min))
        if ks < best_ks:
            best_ks = ks
            best_k_min = int(k_min)
    return best_k_min


def _power_law_cdf(k: NDArray[np.int64], alpha: float, k_min: int) -> NDArray[np.float64]:
    """Continuous-approximation CDF for a discrete power-law tail."""
    k_arr = np.asarray(k, dtype=np.float64)
    return 1.0 - ((k_arr - 0.5) / (k_min - 0.5)) ** (1.0 - alpha)


def _ks_statistic_power_law(tail: NDArray[np.int64], alpha: float, k_min: int) -> float:
    sorted_tail = np.sort(tail)
    n = sorted_tail.size
    empirical_cdf = np.arange(1, n + 1, dtype=np.float64) / n
    fitted_cdf = _power_law_cdf(sorted_tail, alpha, k_min)
    return float(np.max(np.abs(empirical_cdf - fitted_cdf)))


def _exponential_cdf(k: NDArray[np.float64], lam: float, k_min: int) -> NDArray[np.float64]:
    k_arr = np.asarray(k, dtype=np.float64)
    return 1.0 - np.exp(-lam * (k_arr - k_min))


def _ks_statistic_exponential(tail: NDArray[np.float64], lam: float, k_min: int) -> float:
    sorted_tail = np.sort(tail)
    n = sorted_tail.size
    empirical_cdf = np.arange(1, n + 1, dtype=np.float64) / n
    fitted_cdf = _exponential_cdf(sorted_tail, lam, k_min)
    return float(np.max(np.abs(empirical_cdf - fitted_cdf)))


def _draw_power_law_sample(
    n: int, alpha: float, k_min: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    """Inverse-CDF sampler for the continuous power-law approximation."""
    u = rng.random(n)
    raw = (k_min - 0.5) * (1.0 - u) ** (1.0 / (1.0 - alpha)) + 0.5
    return np.maximum(np.rint(raw).astype(np.int64), k_min)


def _ks_p_value_power_law(
    n: int,
    alpha: float,
    k_min: int,
    *,
    observed_ks: float,
    n_bootstrap: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(n_bootstrap):
        synthetic = _draw_power_law_sample(n, alpha, k_min, rng)
        # Re-fit on the synthetic sample then compute its own KS.
        log_ratio = float(np.log(synthetic.astype(np.float64) / (k_min - 0.5)).sum())
        if log_ratio <= 0:
            continue
        synth_alpha = 1.0 + n / log_ratio
        synth_ks = _ks_statistic_power_law(synthetic, synth_alpha, k_min)
        if synth_ks >= observed_ks:
            exceedances += 1
    # Davison & Hinkley (1997) +1 continuity correction.
    return float((exceedances + 1) / (n_bootstrap + 1))
