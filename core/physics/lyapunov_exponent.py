# SPDX-License-Identifier: MIT
"""Maximal Lyapunov Exponent (MLE) for time-series chaos detection.

The MLE measures the average exponential rate of divergence of
infinitesimally close trajectories in phase space:

    λ_max = lim_{t→∞} (1/t) · ln(|δx(t)| / |δx(0)|)

For a market-state trajectory (e.g. Kuramoto R(t)):
- λ_max > 0  →  chaotic / unpredictable regime (gradient alive but turbulent)
- λ_max ≈ 0  →  marginal / edge-of-chaos (critical transition zone)
- λ_max < 0  →  stable / predictable regime (gradient converging)

This module implements the Rosenstein (1993) algorithm for MLE estimation
from a scalar time series, which avoids computing tangent vectors and
works directly on delay-embedded phase-space reconstructions.

Algorithm:
1. Delay-embed the scalar series: x_i → (x_i, x_{i+τ}, ..., x_{i+(m-1)τ})
2. For each point, find its nearest neighbor (excluding temporal neighbors)
3. Track the divergence of these neighbor pairs over time
4. λ_max = mean slope of ln(divergence) vs time

Connection to GeoSync:
- Feed R(t) from Kuramoto → MLE tells you if synchronization is stable
- Feed portfolio returns → MLE tells you if the return process is chaotic
- MLE → Cryptobiosis: extreme positive MLE could trigger DORMANT
- MLE → Kelly: negative MLE = higher conviction → larger Kelly fraction

References:
    Rosenstein, Collins & De Luca (1993). "A practical method for
    calculating largest Lyapunov exponents from small data sets."
    Physica D, 65(1-2), 117-134.

    Wolf, Swift, Swinney & Vastano (1985). "Determining Lyapunov
    exponents from a time series." Physica D, 16(3), 285-317.

Invariants:
    INV-LE1: MLE is finite for any finite bounded input series
    INV-LE2: MLE(noise) ≈ 0; MLE(stable oscillator) < 0; MLE(logistic chaos) > 0
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from numpy.typing import NDArray


def delay_embed(
    x: NDArray[np.float64],
    dim: int = 3,
    tau: int = 1,
) -> NDArray[np.float64]:
    """Delay-embed a scalar time series into a phase-space trajectory.

    Parameters
    ----------
    x : (T,) array
        Scalar time series.
    dim : int
        Embedding dimension (default 3).
    tau : int
        Delay in samples (default 1).

    Returns
    -------
    (T - (dim-1)*tau, dim) array
        Delay-embedded trajectory.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size - (dim - 1) * tau
    if n <= 0:
        raise ValueError(
            f"Series too short ({x.size}) for dim={dim}, tau={tau}. "
            f"Need at least {(dim - 1) * tau + 1} samples."
        )
    # INV-HPC2: build the embedding matrix without overflow
    indices = np.arange(n)[:, None] + np.arange(dim)[None, :] * tau
    result: NDArray[np.float64] = x[indices]
    return result


def maximal_lyapunov_exponent(
    x: NDArray[np.float64],
    dim: int = 3,
    tau: int = 1,
    max_divergence_steps: Optional[int] = None,
    min_temporal_separation: Optional[int] = None,
    dt: float = 1.0,
) -> float:
    """Estimate the maximal Lyapunov exponent from a scalar time series.

    Uses the Rosenstein (1993) nearest-neighbor divergence algorithm.

    Parameters
    ----------
    x : (T,) array
        Scalar time series (e.g. Kuramoto R(t), portfolio returns).
    dim : int
        Embedding dimension (default 3). Rule of thumb: 2·ceil(correlation_dim)+1.
    tau : int
        Embedding delay (default 1). Use first minimum of mutual information.
    max_divergence_steps : int, optional
        Maximum number of steps to track divergence (default T//4).
    min_temporal_separation : int, optional
        Minimum index separation for nearest neighbors (default dim*tau).
        Prevents self-matching in the embedding.
    dt : float
        Time step between samples (default 1.0). Scales the exponent
        to physical units: λ has units of 1/dt.

    Returns
    -------
    float
        Estimated λ_max. Positive = chaos, negative = stability, ≈0 = marginal.
        Returns 0.0 for series too short to estimate.

    Raises
    ------
    ValueError
        If the series is too short for the requested embedding.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 10:
        return 0.0

    # Delay embedding
    embedded = delay_embed(x, dim=dim, tau=tau)
    n_points = embedded.shape[0]

    if n_points < 20:
        return 0.0

    if max_divergence_steps is None:
        max_divergence_steps = max(10, n_points // 4)
    max_divergence_steps = min(max_divergence_steps, n_points - 1)

    if min_temporal_separation is None:
        min_temporal_separation = dim * tau
    min_temporal_separation = max(1, min_temporal_separation)

    # For each point, find its nearest neighbor (excluding temporal vicinity)
    # Using brute-force pairwise distances — O(n²) but robust.
    # For production with N > 10k, replace with scipy.spatial.cKDTree.
    divergence_log = np.full(max_divergence_steps, np.nan, dtype=np.float64)
    counts = np.zeros(max_divergence_steps, dtype=np.int64)

    for i in range(n_points - max_divergence_steps):
        # Distance from point i to all other points
        diffs = embedded[: n_points - max_divergence_steps] - embedded[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))

        # Exclude temporal neighbors
        temporal_mask = np.abs(np.arange(len(dists)) - i) < min_temporal_separation
        dists[temporal_mask] = np.inf
        dists[i] = np.inf  # exclude self

        j = int(np.argmin(dists))
        if not np.isfinite(dists[j]):
            continue

        # Track how this pair diverges over time
        for step in range(max_divergence_steps):
            if i + step >= n_points or j + step >= n_points:
                break
            pair_dist = float(np.sqrt(np.sum((embedded[i + step] - embedded[j + step]) ** 2)))
            if pair_dist > 0:
                if np.isnan(divergence_log[step]):
                    divergence_log[step] = 0.0
                divergence_log[step] += math.log(pair_dist)
                counts[step] += 1

    # Average the log-divergence curves
    valid = counts > 0
    if valid.sum() < 5:
        return 0.0

    mean_log_div = np.where(valid, divergence_log / counts, np.nan)

    # Linear regression on the valid portion to extract the slope = λ_max
    valid_indices = np.where(valid)[0]
    t_values = valid_indices.astype(np.float64) * dt
    y_values = mean_log_div[valid_indices]

    # Use only the initial linear region (first 40% of valid data)
    n_use = max(5, len(valid_indices) * 2 // 5)
    t_fit = t_values[:n_use]
    y_fit = y_values[:n_use]

    # Remove any NaN/Inf
    finite_mask = np.isfinite(y_fit) & np.isfinite(t_fit)
    t_fit = t_fit[finite_mask]
    y_fit = y_fit[finite_mask]

    if len(t_fit) < 3:
        return 0.0

    # Least-squares slope
    t_mean = float(np.mean(t_fit))
    y_mean = float(np.mean(y_fit))
    # INV-HPC2: protect against degenerate fits
    with np.errstate(over="ignore", invalid="ignore"):
        numerator = float(np.sum((t_fit - t_mean) * (y_fit - y_mean)))
        denominator = float(np.sum((t_fit - t_mean) ** 2))

    if denominator < 1e-30:
        return 0.0

    slope = numerator / denominator

    # INV-LE1: MLE must be finite for finite bounded input
    if not math.isfinite(slope):
        return 0.0

    return float(slope)


def spectral_gap(
    adjacency: NDArray[np.float64],
) -> float:
    """Compute the Fiedler eigenvalue (spectral gap) of the graph Laplacian.

    The spectral gap λ₂ is the second-smallest eigenvalue of the
    combinatorial Laplacian L = D - A, where D is the degree matrix
    and A is the adjacency matrix. It controls:

    - **Synchronization rate**: larger λ₂ → faster Kuramoto convergence
    - **Algebraic connectivity**: λ₂ > 0 ⟺ graph is connected
    - **Mixing time**: random walks mix in O(1/λ₂) steps
    - **Cheeger inequality**: λ₂/2 ≤ h(G) ≤ √(2·λ₂) where h is
      the isoperimetric number (bottleneck measure)

    For GeoSync:
    - λ₂ of the asset correlation graph = how quickly market can
      synchronize. Low λ₂ = fragmented sectors = slow regime propagation.
    - λ₂ → 0 during crisis = graph is about to split = regime transition.
    - λ₂ feeds into Kuramoto K_c estimate: synchronization requires
      K > K_c ∝ 1/λ₂ (loosely — the exact relation is more complex).

    Parameters
    ----------
    adjacency : (N, N) array
        Symmetric non-negative adjacency matrix.

    Returns
    -------
    float
        λ₂ (Fiedler eigenvalue). Always ≥ 0 for valid graphs.
        Returns 0.0 for disconnected or trivially small graphs.

    Invariants:
        INV-SG1: λ₂ ≥ 0 always (Laplacian is positive semi-definite)
        INV-SG2: λ₂ > 0 ⟺ graph is connected
    """
    A = np.asarray(adjacency, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {A.shape}")

    n = A.shape[0]
    if n < 2:
        return 0.0

    # Symmetrise (in case of rounding asymmetry)
    A = 0.5 * (A + A.T)

    # INV-HPC2: non-negative adjacency
    A = np.maximum(A, 0.0)
    np.fill_diagonal(A, 0.0)

    # Combinatorial Laplacian: L = D - A
    degrees = A.sum(axis=1)
    L = np.diag(degrees) - A

    # Eigenvalues (real, sorted ascending for symmetric positive semi-definite)
    # INV-HPC2: protect against non-finite entries
    if not np.all(np.isfinite(L)):
        return 0.0

    eigenvalues = np.linalg.eigvalsh(L)

    # λ₂ = second smallest eigenvalue
    # eigenvalues[0] ≈ 0 (constant eigenvector), eigenvalues[1] = λ₂
    # INV-SG1: λ₂ ≥ 0 by positive semi-definiteness of L
    lambda_2 = float(max(0.0, eigenvalues[1]))

    return lambda_2
