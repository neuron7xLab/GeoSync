# SPDX-License-Identifier: MIT
"""T4 — Higher-order Kuramoto with triadic interaction.

Equation:
    dθ_i/dt = ω_i
              + σ₁ · Σ_j A_ij · sin(θ_j - θ_i)           [pairwise]
              + σ₂ · Σ_{j,k ∈ Δ(i)} sin(2θ_j - θ_k - θ_i) [triadic]

where Δ(i) = set of triangles containing node i.

The triadic term captures 3-body interactions:
    - When three assets form a triangle in the correlation network,
      their synchronization dynamics couple through higher-order effects
    - Standard pairwise Kuramoto misses cluster-level coherence
    - Triadic coupling can detect emergent clusters invisible to pairwise

Simplicial structure:
    Build 2-skeleton of clique complex from correlation matrix:
    1. Nodes = assets
    2. Edges (1-simplices) = |ρ_ij| > threshold
    3. Triangles (2-simplices) = cliques of size 3

No XGI dependency required: we build triangle list directly from
adjacency matrix using A³ diagonal trick.

References:
    Skardal & Arenas "Higher order interactions" Commun. Phys. (2020)
    Millán et al. "Explosive higher-order Kuramoto" PRL (2020)
    Bick et al. "What are higher-order networks?" SIAM Rev. (2023)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class HigherOrderKuramotoResult:
    """Result of higher-order Kuramoto simulation."""

    phases: NDArray[np.float64]        # (steps+1, N)
    order_parameter: NDArray[np.float64]  # (steps+1,)
    time: NDArray[np.float64]          # (steps+1,)
    n_triangles: int
    triadic_contribution: NDArray[np.float64]  # (steps+1,) magnitude of σ₂ term


def find_triangles(adj: NDArray[np.bool_]) -> list[tuple[int, int, int]]:
    """Find all triangles in undirected graph.

    Uses A³ trace / 6 for counting, but returns explicit list.
    O(N²·Δ) where Δ = max degree.

    Parameters
    ----------
    adj : (N, N) boolean adjacency matrix.

    Returns
    -------
    List of (i, j, k) triangles with i < j < k.
    """
    n = adj.shape[0]
    triangles: list[tuple[int, int, int]] = []
    for i in range(n):
        neighbors_i = set(np.where(adj[i])[0])
        for j in neighbors_i:
            if j <= i:
                continue
            neighbors_j = set(np.where(adj[j])[0])
            common = neighbors_i & neighbors_j
            for k in common:
                if k > j:
                    triangles.append((i, j, k))
    return triangles


def build_triangle_index(
    n: int, triangles: list[tuple[int, int, int]]
) -> dict[int, list[tuple[int, int]]]:
    """Build per-node index of triangle partners.

    Returns dict mapping node i to list of (j, k) pairs where
    (i, j, k) forms a triangle.
    """
    index: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n)}
    for i, j, k in triangles:
        index[i].append((j, k))
        index[j].append((i, k))
        index[k].append((i, j))
    return index


class HigherOrderKuramotoEngine:
    """Higher-order Kuramoto with pairwise + triadic coupling.

    Parameters
    ----------
    sigma1 : float
        Pairwise coupling strength (default 1.0).
    sigma2 : float
        Triadic coupling strength (default 0.5).
    dt : float
        Integration time step (default 0.01).
    steps : int
        Number of integration steps (default 1000).
    correlation_threshold : float
        Threshold for edge/triangle construction (default 0.3).
    """

    def __init__(
        self,
        sigma1: float = 1.0,
        sigma2: float = 0.5,
        dt: float = 0.01,
        steps: int = 1000,
        correlation_threshold: float = 0.3,
    ) -> None:
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        if steps < 1:
            raise ValueError(f"steps must be ≥ 1, got {steps}")
        self._sigma1 = sigma1
        self._sigma2 = sigma2
        self._dt = dt
        self._steps = steps
        self._corr_thresh = correlation_threshold

    def _build_structures(
        self, corr: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], list[tuple[int, int, int]], dict]:
        """Build adjacency and triangle structures from correlation."""
        adj_bool = np.abs(corr) > self._corr_thresh
        np.fill_diagonal(adj_bool, False)

        # Weighted adjacency
        adj = np.abs(corr) * adj_bool.astype(float)
        np.fill_diagonal(adj, 0.0)

        triangles = find_triangles(adj_bool)
        tri_index = build_triangle_index(corr.shape[0], triangles)

        return adj, triangles, tri_index

    def _dtheta_dt(
        self,
        theta: NDArray[np.float64],
        omega: NDArray[np.float64],
        adj: NDArray[np.float64],
        tri_index: dict[int, list[tuple[int, int]]],
    ) -> tuple[NDArray[np.float64], float]:
        """Compute RHS of higher-order Kuramoto ODE.

        Returns (dθ/dt, triadic_magnitude).
        """
        n = theta.shape[0]

        # Pairwise coupling: σ₁ · Σ_j A_ij · sin(θ_j - θ_i)
        diff = theta[np.newaxis, :] - theta[:, np.newaxis]
        pairwise = self._sigma1 * (adj * np.sin(diff)).sum(axis=1)

        # Triadic coupling: σ₂ · Σ_{j,k ∈ Δ(i)} sin(2θ_j - θ_k - θ_i)
        triadic = np.zeros(n, dtype=np.float64)
        for i in range(n):
            for j, k in tri_index.get(i, []):
                triadic[i] += np.sin(2 * theta[j] - theta[k] - theta[i])
        triadic *= self._sigma2

        triadic_mag = float(np.sqrt(np.sum(triadic ** 2)))

        return omega + pairwise + triadic, triadic_mag

    def _rk4_step(
        self,
        theta: NDArray[np.float64],
        omega: NDArray[np.float64],
        adj: NDArray[np.float64],
        tri_index: dict,
    ) -> tuple[NDArray[np.float64], float]:
        """RK4 step with triadic coupling."""
        dt = self._dt

        k1, m1 = self._dtheta_dt(theta, omega, adj, tri_index)
        k2, m2 = self._dtheta_dt(theta + 0.5 * dt * k1, omega, adj, tri_index)
        k3, m3 = self._dtheta_dt(theta + 0.5 * dt * k2, omega, adj, tri_index)
        k4, m4 = self._dtheta_dt(theta + dt * k3, omega, adj, tri_index)

        theta_new = theta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return theta_new, (m1 + m2 + m3 + m4) / 4.0

    def run(
        self,
        corr: NDArray[np.float64],
        omega: NDArray[np.float64] | None = None,
        theta0: NDArray[np.float64] | None = None,
        seed: int = 42,
    ) -> HigherOrderKuramotoResult:
        """Run higher-order Kuramoto simulation.

        Parameters
        ----------
        corr : (N, N) correlation matrix.
        omega : (N,) natural frequencies (default: random).
        theta0 : (N,) initial phases (default: random).
        seed : int, RNG seed.

        Returns
        -------
        HigherOrderKuramotoResult.
        """
        corr_arr: NDArray[np.float64] = np.asarray(corr, dtype=np.float64)
        n = corr_arr.shape[0]

        rng = np.random.default_rng(seed)
        omega_arr: NDArray[np.float64] = (
            np.asarray(rng.standard_normal(n), dtype=np.float64)
            if omega is None
            else np.asarray(omega, dtype=np.float64)
        )
        theta0_arr: NDArray[np.float64] = (
            np.asarray(rng.uniform(0, 2 * np.pi, n), dtype=np.float64)
            if theta0 is None
            else np.asarray(theta0, dtype=np.float64)
        )

        adj, triangles, tri_index = self._build_structures(corr_arr)

        phases = np.empty((self._steps + 1, n), dtype=np.float64)
        R_arr = np.empty(self._steps + 1, dtype=np.float64)
        triadic_arr = np.empty(self._steps + 1, dtype=np.float64)
        time_arr = np.arange(self._steps + 1, dtype=np.float64) * self._dt

        theta = theta0_arr.copy()
        phases[0] = theta
        R_arr[0] = self._order_parameter(theta)
        triadic_arr[0] = 0.0

        for k in range(self._steps):
            theta, tri_mag = self._rk4_step(theta, omega_arr, adj, tri_index)
            phases[k + 1] = theta
            R_arr[k + 1] = self._order_parameter(theta)
            triadic_arr[k + 1] = tri_mag

        return HigherOrderKuramotoResult(
            phases=phases,
            order_parameter=R_arr,
            time=time_arr,
            n_triangles=len(triangles),
            triadic_contribution=triadic_arr,
        )

    @staticmethod
    def _order_parameter(theta: NDArray[np.float64]) -> float:
        """R = |mean(exp(iθ))|."""
        z = np.exp(1j * theta).mean()
        return float(np.clip(np.abs(z), 0.0, 1.0))

    def run_from_prices(
        self,
        prices: NDArray[np.float64],
        window: int = 60,
        seed: int = 42,
    ) -> HigherOrderKuramotoResult:
        """Convenience: build correlation from prices, then run."""
        prices = np.asarray(prices, dtype=np.float64)
        returns = np.diff(prices, axis=0) / np.maximum(np.abs(prices[:-1]), 1e-12)
        tail = returns[-window:] if returns.shape[0] > window else returns

        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(tail, rowvar=False)
        corr_arr: NDArray[np.float64] = np.asarray(
            np.nan_to_num(corr, nan=0.0), dtype=np.float64
        )

        return self.run(corr_arr, seed=seed)


__all__ = [
    "HigherOrderKuramotoEngine",
    "HigherOrderKuramotoResult",
    "find_triangles",
    "build_triangle_index",
]
