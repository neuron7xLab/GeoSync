# SPDX-License-Identifier: MIT
"""T6 — Graph Diffusion Engine (Fokker-Planck, NOT Maxwell literally).

Information propagation as diffusion on correlation network:

    ∂ρ/∂t = -L·ρ     (graph diffusion equation)

where:
    L = D - A          (graph Laplacian, D = degree matrix)
    D_ij = D_0·exp(κ_ij)  (curvature-dependent diffusion tensor)
    ρ(x,t) = probability density of price state

Solution:
    ρ(t) = exp(-L·t) · ρ(0)    via scipy.linalg.expm

This IS Maxwell-inspired (wave + diffusion) but physically grounded:
    - reduces to heat equation for κ=0
    - reduces to drift equation for D=0
    - coupling to Ricci: information spreads faster on positively curved
      edges (validated: Sandhu et al. 2016, "Graph curvature for
      differentiating cancer networks")

Implementation: graph Laplacian discretisation.
Matrix exponential via Padé approximation (scipy.linalg.expm).

AUDIT NOTE on complexity:
    expm is O(n³). Tractable for n≤500 assets.
    For n>500: use Krylov subspace method (not implemented here,
    documented as future optimisation target for rust/src/graph_diffusion.rs).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm


class GraphDiffusionEngine:
    """Diffusion propagation on curvature-weighted graphs.

    Parameters
    ----------
    D_0 : float
        Base diffusion coefficient (default 1.0).
    """

    def __init__(self, D_0: float = 1.0) -> None:
        if D_0 <= 0:
            raise ValueError(f"D_0 must be > 0, got {D_0}")
        self._D_0 = D_0

    @property
    def D_0(self) -> float:
        return self._D_0

    def build_laplacian(
        self,
        adjacency: NDArray[np.float64],
        curvature: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Build graph Laplacian L = D - W.

        If curvature is provided, edge weights are scaled:
            W_ij = A_ij · D_0 · exp(κ_ij)

        Positive curvature → stronger coupling → faster diffusion.
        Negative curvature → weaker coupling → slower diffusion.

        Parameters
        ----------
        adjacency : (N, N) adjacency/weight matrix.
        curvature : (N, N) Ollivier-Ricci curvature matrix (optional).

        Returns
        -------
        (N, N) graph Laplacian. Eigenvalues are non-negative for
        undirected graphs (verified by construction).
        """
        A = np.asarray(adjacency, dtype=np.float64)
        n = A.shape[0]
        if A.shape != (n, n):
            raise ValueError(f"adjacency must be square, got {A.shape}")

        if curvature is not None:
            kappa = np.asarray(curvature, dtype=np.float64)
            if kappa.shape != (n, n):
                raise ValueError(
                    f"curvature must match adjacency: {kappa.shape} vs {A.shape}"
                )
            W = A * self._D_0 * np.exp(kappa)
        else:
            W = A * self._D_0

        # Ensure symmetry
        W = 0.5 * (W + W.T)
        np.fill_diagonal(W, 0.0)

        # Degree matrix
        D = np.diag(W.sum(axis=1))
        L = D - W
        return L

    @staticmethod
    def propagate(
        rho_0: NDArray[np.float64],
        L: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """Propagate density ρ(t) = exp(-L·t) · ρ(0).

        Parameters
        ----------
        rho_0 : (N,) initial probability density. Must sum to 1.
        L : (N, N) graph Laplacian.
        t : float, time to propagate.

        Returns
        -------
        (N,) propagated density. Sum is preserved (probability conservation).
        """
        rho_0 = np.asarray(rho_0, dtype=np.float64)
        L = np.asarray(L, dtype=np.float64)

        if t < 0:
            raise ValueError(f"t must be ≥ 0, got {t}")
        if t == 0:
            return rho_0.copy()

        propagator = expm(-L * t)
        rho_t = propagator @ rho_0
        # Ensure non-negativity (numerical)
        rho_t = np.maximum(rho_t, 0.0)
        # Renormalise to preserve total probability
        total = rho_t.sum()
        if total > 0:
            rho_t = rho_t / total
        return rho_t

    @staticmethod
    def volatility_front(
        rho_t: NDArray[np.float64],
        threshold: float = 0.1,
        asset_names: list[str] | None = None,
    ) -> list[str | int]:
        """Identify assets at the diffusion wavefront.

        Assets with ρ_i(t) > threshold are at the front of
        information/volatility propagation.

        Parameters
        ----------
        rho_t : (N,) current density.
        threshold : float, density threshold.
        asset_names : optional list of asset identifiers.

        Returns
        -------
        List of asset identifiers (names or indices) above threshold.
        """
        rho = np.asarray(rho_t, dtype=np.float64)
        indices = np.where(rho > threshold)[0]
        if asset_names is not None:
            return [asset_names[i] for i in indices]
        return indices.tolist()

    def laplacian_eigenvalues(
        self, L: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute eigenvalues of graph Laplacian.

        For valid Laplacians:
        - All eigenvalues ≥ 0
        - Smallest eigenvalue = 0 (connected graph has multiplicity 1)
        - Second smallest = algebraic connectivity (Fiedler value)
        """
        eigenvalues = np.linalg.eigvalsh(L)
        return eigenvalues


__all__ = ["GraphDiffusionEngine"]
