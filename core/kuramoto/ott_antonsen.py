# SPDX-License-Identifier: MIT
"""Ott-Antonsen dimensionality reduction for Kuramoto dynamics.

The most important theoretical advance in synchronization theory since
Kuramoto (1975). Ott & Antonsen (2008) showed that for Lorentzian-
distributed natural frequencies, the infinite-N Kuramoto model reduces
to a SINGLE complex ODE for the order parameter z(t) = R(t)·exp(iΨ(t)):

    dz/dt = -(Δ + iω₀)·z + (K/2)·(z̄ − z·|z|²)

where:
    Δ = half-width of the Lorentzian frequency distribution
    ω₀ = center frequency (mean natural frequency)
    K = coupling strength
    z = R·exp(iΨ) = complex order parameter

This means:
    - N-body → 1-body: instead of N coupled ODEs, ONE complex ODE
    - EXACT in thermodynamic limit (N → ∞)
    - Analytical K_c = 2Δ (critical coupling from ODE bifurcation)
    - Analytical R_∞ = √(1 − 2Δ/K) for K > K_c (steady-state R)
    - 10,000x speedup over N-body simulation for R(t) trajectory
    - Bifurcation diagram in closed form

For GeoSync:
    - Feed Δ, ω₀ from market data → get R(t) trajectory analytically
    - Compare analytical R(t) with empirical R(t) → model validation
    - Predict K_c from estimated coupling → regime transition forecast
    - Stability of synchronized state → Kelly confidence scaling

References:
    Ott & Antonsen (2008). "Low dimensional behavior of large systems
    of globally coupled oscillators." Chaos, 18(3), 037113.

    Ott & Antonsen (2009). "Long time evolution of phase oscillator
    systems." Chaos, 19(2), 023117.

Invariants:
    INV-OA1: |z(t)| ≤ 1 always (order parameter bound)
    INV-OA2: K > 2Δ ⟹ R_∞ = √(1 − 2Δ/K) (analytical steady state)
    INV-OA3: K < 2Δ ⟹ R → 0 (subcritical decay, matches INV-K2)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class OttAntonsenResult:
    """Result of Ott-Antonsen mean-field integration."""

    time: NDArray[np.float64]
    z: NDArray[np.complex128]  # complex order parameter z(t)
    R: NDArray[np.float64]  # |z(t)| = order parameter magnitude
    psi: NDArray[np.float64]  # arg(z(t)) = mean phase
    K_c: float  # critical coupling = 2Δ
    R_steady: float  # analytical R_∞ (0 if subcritical)
    is_supercritical: bool  # K > K_c


@dataclass(frozen=True, slots=True)
class ChimeraReport:
    """Chimera state detection report."""

    global_R: float  # global order parameter
    sector_R: NDArray[np.float64]  # per-sector order parameter
    sector_labels: list[str]  # sector names
    is_chimera: bool  # True if some sectors sync, others don't
    sync_sectors: list[str]  # sectors with R > threshold
    desync_sectors: list[str]  # sectors with R < threshold
    chimera_index: float  # std(sector_R) / mean(sector_R)


class OttAntonsenEngine:
    """Exact mean-field reduction of Kuramoto dynamics.

    Integrates the Ott-Antonsen ODE for z(t) instead of N coupled
    phase equations. Produces the exact R(t) trajectory in the
    thermodynamic limit (N → ∞).

    Usage::

        engine = OttAntonsenEngine(K=2.0, delta=0.5, omega0=0.0)
        result = engine.integrate(T=50.0, dt=0.01, R0=0.01)
        print(f"R_∞ = {result.R[-1]:.4f}, K_c = {result.K_c:.4f}")
    """

    def __init__(
        self,
        K: float,
        delta: float,
        omega0: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        K : float
            Global coupling strength.
        delta : float
            Half-width of Lorentzian frequency distribution.
            g(ω) = (Δ/π) / ((ω − ω₀)² + Δ²)
        omega0 : float
            Center of frequency distribution (default 0).
        """
        if delta <= 0:
            raise ValueError(f"delta must be > 0, got {delta}")
        self.K = float(K)
        self.delta = float(delta)
        self.omega0 = float(omega0)

    @property
    def K_c(self) -> float:
        """Critical coupling K_c = 2Δ (Ott-Antonsen bifurcation)."""
        return 2.0 * self.delta

    @property
    def R_steady(self) -> float:
        """Analytical steady-state R for K > K_c.

        R_∞ = √(1 − 2Δ/K) = √(1 − K_c/K)

        Returns 0.0 if K ≤ K_c (subcritical).
        """
        if self.K <= self.K_c:
            return 0.0
        return math.sqrt(1.0 - self.K_c / self.K)

    def _dz_dt(self, z: complex) -> complex:
        """Ott-Antonsen ODE right-hand side.

        dz/dt = -(Δ + iω₀)·z + (K/2)·(z̄ − z·|z|²)
        """
        z_conj = z.conjugate()
        z_abs_sq = z.real**2 + z.imag**2
        return -(self.delta + 1j * self.omega0) * z + (self.K / 2.0) * (z_conj - z * z_abs_sq)

    def integrate(
        self,
        T: float = 50.0,
        dt: float = 0.01,
        R0: float = 0.01,
        psi0: float = 0.0,
    ) -> OttAntonsenResult:
        """Integrate the Ott-Antonsen ODE via RK4.

        Parameters
        ----------
        T : float
            Total integration time.
        dt : float
            Time step.
        R0 : float
            Initial order parameter magnitude (default: small perturbation).
        psi0 : float
            Initial mean phase (default: 0).

        Returns
        -------
        OttAntonsenResult
        """
        n_steps = int(T / dt)
        time = np.linspace(0.0, T, n_steps + 1)
        z_arr = np.empty(n_steps + 1, dtype=np.complex128)

        # Initial condition
        # INV-OA1: |z| ≤ 1
        R0 = min(R0, 1.0)
        z = complex(R0 * math.cos(psi0), R0 * math.sin(psi0))
        z_arr[0] = z

        # RK4 integration of the complex ODE
        for step in range(n_steps):
            k1 = self._dz_dt(z)
            k2 = self._dz_dt(z + 0.5 * dt * k1)
            k3 = self._dz_dt(z + 0.5 * dt * k2)
            k4 = self._dz_dt(z + dt * k3)
            z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # INV-OA1: project back to unit disk if numerical drift
            if abs(z) > 1.0:
                z = z / abs(z)

            z_arr[step + 1] = z

        R = np.abs(z_arr).astype(np.float64)
        psi = np.angle(z_arr).astype(np.float64)

        return OttAntonsenResult(
            time=time,
            z=z_arr,
            R=R,
            psi=psi,
            K_c=self.K_c,
            R_steady=self.R_steady,
            is_supercritical=self.K > self.K_c,
        )


def detect_chimera(
    phases: NDArray[np.float64],
    sector_assignments: NDArray[np.int64],
    sector_labels: list[str] | None = None,
    sync_threshold: float = 0.7,
    desync_threshold: float = 0.3,
) -> ChimeraReport:
    """Detect chimera states: partial synchronization across sectors.

    A chimera state occurs when some groups of oscillators synchronize
    while others remain incoherent — "synchronization coexists with
    disorder." In financial markets: tech stocks lock while energy
    trades independently.

    Parameters
    ----------
    phases : (N,) array
        Current phase of each oscillator.
    sector_assignments : (N,) array of int
        Sector index for each oscillator (0-indexed).
    sector_labels : list[str], optional
        Human-readable sector names.
    sync_threshold : float
        R above this = "synchronized" (default 0.7).
    desync_threshold : float
        R below this = "desynchronized" (default 0.3).

    Returns
    -------
    ChimeraReport
    """
    phases = np.asarray(phases, dtype=np.float64)
    sectors = np.asarray(sector_assignments, dtype=np.int64)
    n_sectors = int(sectors.max()) + 1

    if sector_labels is None:
        sector_labels = [f"sector_{i}" for i in range(n_sectors)]

    # Global order parameter
    global_z = np.mean(np.exp(1j * phases))
    global_R = float(np.abs(global_z))

    # Per-sector order parameter
    sector_R = np.zeros(n_sectors, dtype=np.float64)
    for s in range(n_sectors):
        mask = sectors == s
        if mask.sum() > 0:
            sector_z = np.mean(np.exp(1j * phases[mask]))
            sector_R[s] = float(np.abs(sector_z))

    # Chimera detection: some sectors sync, others desync
    sync_sectors = [sector_labels[i] for i in range(n_sectors) if sector_R[i] > sync_threshold]
    desync_sectors = [sector_labels[i] for i in range(n_sectors) if sector_R[i] < desync_threshold]
    is_chimera = len(sync_sectors) > 0 and len(desync_sectors) > 0

    # Chimera index: std/mean of sector R values (high = chimera-like)
    mean_R = float(np.mean(sector_R))
    chimera_index = float(np.std(sector_R) / mean_R) if mean_R > 1e-10 else 0.0

    return ChimeraReport(
        global_R=global_R,
        sector_R=sector_R,
        sector_labels=sector_labels,
        is_chimera=is_chimera,
        sync_sectors=sync_sectors,
        desync_sectors=desync_sectors,
        chimera_index=chimera_index,
    )
