"""Kuramoto-Kelly Adaptive Position Sizing.

Maps the Kuramoto order parameter R (phase synchrony of market returns)
to a Kelly-criterion position fraction. High coherence (trending market)
→ aggressive sizing; chaotic regime → conservative sizing.

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfilt

from core.neuro.signal_bus import NeuroSignalBus


class KuramotoKellyAdapter:
    """Adapts Kelly fraction using Kuramoto order parameter from price returns.

    Parameters
    ----------
    bus : NeuroSignalBus
        Shared signal bus for publishing Kuramoto R.
    floor : float
        Minimum scaling factor applied to kelly_base (default 0.1).
    ceil : float
        Maximum scaling factor applied to kelly_base (default 1.0).
    R_low : float
        Order parameter below which the market is considered chaotic (0.3).
    R_high : float
        Order parameter above which the market is considered coherent (0.7).
    """

    def __init__(
        self,
        bus: NeuroSignalBus,
        floor: float = 0.1,
        ceil: float = 1.0,
        R_low: float = 0.3,
        R_high: float = 0.7,
    ) -> None:
        self._bus = bus
        self._floor = floor
        self._ceil = ceil
        self._R_low = R_low
        self._R_high = R_high

    # ── Public API ────────────────────────────────────────────────────

    def compute_kelly_fraction(self, kelly_base: float, prices: np.ndarray) -> float:
        """Compute coherence-adjusted Kelly fraction from price series.

        Parameters
        ----------
        kelly_base : float
            Raw Kelly fraction before adjustment.
        prices : np.ndarray
            Price series (at least 3 elements).

        Returns
        -------
        float
            Adjusted Kelly fraction ∈ [floor * kelly_base, kelly_base].
        """
        returns = np.diff(prices) / prices[:-1]
        R = self.compute_order_parameter(returns)
        self._bus.publish_kuramoto(R)

        R_norm = np.clip(
            (R - self._R_low) / (self._R_high - self._R_low), 0.0, 1.0
        )
        scale = self._floor + (self._ceil - self._floor) * R_norm
        return kelly_base * scale

    def compute_order_parameter(self, returns: np.ndarray) -> float:
        """Compute Kuramoto order parameter R from return series.

        Decomposes the signal into multiple frequency-band oscillators
        via bandpass filters + Hilbert transform. R is the mean
        phase synchrony across bands:
            R = mean_t |mean_k exp(i * phase_k(t))|

        This captures regime coherence: when all frequency components
        are in phase (trending market), R is high; when they are
        desynchronised (noisy/chaotic market), R is low.

        Parameters
        ----------
        returns : np.ndarray
            Array of log or simple returns.

        Returns
        -------
        float
            R ∈ [0, 1]. Returns 0.0 for degenerate inputs.
        """
        if returns is None or len(returns) < 2:
            return 0.0

        # Handle NaN / constant input
        if np.any(np.isnan(returns)):
            returns = returns[~np.isnan(returns)]
            if len(returns) < 2:
                return 0.0

        if np.std(returns) < 1e-15:
            return 0.0

        n = len(returns)
        n_bands = 5
        fs = 1.0

        # Short series: fall back to phase-velocity coherence
        if n < 30:
            analytic = hilbert(returns)
            phases = np.angle(analytic)
            R = float(np.abs(np.mean(np.exp(1j * phases))))
            return R

        # Multi-band Kuramoto: each band is an oscillator
        band_edges = np.linspace(0.02, 0.48, n_bands + 1)
        phase_matrix = np.zeros((n_bands, n))
        valid_bands = 0

        for i in range(n_bands):
            low, high = band_edges[i], band_edges[i + 1]
            try:
                sos = butter(3, [low, high], btype="band", fs=fs, output="sos")
                filtered = sosfilt(sos, returns)
                if np.std(filtered) < 1e-15:
                    continue
                analytic = hilbert(filtered)
                phase_matrix[valid_bands] = np.angle(analytic)
                valid_bands += 1
            except Exception:
                continue

        if valid_bands < 2:
            # Not enough bands; fall back to single-series
            analytic = hilbert(returns)
            phases = np.angle(analytic)
            R = float(np.abs(np.mean(np.exp(1j * phases))))
            return R

        # R(t) = |mean_k exp(i * phase_k(t))| over valid bands
        phases_used = phase_matrix[:valid_bands]
        R_t = np.abs(np.mean(np.exp(1j * phases_used), axis=0))
        R = float(np.mean(R_t))
        return R

    def classify_regime(self, R: float) -> str:
        """Classify market regime from order parameter.

        Returns
        -------
        str
            "chaotic" | "transitional" | "coherent"
        """
        if R < 0.3:
            return "chaotic"
        elif R < 0.7:
            return "transitional"
        else:
            return "coherent"
