# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Ricci temporal calibrator — honest empirical measurement.

Measures actual κ → dislocation temporal offset from data.
No post-hoc claims. If not predictive, says so explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibrationResult:
    """Empirical κ → dislocation temporal offset measurement."""

    offset_bars: float  # median (negative = lagging)
    offset_p5: float  # 5th percentile
    offset_p95: float  # 95th percentile
    is_predictive: bool  # p-value < 0.05 on lead > 0
    p_value: float
    n_events: int
    recommendation: str  # USE / CAUTION / REMOVE_CLAIM
    honest_statement: str


class RicciTemporalCalibrator:
    """Empirical calibration of κ → dislocation temporal offset.

    Method:
      1. Find κ < threshold events
      2. For each: find next |returns| > dislocation_sigma × σ
      3. Measure temporal offset distribution
      4. Test if median offset > 0 (one-sided sign test)
    """

    def calibrate(
        self,
        kappa_series: np.ndarray,
        returns_series: np.ndarray,
        kappa_threshold: float = -0.3,
        dislocation_sigma: float = 2.0,
        max_horizon_bars: int = 30,
    ) -> CalibrationResult:
        """Calibrate temporal offset from κ and returns series."""
        kappa = np.asarray(kappa_series, dtype=np.float64)
        returns = np.asarray(returns_series, dtype=np.float64)

        n = min(len(kappa), len(returns))
        if n < 50:
            return self._no_data()

        kappa = kappa[:n]
        returns = returns[:n]

        # Dislocation threshold: |return| > dislocation_sigma × σ
        sigma = float(np.std(returns[np.isfinite(returns)]))
        if sigma < 1e-12:
            return self._no_data()
        dis_thresh = dislocation_sigma * sigma

        # Find κ < threshold events
        kappa_events = np.where(kappa < kappa_threshold)[0]
        if len(kappa_events) == 0:
            return self._no_data()

        # Measure temporal offsets
        offsets: list[float] = []
        for idx in kappa_events:
            # Search forward for dislocation
            found = False
            for horizon in range(1, min(max_horizon_bars + 1, n - idx)):
                if abs(returns[idx + horizon]) > dis_thresh:
                    offsets.append(float(horizon))
                    found = True
                    break
            if not found:
                offsets.append(float(max_horizon_bars))  # censored

        if len(offsets) < 3:
            return self._no_data()

        lt = np.array(offsets)
        median_lt = float(np.median(lt))
        p5 = float(np.percentile(lt, 5))
        p95 = float(np.percentile(lt, 95))

        # One-sided sign test: H0: median ≤ 0, H1: median > 0
        # Since offsets are always > 0 by construction,
        # test if significantly below max_horizon (not censored)
        n_events = len(lt)
        n_uncensored = int(np.sum(lt < max_horizon_bars))
        # Binomial test: are most events uncensored?
        if n_events > 0:
            # Approximate p-value: if >50% uncensored, likely predictive
            from scipy.stats import binomtest  # noqa: PLC0415

            try:
                p_value = float(
                    binomtest(n_uncensored, n_events, 0.5, alternative="greater").pvalue
                )
            except Exception:
                p_value = 1.0
        else:
            p_value = 1.0

        # Shuffle test: compare with random κ timing
        rng_shuffle = np.random.Generator(np.random.PCG64(42))
        shuffle_medians: list[float] = []
        for _ in range(100):
            shuffled_idx = rng_shuffle.permutation(n)
            shuffled_kappa = kappa[shuffled_idx]
            shuffled_events = np.where(shuffled_kappa < kappa_threshold)[0]
            s_lts: list[float] = []
            for si in shuffled_events:
                for h in range(1, min(max_horizon_bars + 1, n - si)):
                    if abs(returns[si + h]) > dis_thresh:
                        s_lts.append(float(h))
                        break
                else:
                    s_lts.append(float(max_horizon_bars))
            if s_lts:
                shuffle_medians.append(float(np.median(s_lts)))

        # Predictive only if real median significantly below shuffle median
        if shuffle_medians:
            shuffle_median = float(np.median(shuffle_medians))
            is_predictive = (
                p_value < 0.05
                and median_lt < shuffle_median * 0.7  # 30% better than random
                and median_lt < max_horizon_bars * 0.8
            )
        else:
            is_predictive = False

        if is_predictive:
            ci_half = (p95 - p5) / 2
            recommendation = "USE"
            honest = (
                f"κ<{kappa_threshold} empirically associated with dislocation "
                f"within {median_lt:.1f}±{ci_half:.1f} bars "
                f"(n={n_events}, p={p_value:.3f}). "
                "Calibrated on provided data — revalidate on live instrument."
            )
        elif n_uncensored > n_events * 0.3:
            recommendation = "CAUTION"
            honest = (
                f"κ<{kappa_threshold} shows weak association with dislocation "
                f"(median={median_lt:.1f} bars, n={n_events}, p={p_value:.3f}). "
                "Use with caution. Calibrate on larger dataset."
            )
        else:
            recommendation = "REMOVE_CLAIM"
            honest = (
                "κ<0 indicates concurrent topology fragility. "
                "Predictive temporal offset NOT established on available data. "
                "Calibrate on real Askar EURUSD tick data before claiming edge."
            )

        return CalibrationResult(
            offset_bars=round(median_lt, 1),
            offset_p5=round(p5, 1),
            offset_p95=round(p95, 1),
            is_predictive=is_predictive,
            p_value=round(p_value, 4),
            n_events=n_events,
            recommendation=recommendation,
            honest_statement=honest,
        )

    def _no_data(self) -> CalibrationResult:
        return CalibrationResult(
            offset_bars=0.0,
            offset_p5=0.0,
            offset_p95=0.0,
            is_predictive=False,
            p_value=1.0,
            n_events=0,
            recommendation="REMOVE_CLAIM",
            honest_statement=(
                "κ<0 indicates concurrent topology fragility. "
                "Predictive temporal offset NOT established on available data. "
                "Calibrate on real Askar EURUSD tick data before claiming edge."
            ),
        )
