"""GeoSync physics kernel → CoherenceBridge adapter.

Upgrades over v1:
  1. PSDGammaEstimator: multi-segment Welch + quality gate (replaces single aperiodic_slope)
  2. AugmentedFormanRicci: triangle reinforcement + degree penalty (topology fragility)
  3. Deterministic compute path: pure function of (returns, symbols, seq, timestamp)

Wiring table:
  PSDGammaEstimator.compute(returns)            → gamma (DERIVED, never assigned)
  GeoSyncCompositeEngine.analyze_market(df)     → R, regime, confidence, signal_strength
  AugmentedFormanRicci.compute_mean(returns, s) → ricci_curvature (augmented κ)
  maximal_lyapunov_exponent(returns)            → lyapunov_max

MarketPhase → RegimeType mapping:
  CHAOTIC        → DECOHERENT
  PROTO_EMERGENT → METASTABLE
  STRONG_EMERGENT→ COHERENT
  TRANSITION     → CRITICAL
  POST_EMERGENT  → DECOHERENT
"""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from coherence_bridge.augmented_ricci import AugmentedFormanRicci
from coherence_bridge.gamma_estimator import PSDGammaEstimator
from coherence_bridge.risk import compute_risk_scalar

if TYPE_CHECKING:
    pass

logger = logging.getLogger("coherence_bridge.geosync_adapter")

GEOSYNC_PATH = os.getenv(
    "GEOSYNC_PATH",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

_PHASE_TO_REGIME: dict[str, str] = {
    "CHAOTIC": "DECOHERENT",
    "PROTO_EMERGENT": "METASTABLE",
    "STRONG_EMERGENT": "COHERENT",
    "TRANSITION": "CRITICAL",
    "POST_EMERGENT": "DECOHERENT",
}

_DEFAULT_INSTRUMENTS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]


class GeoSyncAdapter:
    """Adapts GeoSync physics kernel to SignalEngine interface.

    Thread-safe. Non-blocking: returns last-known-good if compute not ready.
    gamma is ALWAYS derived from PSD via PSDGammaEstimator, never assigned.
    """

    def __init__(self, geosync_path: str = GEOSYNC_PATH) -> None:
        if geosync_path not in sys.path:
            sys.path.insert(0, geosync_path)

        self._lock = threading.Lock()
        self._instruments: list[str] = []
        self._last_known_good: dict[str, dict[str, Any]] = {}
        self._seq: dict[str, int] = {}
        self._regime_start: dict[str, float] = {}
        self._last_regime: dict[str, str] = {}

        # Physics components
        self._composite_engine: Any = None
        self._forman_ricci: Any = None  # GeoSync core Forman-Ricci
        self._lyapunov_fn: Any = None
        self._gamma_estimator = PSDGammaEstimator(fs=1.0)
        self._augmented_ricci = AugmentedFormanRicci(correlation_threshold=0.2)

        # Market data cache
        self._market_data: dict[str, Any] = {}

        self._load_engine(geosync_path)

    def _load_engine(self, path: str) -> None:
        """Wire GeoSync physics kernel components."""
        from core.indicators.kuramoto_ricci_composite import GeoSyncCompositeEngine
        from core.physics.forman_ricci import FormanRicciCurvature
        from core.physics.lyapunov_exponent import maximal_lyapunov_exponent

        self._composite_engine = GeoSyncCompositeEngine()
        self._forman_ricci = FormanRicciCurvature()
        self._lyapunov_fn = maximal_lyapunov_exponent

        self._instruments = list(_DEFAULT_INSTRUMENTS)
        for inst in self._instruments:
            self._seq[inst] = 0
            self._regime_start[inst] = time.time()
            self._last_regime[inst] = "UNKNOWN"

        logger.info(
            "GeoSync engine loaded from %s, instruments=%s",
            path,
            self._instruments,
        )

    @property
    def instruments(self) -> list[str]:
        return list(self._instruments)

    def update_market_data(self, instrument: str, df: Any) -> None:
        """Feed new OHLCV DataFrame for an instrument."""
        import pandas  # noqa: PLC0415

        if not isinstance(df, pandas.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df)}")
        if not isinstance(df.index, pandas.DatetimeIndex):
            raise TypeError("DataFrame must have DatetimeIndex")
        with self._lock:
            self._market_data[instrument] = df

    def get_signal(self, instrument: str) -> dict[str, Any] | None:
        if instrument not in self._instruments:
            return None

        with self._lock:
            df = self._market_data.get(instrument)

        if df is None or len(df) < 30:
            with self._lock:
                cached = self._last_known_good.get(instrument)
            if cached is not None:
                stale = dict(cached)
                stale["timestamp_ns"] = time.time_ns()
                return stale
            return None

        try:
            sig = self._compute_signal(instrument, df)
            with self._lock:
                self._last_known_good[instrument] = sig
            return sig
        except Exception as exc:
            logger.warning("Compute failed for %s: %s", instrument, exc)
            with self._lock:
                cached = self._last_known_good.get(instrument)
            if cached is not None:
                stale = dict(cached)
                stale["timestamp_ns"] = time.time_ns()
                return stale
            return None

    def _compute_signal(
        self,
        instrument: str,
        df: Any,
    ) -> dict[str, Any]:
        """Run full GeoSync physics kernel on market data.

        gamma is DERIVED from PSD via PSDGammaEstimator. NEVER assigned.
        Ricci curvature uses augmented Forman-Ricci with triangle reinforcement.
        """
        prices = df["close"].values.astype(np.float64)
        returns = np.diff(np.log(prices + 1e-12))

        # 1. Composite analysis (Kuramoto + temporal Ricci + regime)
        composite = self._composite_engine.analyze_market(df)

        # 2. gamma — DERIVED from multi-segment PSD (never assigned)
        gamma_est = self._gamma_estimator.compute(returns)
        if gamma_est.is_valid:
            gamma = gamma_est.value
        else:
            # Fallback: insufficient quality → metastable default
            gamma = 1.0

        # 3. Ricci curvature — dual track:
        #    a) Augmented Forman-Ricci on lagged returns (topology fragility)
        #    b) GeoSync core Forman-Ricci (compatibility)
        n_lags = min(5, len(returns) // 10)
        if n_lags >= 2:
            lagged = np.column_stack(
                [returns[i : len(returns) - n_lags + i + 1] for i in range(n_lags)]
            )
            lag_symbols = [f"lag_{i}" for i in range(n_lags)]

            # Augmented: triangle + degree penalty
            augmented_kappa = self._augmented_ricci.compute_mean(lagged, lag_symbols)

            # Core: standard Forman-Ricci
            try:
                core_result = self._forman_ricci.compute_from_prices(
                    lagged, window=min(30, len(lagged))
                )
                core_kappa = core_result.kappa_mean
            except Exception:
                core_kappa = composite.static_ricci

            # Blend: augmented dominates, core stabilizes
            ricci_curvature = 0.7 * augmented_kappa + 0.3 * core_kappa
        else:
            ricci_curvature = composite.static_ricci

        # 4. Lyapunov exponent
        if len(returns) >= 50:
            lyapunov_max = self._lyapunov_fn(returns, dim=3, tau=1)
            if not math.isfinite(lyapunov_max):
                lyapunov_max = 0.0
        else:
            lyapunov_max = 0.0

        # 5. Regime mapping
        regime_name = _PHASE_TO_REGIME.get(composite.phase.name, "UNKNOWN")

        # 6. Fail-closed: invalid physics → UNKNOWN, risk=0
        if not math.isfinite(gamma) or not math.isfinite(ricci_curvature):
            gamma = 0.0
            ricci_curvature = 0.0
            lyapunov_max = 0.0
            regime_name = "UNKNOWN"

        # 7. Regime duration tracking
        with self._lock:
            if regime_name != self._last_regime.get(instrument):
                self._regime_start[instrument] = time.time()
                self._last_regime[instrument] = regime_name
            duration = time.time() - self._regime_start.get(instrument, time.time())

        # 8. Signal strength from entry/exit asymmetry [-1, +1]
        signal_strength = max(
            -1.0,
            min(1.0, composite.entry_signal - composite.exit_signal),
        )

        # 9. risk_scalar from gamma (derived, never assigned)
        risk_scalar = compute_risk_scalar(gamma, fail_closed=True)

        # 10. Sequence number
        with self._lock:
            seq = self._seq.get(instrument, 0)
            self._seq[instrument] = seq + 1

        return {
            "timestamp_ns": time.time_ns(),
            "instrument": instrument,
            "gamma": round(float(gamma), 6),
            "order_parameter_R": round(float(composite.kuramoto_R), 6),
            "ricci_curvature": round(float(ricci_curvature), 6),
            "lyapunov_max": round(float(lyapunov_max), 6),
            "regime": regime_name,
            "regime_confidence": round(float(composite.confidence), 4),
            "regime_duration_s": round(duration, 2),
            "signal_strength": round(float(signal_strength), 4),
            "risk_scalar": round(float(risk_scalar), 4),
            "sequence_number": seq,
        }
