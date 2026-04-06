from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
import networkx as nx

from core.metrics.aperiodic import aperiodic_slope
from core.physics.lyapunov_exponent import maximal_lyapunov_exponent


class InvariantViolation(RuntimeError):
    """Raised when an invariant is violated by domain application."""


class SSI:
    """Simple safety switch interface for domain-scoped execution."""

    def apply(self, *, domain: str) -> None:
        if domain.upper() == "INTERNAL":
            raise InvariantViolation("SSI.apply(domain=INTERNAL) is forbidden")


@dataclass(frozen=True)
class PSDGammaEstimator:
    fs: float = 1.0

    def compute(self, data: np.ndarray) -> float:
        slope = aperiodic_slope(data, fs=self.fs)
        g=float(np.clip(-slope, -5.0, 5.0))
        return g


@dataclass(frozen=True)
class KernelBundle:
    gamma_estimator: PSDGammaEstimator
    ricci: "FormanRicciKernel"


@dataclass(frozen=True)
class FormanRicciKernel:
    correlation_threshold: float = 0.2

    def compute_mean(self, returns: pd.DataFrame) -> float:
        corr = returns.corr().fillna(0.0)
        graph = nx.Graph()
        graph.add_nodes_from(corr.columns)
        for i, u in enumerate(corr.columns):
            for j, v in enumerate(corr.columns):
                if j <= i:
                    continue
                w = float(abs(corr.iloc[i, j]))
                if w >= self.correlation_threshold:
                    graph.add_edge(u, v, weight=w)

        if graph.number_of_edges() == 0:
            return 0.0

        curvatures: list[float] = []
        for u, v, data in graph.edges(data=True):
            w = float(data.get("weight", 1.0))
            deg_u = max(1, graph.degree(u))
            deg_v = max(1, graph.degree(v))
            # Simple Forman-Ricci approximation for weighted undirected graphs.
            kappa = w * ((1.0 / np.sqrt(deg_u)) + (1.0 / np.sqrt(deg_v)) - 1.0)
            curvatures.append(float(kappa))
        return float(np.mean(curvatures))


class GeoSyncAdapter:
    """Thread-safe signal adapter bound to GeoSync physics kernels."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self._lock = threading.RLock()
        self._sequence: dict[str, int] = {}
        self._last_transition_ns: dict[str, int] = {}
        self._last_known_good: dict[str, dict[str, Any]] = {}
        self._history: dict[str, list[float]] = {}
        self._ssi = SSI()

        self._config = self._load_config(config_path)
        self.instruments = list(self._config.get("instruments", []))
        if not self.instruments:
            raise ValueError("GeoSync config must provide at least one instrument")

        self._window_size = int(self._config.get("window_size", 300))
        self._cycle_step = int(self._config.get("cycle_step", 60))
        self._engine = self._load_engine()

    def _load_config(self, config_path: str | Path | None) -> dict[str, Any]:
        if config_path is not None:
            path = Path(config_path)
        else:
            path = Path("conf/coherence_bridge.yaml")

        if not path.exists():
            return {
                "instruments": ["EURUSD"],
                "window_size": 300,
                "cycle_step": 60,
            }

        data = yaml.safe_load(path.read_text()) or {}
        return data

    def _load_engine(self) -> KernelBundle:
        return KernelBundle(
            gamma_estimator=PSDGammaEstimator(fs=1.0),
            ricci=FormanRicciKernel(correlation_threshold=0.2),
        )

    def update_tick(self, instrument: str, price: float) -> None:
        series = self._history.setdefault(instrument, [])
        series.append(float(price))
        if len(series) > self._window_size * 4:
            del series[: len(series) - self._window_size * 4]

    def get_signal(self, instrument: str) -> dict[str, Any] | None:
        if instrument not in self.instruments:
            raise KeyError(f"Unknown instrument {instrument!r}; expected one of {self.instruments}")

        series = np.asarray(self._history.get(instrument, []), dtype=float)
        now_ns = time.time_ns()

        if series.size < self._window_size or series.size % self._cycle_step != 0:
            with self._lock:
                cached = self._last_known_good.get(instrument)
                if cached is None:
                    return None
                out = dict(cached)
                out["timestamp_ns"] = now_ns
                self._last_known_good[instrument] = out
                return out

        signal = self._compute_signal(instrument, series[-self._window_size :], now_ns=now_ns)
        with self._lock:
            self._last_known_good[instrument] = dict(signal)
        return signal

    def _compute_signal(self, instrument: str, window: np.ndarray, *, now_ns: int) -> dict[str, Any]:
        returns = np.diff(np.log(np.clip(window, 1e-12, None)))
        psd_gamma=self._engine.gamma_estimator.compute(returns)

        phases = self._estimate_phases(returns)
        order_parameter_R = float(np.clip(np.abs(np.mean(np.exp(1j * phases))), 0.0, 1.0))

        ricci_curvature = self._compute_ricci(returns)
        lyapunov_max = float(maximal_lyapunov_exponent(returns.astype(np.float64), dt=1.0))

        signal_strength = float(np.clip(self._phase_asymmetry(phases), -1.0, 1.0))
        risk_scalar = float(max(0.0, 1.0 - abs(psd_gamma - 1.0)))

        regime, regime_confidence = self._classify_regime(
            order_parameter_R=order_parameter_R,
            ricci_curvature=ricci_curvature,
            lyapunov_max=lyapunov_max,
            signal_strength=signal_strength,
        )

        with self._lock:
            prev_regime = self._last_known_good.get(instrument, {}).get("regime")
            if prev_regime != regime:
                self._last_transition_ns[instrument] = now_ns
            t0 = self._last_transition_ns.setdefault(instrument, now_ns)
            regime_duration_s = max(0.0, (now_ns - t0) / 1e9)

            next_seq = self._sequence.get(instrument, 0) + 1
            self._sequence[instrument] = next_seq

        signal = {
            "instrument": instrument,
            "timestamp_ns": now_ns,
            "gamma": float(psd_gamma),
            "order_parameter_R": order_parameter_R,
            "ricci_curvature": float(ricci_curvature),
            "lyapunov_max": float(lyapunov_max),
            "regime": regime,
            "regime_confidence": float(np.clip(regime_confidence, 0.0, 1.0)),
            "regime_duration_s": float(regime_duration_s),
            "signal_strength": signal_strength,
            "risk_scalar": risk_scalar,
            "sequence_number": int(next_seq),
        }
        return signal

    @staticmethod
    def _estimate_phases(returns: np.ndarray) -> np.ndarray:
        centered = returns - np.mean(returns)
        shifted = np.roll(centered, 1)
        shifted[0] = centered[0]
        return np.arctan2(centered, shifted + 1e-12)

    def _compute_ricci(self, returns: np.ndarray) -> float:
        cols = ["a", "b", "c"]
        chunks = np.array_split(returns, 3)
        frame = pd.DataFrame(
            {c: np.pad(chunk, (0, max(0, max(map(len, chunks)) - len(chunk))), mode="edge") for c, chunk in zip(cols, chunks)},
        )
        return float(self._engine.ricci.compute_mean(frame))

    @staticmethod
    def _phase_asymmetry(phases: np.ndarray) -> float:
        x = phases - np.mean(phases)
        denom = np.std(x) ** 3 + 1e-12
        skew = float(np.mean(x**3) / denom)
        return float(np.tanh(skew / 3.0))

    @staticmethod
    def _classify_regime(
        *,
        order_parameter_R: float,
        ricci_curvature: float,
        lyapunov_max: float,
        signal_strength: float,
    ) -> tuple[str, float]:
        score_coherent = 0.6 * order_parameter_R + 0.2 * (1.0 - max(lyapunov_max, 0.0)) + 0.2 * max(0.0, ricci_curvature)
        score_decoherent = 0.55 * (1.0 - order_parameter_R) + 0.25 * max(lyapunov_max, 0.0) + 0.2 * max(0.0, -ricci_curvature)
        score_critical = 0.6 * (1.0 - abs(signal_strength)) + 0.4 * (1.0 - abs(order_parameter_R - 0.5) * 2.0)
        score_metastable = 1.0 - abs(score_coherent - score_decoherent)

        scores = {
            "COHERENT": max(0.0, score_coherent),
            "DECOHERENT": max(0.0, score_decoherent),
            "CRITICAL": max(0.0, score_critical),
            "METASTABLE": max(0.0, score_metastable),
        }
        total = sum(scores.values()) + 1e-12
        probs = {k: v / total for k, v in scores.items()}
        regime = max(probs, key=probs.get)
        return regime, probs[regime]
