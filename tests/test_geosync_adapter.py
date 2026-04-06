from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from coherence_bridge.geosync_adapter import GeoSyncAdapter, InvariantViolation, SSI


def _prime(adapter: GeoSyncAdapter, instrument: str, n: int = 300) -> None:
    x = np.linspace(100.0, 120.0, n)
    for px in x:
        adapter.update_tick(instrument, float(px))


def test_adapter_returns_all_12_fields() -> None:
    adapter = GeoSyncAdapter()
    instrument = adapter.instruments[0]
    _prime(adapter, instrument, 300)

    sig = adapter.get_signal(instrument)
    assert sig is not None
    assert len(sig.keys()) == 12
    required = {
        "instrument",
        "timestamp_ns",
        "gamma",
        "order_parameter_R",
        "ricci_curvature",
        "lyapunov_max",
        "regime",
        "regime_confidence",
        "regime_duration_s",
        "signal_strength",
        "risk_scalar",
        "sequence_number",
    }
    assert required.issubset(sig.keys())
    assert 0.0 <= sig["order_parameter_R"] <= 1.0
    assert -1.0 <= sig["signal_strength"] <= 1.0
    assert sig["risk_scalar"] == max(0.0, 1.0 - abs(sig["gamma"] - 1.0))


def test_gamma_is_not_assigned() -> None:
    source = Path("coherence_bridge/geosync_adapter.py").read_text(encoding="utf-8")
    assert "gamma =" not in source


def test_non_blocking_returns_last_known_good() -> None:
    adapter = GeoSyncAdapter()
    instrument = adapter.instruments[0]
    _prime(adapter, instrument, 300)
    first = adapter.get_signal(instrument)
    assert first is not None

    adapter.update_tick(instrument, 123.45)  # 301 points: cycle not ready
    second = adapter.get_signal(instrument)
    assert second is not None
    assert second["sequence_number"] == first["sequence_number"]
    assert second["timestamp_ns"] >= first["timestamp_ns"]


def test_thread_safety_concurrent_get_signal() -> None:
    adapter = GeoSyncAdapter()
    instrument = adapter.instruments[0]
    _prime(adapter, instrument, 360)

    def run_once() -> int:
        sig = adapter.get_signal(instrument)
        assert sig is not None
        return int(sig["sequence_number"])

    with ThreadPoolExecutor(max_workers=8) as ex:
        seqs = list(ex.map(lambda _: run_once(), range(20)))

    assert len(seqs) == 20
    assert len(set(seqs)) == 20


def test_ssi_internal_raises_invariant_violation() -> None:
    ssi = SSI()
    try:
        ssi.apply(domain="INTERNAL")
    except InvariantViolation:
        pass
    else:
        raise AssertionError("Expected InvariantViolation")
