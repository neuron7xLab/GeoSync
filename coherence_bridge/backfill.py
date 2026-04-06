from __future__ import annotations

from typing import Any

import pandas as pd

from .geosync_adapter import GeoSyncAdapter


def compute_signals_on_window(
    tick_data: pd.DataFrame,
    *,
    instrument: str,
    window_size: int = 300,
    step: int = 60,
    adapter: GeoSyncAdapter | None = None,
) -> list[dict[str, Any]]:
    """Compute contract-aligned signals over rolling windows of tick data."""
    if "price" not in tick_data.columns:
        raise ValueError("tick_data must include a 'price' column")
    if tick_data.empty:
        return []

    engine = adapter or GeoSyncAdapter()
    if instrument not in engine.instruments:
        raise KeyError(f"instrument {instrument!r} not configured in adapter")

    signals: list[dict[str, Any]] = []
    prices = tick_data["price"].astype(float).to_numpy()

    for end in range(window_size, len(prices) + 1, step):
        start = end - window_size
        for px in prices[start:end]:
            engine.update_tick(instrument, float(px))
        sig = engine.get_signal(instrument)
        if sig is not None:
            signals.append(sig)

    return signals
