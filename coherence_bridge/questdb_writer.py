from __future__ import annotations

import pandas as pd

try:
    from questdb.ingress import Sender
except Exception:  # pragma: no cover - optional dependency
    Sender = None


class QuestDBWriter:
    def __init__(self, conf: str) -> None:
        self._conf = conf

    def write_dataframe(self, df: pd.DataFrame) -> None:
        if Sender is None:
            raise RuntimeError("questdb.ingress.Sender is unavailable")
        with Sender.from_conf(self._conf) as sender:
            sender.dataframe(df, table_name="coherence_signals", at="timestamp_ns")
            sender.flush()
