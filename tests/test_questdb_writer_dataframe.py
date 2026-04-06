from __future__ import annotations

import pandas as pd

import coherence_bridge.questdb_writer as qw
from coherence_bridge.questdb_writer import QuestDBWriter


class _DummySender:
    called = False
    args = None

    @classmethod
    def from_conf(cls, conf: str):
        cls.conf = conf
        return cls()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def dataframe(self, df: pd.DataFrame, table_name: str, at: str) -> None:
        _DummySender.called = True
        _DummySender.args = (df.copy(), table_name, at)

    def flush(self) -> None:
        pass


def test_write_dataframe_calls_sender_dataframe(monkeypatch) -> None:
    monkeypatch.setattr(qw, "Sender", _DummySender)
    writer = QuestDBWriter("http::addr=localhost:9000;")
    df = pd.DataFrame({"timestamp_ns": [1, 2], "value": [0.1, 0.2]})

    writer.write_dataframe(df)

    assert _DummySender.called
    _, table_name, at = _DummySender.args
    assert table_name == "coherence_signals"
    assert at == "timestamp_ns"
