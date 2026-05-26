from __future__ import annotations

import csv
from pathlib import Path

import yaml


def test_sample_financial_data_matches_contract() -> None:
    contract = yaml.safe_load(
        Path('docs/architecture/methodology/financial/FINANCIAL_DATA_CONTRACT.yaml').read_text(encoding='utf-8')
    )
    required = [f['name'] for f in contract['required_fields']]

    rows = list(csv.DictReader(Path('docs/architecture/methodology/financial/sample_financial_data.csv').open()))
    assert rows
    assert set(required).issubset(rows[0].keys())

    for row in rows:
        assert float(row['high']) >= float(row['low'])
        assert float(row['open']) > 0
        assert float(row['close']) > 0
        assert float(row['volume']) >= 0
        assert float(row['spread_bps']) >= 0
        assert float(row['fee_bps']) >= 0
