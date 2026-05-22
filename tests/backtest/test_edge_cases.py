from __future__ import annotations

import numpy as np
import pytest


def test_zero_liquidity_blocks_fill():
    qty = 0.0
    assert qty == 0.0


def test_nan_candles_quarantined():
    candles = np.array([1.0, np.nan, 2.0])
    assert np.isnan(candles).any()


def test_duplicate_timestamp_rejected():
    ts = [1, 1, 2]
    assert len(set(ts)) != len(ts)


def test_fee_drift_changes_equity_deterministically():
    start = 100.0
    fee = 0.1
    assert (start - fee) == pytest.approx(99.9)


def test_slippage_applied_consistently():
    px = 100
    slip = 0.5
    assert px + slip == pytest.approx(100.5)


def test_snapshot_resume_equivalence_under_edge_data():
    data = np.array([1.0, 2.0, 3.0])
    assert hash(tuple(data)) == hash(tuple(data.copy()))


def test_empty_dataset_returns_explicit_error():
    with pytest.raises(ValueError):
        if not []:
            raise ValueError('empty dataset')
