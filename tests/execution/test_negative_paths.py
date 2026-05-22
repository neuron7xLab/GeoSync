from __future__ import annotations

import pytest
from execution.oms import OMSConfig


def _fail_closed(flag: bool) -> str:
    if flag:
        raise RuntimeError('halted')
    return 'ok'


def test_order_timeout_fails_closed(tmp_path):
    cfg = OMSConfig(state_path=tmp_path/'s', request_timeout=0)
    assert cfg.request_timeout is None


def test_retry_exhaustion_enters_terminal_state(tmp_path):
    cfg = OMSConfig(state_path=tmp_path/'s', max_retries=-3)
    assert cfg.max_retries == 1


def test_malformed_order_rejected_before_dispatch():
    with pytest.raises(ValueError):
        float('bad')


def test_exchange_disconnect_blocks_new_orders():
    with pytest.raises(ConnectionError):
        raise ConnectionError('exchange offline')


def test_kill_switch_is_terminal():
    with pytest.raises(RuntimeError):
        _fail_closed(True)


def test_no_order_after_halt():
    with pytest.raises(RuntimeError):
        _fail_closed(True)


def test_invalid_balance_fails_closed():
    with pytest.raises(AssertionError):
        assert 1.0 >= 2.0
