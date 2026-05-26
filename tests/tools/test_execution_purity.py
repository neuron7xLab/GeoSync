from __future__ import annotations

from pathlib import Path


def test_conftest_uses_stable_coroutine_inspection_api() -> None:
    text = Path('conftest.py').read_text(encoding='utf-8')
    assert 'inspect.iscoroutinefunction' in text
    assert 'asyncio.iscoroutinefunction' not in text
