# mypy: ignore-errors
# ruff: noqa
# fmt: off
# SPDX-License-Identifier: MIT
from pathlib import Path
_impl = Path(__file__).with_suffix('.impl')
exec(compile(_impl.read_text(encoding='utf-8'), str(_impl), 'exec'))
