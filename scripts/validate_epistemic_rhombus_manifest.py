# mypy: ignore-errors
# SPDX-License-Identifier: MIT
import runpy
from pathlib import Path

_is_main = __name__ == "__main__"
_impl = Path(__file__).with_suffix(".impl")
_symbols = runpy.run_path(str(_impl))
globals().update(_symbols)

if _is_main:
    raise SystemExit(_symbols["main"]())
