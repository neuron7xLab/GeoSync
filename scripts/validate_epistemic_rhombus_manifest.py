# SPDX-License-Identifier: MIT
from pathlib import Path
import runpy

_impl = Path(__file__).with_suffix(".impl")
globals().update(runpy.run_path(str(_impl)))
