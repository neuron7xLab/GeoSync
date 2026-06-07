# mypy: ignore-errors
# ruff: noqa
# fmt: off
# SPDX-License-Identifier: MIT
import runpy
from pathlib import Path

globals().update(runpy.run_path(str(Path(__file__).with_suffix('.impl'))))
