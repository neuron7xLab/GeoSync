# SPDX-License-Identifier: MIT
"""Canonical neuro namespace with src-layout compatibility."""

from __future__ import annotations

from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent
_SRC_NEURO = _PACKAGE_ROOT.parents[2] / "src" / "geosync" / "core" / "neuro"

_package_path = [str(_PACKAGE_ROOT)]
_existing_path = globals().get("__path__")
if isinstance(_existing_path, list):
    _package_path = [str(item) for item in _existing_path]
if str(_PACKAGE_ROOT) not in _package_path:
    _package_path.insert(0, str(_PACKAGE_ROOT))
if _SRC_NEURO.exists() and str(_SRC_NEURO) not in _package_path:
    _package_path.append(str(_SRC_NEURO))
globals()["__path__"] = _package_path
