# SPDX-License-Identifier: MIT
from __future__ import annotations

"""Canonical neuro namespace with src-layout compatibility."""

from pathlib import Path
from typing import cast

_PACKAGE_ROOT = Path(__file__).resolve().parent
_SRC_NEURO = _PACKAGE_ROOT.parents[2] / "src" / "geosync" / "core" / "neuro"

_package_path = cast(list[str], globals().setdefault("__path__", [str(_PACKAGE_ROOT)]))
if str(_PACKAGE_ROOT) not in _package_path:
    _package_path.insert(0, str(_PACKAGE_ROOT))

if _SRC_NEURO.exists():
    src_neuro_path = str(_SRC_NEURO)
    if src_neuro_path not in _package_path:
        _package_path.append(src_neuro_path)
    if __spec__ is not None and __spec__.submodule_search_locations is not None:
        locations = list(__spec__.submodule_search_locations)
        if src_neuro_path not in locations:
            locations.append(src_neuro_path)
            __spec__.submodule_search_locations = locations
