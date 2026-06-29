# SPDX-License-Identifier: MIT
from __future__ import annotations

"""Canonical GeoSync core namespace.

The top-level :mod:`geosync` package is a lightweight compatibility shim whose
module search path also points at ``src/geosync``. Creating ``geosync.core`` for
the dopamine canonical surface must preserve that namespace behavior; otherwise
legacy imports such as ``geosync.core.compat`` stop resolving through the
``src/geosync/core`` rail.
"""

from pathlib import Path
from typing import cast

_PACKAGE_ROOT = Path(__file__).resolve().parent
_SRC_CORE = _PACKAGE_ROOT.parents[1] / "src" / "geosync" / "core"

_package_path = cast(list[str], globals().setdefault("__path__", [str(_PACKAGE_ROOT)]))
if str(_PACKAGE_ROOT) not in _package_path:
    _package_path.insert(0, str(_PACKAGE_ROOT))

if _SRC_CORE.exists():
    src_core_path = str(_SRC_CORE)
    if src_core_path not in _package_path:
        _package_path.append(src_core_path)
    if __spec__ is not None and __spec__.submodule_search_locations is not None:
        locations = list(__spec__.submodule_search_locations)
        if src_core_path not in locations:
            locations.append(src_core_path)
            __spec__.submodule_search_locations = locations
