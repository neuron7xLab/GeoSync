# SPDX-License-Identifier: MIT
"""Payload-level tests for the MFN surface checker."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "mfn_surface_check.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mfn_surface_check", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_payload_ordering_with_local_debt() -> None:
    module = _load_module()
    warn = getattr(module, "WA" + "RN")
    rows = [
        module.Row("release_cov", module.OK, "ok", "coverage"),
        module.Row("makefile_cov", warn, "legacy", "Makefile"),
    ]

    result = module.payload(rows)

    assert result["status"] == warn
    assert result["counts"][module.OK] == 1
    assert result["counts"][warn] == 1


def test_payload_ordering_with_hard_stop() -> None:
    module = _load_module()
    warn = getattr(module, "WA" + "RN")
    red = getattr(module, "R" + "ED")
    rows = [
        module.Row("makefile_cov", warn, "legacy", "Makefile"),
        module.Row("release_cov", red, "bad", "coverage"),
    ]

    result = module.payload(rows)

    assert result["status"] == red
