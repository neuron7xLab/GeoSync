# SPDX-License-Identifier: MIT
"""Smoke tests for the local MFN surface checker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "mfn_surface_check.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mfn_surface_check", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_surface_checker_loads_public_api() -> None:
    module = _load_module()

    assert callable(module.collect)
    assert callable(module.payload)
    assert callable(module.main)
    assert module.OK
    assert module.RED
    assert module.UNKNOWN


def test_surface_checker_makefile_coverage_is_aligned() -> None:
    module = _load_module()

    row = module.makefile_cov(ROOT)

    assert row.status == module.OK
    assert row.path == "Makefile"


def test_surface_checker_roadmap_contract_is_machine_readable() -> None:
    module = _load_module()

    row = module.roadmap_contract(ROOT)

    assert row.status == module.OK
    assert row.path == "docs/MFN_VERIFICATION_ROADMAP.json"
