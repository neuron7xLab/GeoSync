# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Tests for deterministic changed-file -> component-test routing."""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "testing" / "select_component_tests.py"
MATRIX = ROOT / "data" / "testing" / "component_test_matrix.json"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("component_tests", TOOL)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["component_tests"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _matrix() -> dict[str, Any]:
    return json.loads(MATRIX.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_live_component_test_matrix_is_valid() -> None:
    tool = _load_tool()
    assert tool.validate_matrix(MATRIX, ROOT) == []


def test_stuart_landau_change_selects_only_component_selector() -> None:
    tool = _load_tool()
    plan = tool.select_plan(["core/physics/stuart_landau_es.py"], MATRIX, ROOT)
    assert plan["status"] == "ready"
    assert [component["id"] for component in plan["components"]] == [
        "stuart_landau_runtime"
    ]
    assert plan["selectors"] == ["tests/unit/physics/test_T2b_stuart_landau_es.py"]


def test_setup_action_change_selects_dependency_and_setup_tests() -> None:
    tool = _load_tool()
    plan = tool.select_plan([".github/actions/setup-geosync/action.yml"], MATRIX, ROOT)
    assert plan["status"] == "ready"
    assert [component["id"] for component in plan["components"]] == [
        "dependency_operational_determinism"
    ]
    assert plan["selectors"] == [
        "tests/deps/test_operational_dependency_determinism.py",
        "tests/ci/test_setup_geosync_toolchain.py",
    ]


def test_router_change_selects_self_tests() -> None:
    tool = _load_tool()
    plan = tool.select_plan(["tools/testing/select_component_tests.py"], MATRIX, ROOT)
    assert plan["status"] == "ready"
    assert [component["id"] for component in plan["components"]] == [
        "controlled_component_test_router"
    ]
    assert plan["selectors"] == ["tests/testing/test_select_component_tests.py"]


def test_docs_only_unmapped_change_is_empty_not_invalid() -> None:
    tool = _load_tool()
    plan = tool.select_plan(["docs/notes/unmapped.md"], MATRIX, ROOT)
    assert plan["status"] == "empty"
    assert plan["components"] == []
    assert plan["selectors"] == []
    assert plan["pytest_command"] == []


def test_matrix_rejects_missing_selector_path(tmp_path: Path) -> None:
    tool = _load_tool()
    data = deepcopy(_matrix())
    data["components"][0]["selectors"] = ["tests/missing/test_absent.py"]
    matrix = tmp_path / "component_test_matrix.json"
    _write(matrix, data)
    errors = tool.validate_matrix(matrix, ROOT)
    assert any("selector path does not exist" in error for error in errors)


def test_matrix_rejects_duplicate_component_id(tmp_path: Path) -> None:
    tool = _load_tool()
    data = deepcopy(_matrix())
    data["components"][1]["id"] = data["components"][0]["id"]
    matrix = tmp_path / "component_test_matrix.json"
    _write(matrix, data)
    errors = tool.validate_matrix(matrix, ROOT)
    assert any("duplicate id" in error for error in errors)


def test_matrix_rejects_unbounded_runtime(tmp_path: Path) -> None:
    tool = _load_tool()
    data = deepcopy(_matrix())
    data["components"][0]["max_runtime_seconds"] = 1200
    matrix = tmp_path / "component_test_matrix.json"
    _write(matrix, data)
    errors = tool.validate_matrix(matrix, ROOT)
    assert any("max_runtime_seconds" in error for error in errors)
