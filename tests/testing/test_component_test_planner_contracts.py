# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Maturity contracts for typed component-test planning."""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PLANNER = ROOT / "tools" / "testing" / "component_test_planner.py"
MATRIX = ROOT / "data" / "testing" / "component_test_matrix.json"


def _load_planner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("component_test_planner", PLANNER)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["component_test_planner"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _matrix() -> dict[str, Any]:
    return json.loads(MATRIX.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_router_policy_is_declared_in_matrix() -> None:
    planner = _load_planner()
    data = _matrix()
    assert data["router_policy"] == {
        "default_max_run_selectors": 1,
        "priority_component_ids": ["controlled_component_test_router"],
        "python_executable_env": "COMPONENT_ROUTER_PYTHON",
    }
    assert planner.max_run_selectors(None, MATRIX) == 1


def test_execution_python_uses_explicit_argument() -> None:
    planner = _load_planner()
    assert planner.execution_python(ROOT, MATRIX, "python-explicit", {}) == "python-explicit"


def test_execution_python_uses_policy_env_before_venv(tmp_path: Path) -> None:
    planner = _load_planner()
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")
    assert planner.execution_python(
        tmp_path,
        MATRIX,
        None,
        {"COMPONENT_ROUTER_PYTHON": "python-env"},
    ) == "python-env"


def test_execution_python_uses_venv_without_override(tmp_path: Path) -> None:
    planner = _load_planner()
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")
    assert planner.execution_python(tmp_path, MATRIX, None, {}) == str(venv_python)


def test_selector_plan_is_invariant_to_order_duplicates_and_dot_prefix() -> None:
    planner = _load_planner()
    left = planner.select_plan(
        [
            "./tools/testing/select_component_tests.py",
            "core/physics/stuart_landau_es.py",
            "core/physics/stuart_landau_es.py",
        ],
        MATRIX,
        ROOT,
    )
    right = planner.select_plan(
        [
            "core/physics/stuart_landau_es.py",
            "tools/testing/select_component_tests.py",
        ],
        MATRIX,
        ROOT,
    )
    assert set(left["selectors"]) == set(right["selectors"])
    assert len(left["selectors"]) == len(set(left["selectors"]))


def test_transition_events_are_schema_bound() -> None:
    planner = _load_planner()
    plan = planner.select_plan(["tools/testing/component_test_planner.py"], MATRIX, ROOT)
    events = [entry["event"] for entry in plan["transitions"]]
    assert set(events).issubset(set(planner.TRANSITION_EVENTS))
    assert events[-1] == "plan_finalized"


def test_matrix_rejects_unknown_router_priority_id(tmp_path: Path) -> None:
    planner = _load_planner()
    data = deepcopy(_matrix())
    data["router_policy"]["priority_component_ids"] = ["absent_component"]
    matrix = tmp_path / "component_test_matrix.json"
    _write(matrix, data)
    errors = planner.validate_matrix(matrix, ROOT)
    assert any("priority id has no component" in error for error in errors)
