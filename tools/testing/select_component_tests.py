#!/usr/bin/env python3
# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Select isolated component tests from changed repository paths."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "data" / "testing" / "component_test_matrix.json"
SCHEMA = "geosync.component_test_matrix.v1"
BLOCKING_STATUSES = {"invalid_matrix", "unmapped_required_surface"}
DEFAULT_MAX_RUN_SELECTORS = 1
PRIORITY_COMPONENT_IDS = ("controlled_component_test_router",)


def _load_matrix(path: Path = MATRIX) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _selector_path(selector: str) -> str:
    return selector.split("::", 1)[0]


def _string_list(value: object, label: str, errors: list[str]) -> list[str]:
    if not isinstance(value, list) or not value:
        errors.append(f"{label} must be a non-empty list")
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            errors.append(f"{label} must contain non-empty strings")
            continue
        out.append(item)
    return out


def _transition(transitions: list[dict[str, Any]], event: str, **state: Any) -> None:
    entry: dict[str, Any] = {"seq": len(transitions) + 1, "event": event}
    entry.update(state)
    transitions.append(entry)


def validate_matrix(path: Path = MATRIX, repo_root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    try:
        matrix = _load_matrix(path)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path}: cannot load component test matrix: {exc}"]

    if matrix.get("schema") != SCHEMA:
        errors.append(f"{path}: schema must be {SCHEMA}")
    if matrix.get("status") != "normative":
        errors.append(f"{path}: status must be normative")

    rule = matrix.get("decision_rule")
    if not isinstance(rule, dict):
        errors.append(f"{path}: decision_rule must be an object")
    else:
        if rule.get("controlled_input") != "changed repository path":
            errors.append(f"{path}: decision_rule.controlled_input is invalid")
        if rule.get("expected_output") != "deterministic pytest selector plan":
            errors.append(f"{path}: decision_rule.expected_output is invalid")
        if rule.get("fail_closed_on_invalid_matrix") is not True:
            errors.append(f"{path}: fail_closed_on_invalid_matrix must be true")
        if rule.get("fail_closed_on_unmapped_required_surface") is not True:
            errors.append(f"{path}: fail_closed_on_unmapped_required_surface must be true")

    required_patterns = _string_list(
        matrix.get("required_mapping_patterns"),
        f"{path}: required_mapping_patterns",
        errors,
    )
    for pattern in required_patterns:
        if pattern.startswith("docs/"):
            errors.append(f"{path}: docs/* cannot be a required mapping pattern")

    components = matrix.get("components")
    if not isinstance(components, list) or not components:
        return errors + [f"{path}: components must be a non-empty list"]

    seen_ids: set[str] = set()
    for index, component in enumerate(components, start=1):
        prefix = f"{path}: components[{index}]"
        if not isinstance(component, dict):
            errors.append(f"{prefix}: must be an object")
            continue
        component_id = component.get("id")
        if not isinstance(component_id, str) or not component_id.strip():
            errors.append(f"{prefix}: id must be a non-empty string")
        elif component_id in seen_ids:
            errors.append(f"{prefix}: duplicate id {component_id}")
        else:
            seen_ids.add(component_id)

        selectors = _string_list(
            component.get("selectors"),
            f"{prefix}: selectors",
            errors,
        )
        _string_list(component.get("patterns"), f"{prefix}: patterns", errors)
        for selector in selectors:
            selector_file = repo_root / _selector_path(selector)
            if not selector_file.exists():
                errors.append(f"{prefix}: selector path does not exist: {selector}")
        if component.get("expected_outcome") != "pass":
            errors.append(f"{prefix}: expected_outcome must be pass")
        if component.get("isolation_level") != "component":
            errors.append(f"{prefix}: isolation_level must be component")
        max_runtime = component.get("max_runtime_seconds")
        if not isinstance(max_runtime, int) or not (1 <= max_runtime <= 600):
            errors.append(f"{prefix}: max_runtime_seconds must be an int in [1, 600]")

    return errors


def _matches(path: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(path, pattern)


def _path_matches_any(path: str, patterns: list[str]) -> bool:
    return any(_matches(path, pattern) for pattern in patterns)


def _is_required_mapping(path: str, matrix: dict[str, Any]) -> bool:
    patterns = matrix.get("required_mapping_patterns", [])
    return isinstance(patterns, list) and _path_matches_any(path, patterns)


def _normalize_path(path: str) -> str:
    normalized = path.strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _execution_python(repo_root: Path) -> str:
    venv_python = repo_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _prioritize_selectors(
    selectors: list[str], matched_components: list[dict[str, Any]]
) -> list[str]:
    priority: list[str] = []
    seen: set[str] = set()
    for component_id in PRIORITY_COMPONENT_IDS:
        for component in matched_components:
            if component["id"] != component_id:
                continue
            for selector in component["selectors"]:
                if selector not in seen:
                    priority.append(selector)
                    seen.add(selector)
    priority.extend(selector for selector in selectors if selector not in seen)
    return priority


def _invalid_plan(errors: list[str], transitions: list[dict[str, Any]]) -> dict[str, Any]:
    _transition(
        transitions,
        "matrix_validation_failed",
        status="invalid_matrix",
        error_count=len(errors),
    )
    return {
        "status": "invalid_matrix",
        "errors": errors,
        "components": [],
        "selectors": [],
        "transitions": transitions,
    }


def _blocked_plan(
    normalized: list[str],
    matched_components: list[dict[str, Any]],
    selectors: list[str],
    missing_routes: list[str],
    transitions: list[dict[str, Any]],
) -> dict[str, Any]:
    _transition(
        transitions,
        "plan_blocked",
        status="unmapped_required_surface",
        cause="missing required component-test route",
    )
    return {
        "status": "unmapped_required_surface",
        "changed_files": normalized,
        "components": matched_components,
        "selectors": selectors,
        "unmapped_required": missing_routes,
        "errors": [
            "required mapping surface has no component-test route: " + path
            for path in missing_routes
        ],
        "pytest_command": [],
        "transitions": transitions,
    }


def select_plan(
    changed_files: list[str], matrix_path: Path = MATRIX, repo_root: Path = ROOT
) -> dict[str, Any]:
    transitions: list[dict[str, Any]] = []
    errors = validate_matrix(matrix_path, repo_root)
    if errors:
        return _invalid_plan(errors, transitions)

    _transition(transitions, "matrix_validated", status="ok")
    matrix = _load_matrix(matrix_path)
    matched_components: list[dict[str, Any]] = []
    selectors: list[str] = []
    seen_selectors: set[str] = set()
    mapped_paths: set[str] = set()

    normalized = [_normalize_path(path) for path in changed_files if path.strip()]
    _transition(
        transitions,
        "paths_normalized",
        input_count=len(changed_files),
        normalized_count=len(normalized),
        changed_files=normalized,
    )
    for component in matrix["components"]:
        matched_paths = [
            path
            for path in normalized
            if any(_matches(path, pattern) for pattern in component["patterns"])
        ]
        if not matched_paths:
            continue
        mapped_paths.update(matched_paths)
        matched_components.append(
            {
                "id": component["id"],
                "matched_paths": matched_paths,
                "selectors": component["selectors"],
                "expected_outcome": component["expected_outcome"],
                "isolation_level": component["isolation_level"],
                "max_runtime_seconds": component["max_runtime_seconds"],
            }
        )
        for selector in component["selectors"]:
            if selector not in seen_selectors:
                seen_selectors.add(selector)
                selectors.append(selector)

    _transition(
        transitions,
        "components_matched",
        component_ids=[component["id"] for component in matched_components],
        selector_count=len(selectors),
    )
    missing_routes = [
        path
        for path in normalized
        if path not in mapped_paths and _is_required_mapping(path, matrix)
    ]
    _transition(
        transitions,
        "required_surface_checked",
        missing_route_count=len(missing_routes),
        missing_routes=missing_routes,
    )
    if missing_routes:
        return _blocked_plan(
            normalized,
            matched_components,
            selectors,
            missing_routes,
            transitions,
        )

    selectors = _prioritize_selectors(selectors, matched_components)
    status = "ready" if selectors else "empty"
    _transition(
        transitions,
        "selectors_prioritized",
        selector_count=len(selectors),
        selectors=selectors,
    )
    _transition(transitions, "plan_finalized", status=status)
    return {
        "status": status,
        "changed_files": normalized,
        "components": matched_components,
        "selectors": selectors,
        "pytest_command": [
            _execution_python(repo_root),
            "-m",
            "pytest",
            *selectors,
        ]
        if selectors
        else [],
        "transitions": transitions,
    }


def _read_changed_files(args: argparse.Namespace) -> list[str]:
    changed: list[str] = list(args.changed_file or [])
    if args.changed_files_from:
        changed.extend(
            line.strip()
            for line in args.changed_files_from.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return changed


def _max_run_selectors(value: str | None) -> int:
    if not value:
        return DEFAULT_MAX_RUN_SELECTORS
    try:
        parsed = int(value)
    except ValueError:
        return DEFAULT_MAX_RUN_SELECTORS
    return parsed if parsed > 0 else DEFAULT_MAX_RUN_SELECTORS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument("--changed-files-from", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--print-pytest", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)

    errors = validate_matrix(args.matrix, args.repo_root)
    if errors:
        print("COMPONENT TEST MATRIX: FAIL", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    if args.validate_only:
        print("COMPONENT TEST MATRIX: PASS")
        return 0

    plan = select_plan(_read_changed_files(args), args.matrix, args.repo_root)
    if args.print_pytest:
        print(" ".join(plan.get("pytest_command", [])))
    else:
        print(json.dumps(plan, indent=2, sort_keys=True))

    if plan["status"] in BLOCKING_STATUSES:
        return 2
    if args.run and plan["selectors"]:
        limit = _max_run_selectors(os.environ.get("COMPONENT_ROUTER_MAX_SELECTORS"))
        selectors = plan["selectors"][:limit]
        print(f"COMPONENT_ROUTER_SELECTED {len(selectors)}/{len(plan['selectors'])}")
        command = [_execution_python(args.repo_root), "-m", "pytest", *selectors]
        return subprocess.run(command, cwd=args.repo_root, check=False).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
