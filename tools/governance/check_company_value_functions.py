#!/usr/bin/env python3
# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Validate company-grade value functions.

The manifest converts company values into measurable engineering constraints:
value -> invariant -> evidence surface -> fail-closed verdict.

Stdlib-only by design so governance can run without trusting the dependency graph.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "governance" / "company_value_functions.json"
REQUIRED_IDS = tuple(f"VF-{idx:02d}" for idx in range(1, 8))
REQUIRED_DECISION_FLAGS = (
    "hard_constraints_first",
    "forbid_green_by_vacuum",
    "forbid_unmeasured_success_claims",
    "fail_closed_on_missing_evidence",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_non_empty_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value)


def check(path: Path = MANIFEST) -> list[str]:
    errors: list[str] = []
    try:
        data = _load(path)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path}: cannot load value-function manifest: {exc}"]

    if data.get("schema") != "geosync.company_value_functions.v1":
        errors.append(f"{path}: schema must be geosync.company_value_functions.v1")
    if data.get("status") != "normative":
        errors.append(f"{path}: status must be normative")

    decision_rule = data.get("decision_rule")
    if not isinstance(decision_rule, dict):
        errors.append(f"{path}: decision_rule must be an object")
        decision_rule = {}
    for flag in REQUIRED_DECISION_FLAGS:
        if decision_rule.get(flag) is not True:
            errors.append(f"{path}: decision_rule.{flag} must be true")
    minimum_total = decision_rule.get("minimum_total_score")
    if not isinstance(minimum_total, float | int) or minimum_total < 0.90:
        errors.append(f"{path}: minimum_total_score must be >= 0.90")
    minimum_individual = decision_rule.get("minimum_individual_score")
    if not isinstance(minimum_individual, float | int) or minimum_individual < 0.75:
        errors.append(f"{path}: minimum_individual_score must be >= 0.75")

    value_functions = data.get("value_functions")
    if not isinstance(value_functions, list):
        return errors + [f"{path}: value_functions must be a list"]
    if len(value_functions) != len(REQUIRED_IDS):
        errors.append(f"{path}: expected {len(REQUIRED_IDS)} value functions")

    ids: list[str] = []
    total_weight = 0.0
    hard_constraints = 0
    for idx, vf in enumerate(value_functions, start=1):
        prefix = f"{path}: value_functions[{idx}]"
        if not isinstance(vf, dict):
            errors.append(f"{prefix}: must be an object")
            continue
        vf_id = vf.get("id")
        ids.append(str(vf_id))
        if vf_id not in REQUIRED_IDS:
            errors.append(f"{prefix}: invalid id {vf_id!r}")
        for key in ("name", "principle"):
            if not _is_non_empty_str(vf.get(key)):
                errors.append(f"{prefix}: {key} must be a non-empty string")
        weight = vf.get("weight")
        if not isinstance(weight, float | int) or weight <= 0:
            errors.append(f"{prefix}: weight must be positive")
        else:
            total_weight += float(weight)
        if vf.get("hard_constraint") is True:
            hard_constraints += 1
        elif vf.get("hard_constraint") is not False:
            errors.append(f"{prefix}: hard_constraint must be boolean")
        for key in (
            "measurable_invariants",
            "evidence_surfaces",
            "score_inputs",
            "failure_modes",
        ):
            values = vf.get(key)
            if not _is_non_empty_list(values):
                errors.append(f"{prefix}: {key} must be a non-empty list")
                continue
            if not all(_is_non_empty_str(item) for item in values):
                errors.append(f"{prefix}: {key} must contain only non-empty strings")
        if not vf.get("evidence_surfaces"):
            continue
        executable_prefixes = ("tools/", "scripts/", ".github/")
        has_executable_surface = any(
            str(surface).startswith(executable_prefixes)
            for surface in vf["evidence_surfaces"]
        )
        if not has_executable_surface:
            errors.append(f"{prefix}: at least one evidence surface must be executable/CI")

    if tuple(ids) != REQUIRED_IDS:
        errors.append(f"{path}: value function ids must be ordered {REQUIRED_IDS}")
    if abs(total_weight - 1.0) > 0.000001:
        errors.append(
            f"{path}: value function weights must sum to 1.0, got {total_weight:.6f}"
        )
    if hard_constraints < 3:
        errors.append(f"{path}: at least three value functions must be hard constraints")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args(argv)

    errors = check(args.manifest)
    if errors:
        print("COMPANY VALUE FUNCTION GATE: FAIL", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    print("COMPANY VALUE FUNCTION GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
