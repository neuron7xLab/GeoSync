# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Traceability tests for PR preflight requirements."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQS = ROOT / "tools" / "ci" / "pr_preflight_requirements.json"
TEST_GLOB = "test_pr_preflight*.py"
VAGUE = {"better", "robust", "clean", "solid", "proper", "good", "improved", "production-ready"}


def _requirements() -> list[dict[str, object]]:
    payload = json.loads(REQS.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["system"] == "pr_preflight"
    reqs = payload["requirements"]
    assert isinstance(reqs, list)
    return reqs


def _test_names() -> set[str]:
    names: set[str] = set()
    for path in (ROOT / "tests" / "ci").glob(TEST_GLOB):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names.update(node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"))
    return names


def test_requirements_have_required_traceability_fields() -> None:
    required = {
        "id",
        "statement",
        "priority",
        "verification_method",
        "acceptance_criteria",
        "linked_tests",
        "linked_code",
    }
    for req in _requirements():
        missing = required - req.keys()
        assert not missing, req
        assert req["priority"] in {"MUST", "SHOULD", "MAY"}
        assert req["verification_method"] in {"unit_test", "behavioral_test", "schema_test", "shell_test", "manual_review"}
        assert req["failure_status"] in {"FAIL", "BLOCKED"}
        assert req["acceptance_criteria"]
        assert req["linked_code"]


def test_must_requirements_have_existing_linked_tests() -> None:
    names = _test_names()
    for req in _requirements():
        linked = req["linked_tests"]
        assert isinstance(linked, list)
        if req["priority"] == "MUST":
            assert linked, req["id"]
        missing = [name for name in linked if name not in names]
        assert missing == [], f"{req['id']} references missing tests: {missing}"


def test_requirement_ids_are_unique() -> None:
    ids = [req["id"] for req in _requirements()]
    assert len(ids) == len(set(ids))


def test_acceptance_criteria_are_not_vague_only() -> None:
    for req in _requirements():
        criteria = req["acceptance_criteria"]
        assert isinstance(criteria, list)
        for criterion in criteria:
            words = {part.strip(" .,;:()[]{}\"'").lower() for part in str(criterion).split()}
            assert words - VAGUE, f"{req['id']} has vague-only criterion: {criterion!r}"
