# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Tests for the math-boundaries operating-domain registry gate."""

from __future__ import annotations

from typing import Any

from scripts.ci.check_math_boundaries import REGISTRY_PATH, check_registry, main


def _good_kernel() -> dict[str, Any]:
    return {
        "id": "k1",
        "name": "Kernel One",
        "source": "scripts/ci/check_math_boundaries.py",  # a file that exists
        "invariants": ["INV-OK"],
        "valid": "valid domain",
        "degraded": "degraded domain",
        "invalid": "invalid domain",
        "falsifier": "what would falsify",
    }


KNOWN = {"INV-OK", "INV-ALSO"}


def test_repository_registry_is_consistent() -> None:
    """The shipped docs/math_boundaries.yaml passes its own gate."""
    assert main() == 0


def test_registry_file_present() -> None:
    assert REGISTRY_PATH.exists()


def test_valid_registry_has_no_errors(tmp_path: Any) -> None:
    reg = {"version": 1, "kernels": [_good_kernel()]}
    assert check_registry(reg, known_invariants=KNOWN, root=REGISTRY_PATH.parents[1]) == []


def test_empty_kernels_fails() -> None:
    errors = check_registry({"kernels": []}, known_invariants=KNOWN, root=REGISTRY_PATH.parents[1])
    assert any("non-empty list" in e for e in errors)


def test_missing_text_field_fails() -> None:
    k = _good_kernel()
    del k["falsifier"]
    errors = check_registry({"kernels": [k]}, known_invariants=KNOWN, root=REGISTRY_PATH.parents[1])
    assert any("falsifier" in e for e in errors)


def test_unknown_invariant_fails() -> None:
    k = _good_kernel()
    k["invariants"] = ["INV-DOES-NOT-EXIST"]
    errors = check_registry({"kernels": [k]}, known_invariants=KNOWN, root=REGISTRY_PATH.parents[1])
    assert any("unknown invariant" in e for e in errors)


def test_missing_source_path_fails() -> None:
    k = _good_kernel()
    k["source"] = "core/does/not/exist.py"
    errors = check_registry({"kernels": [k]}, known_invariants=KNOWN, root=REGISTRY_PATH.parents[1])
    assert any("source path does not exist" in e for e in errors)


def test_duplicate_ids_fail() -> None:
    reg = {"kernels": [_good_kernel(), _good_kernel()]}
    errors = check_registry(reg, known_invariants=KNOWN, root=REGISTRY_PATH.parents[1])
    assert any("duplicate kernel id" in e for e in errors)
