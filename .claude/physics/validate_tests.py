#!/usr/bin/env python3
# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Physics test validator — semantic validation against invariant registry.

Levels of validation (test mode):
  L1: INV-* reference exists in docstring (syntax)
  L2: INV-* ID exists in INVARIANTS.yaml (validity)
  L3: Test structure matches invariant type AND test_type (semantic coherence)
  L4: Assertion error messages contain required fields (observability)
  L5: Magic number detection in thresholds (heuristic)

Levels of validation (code audit mode):
  C1: Silent clamp/clip without logging (hidden invariant repair)
  C2: Bare numeric bounds without INV-* comment (undocumented constraint)

Usage:
    python .claude/physics/validate_tests.py tests/unit/physics/
    python .claude/physics/validate_tests.py tests/unit/physics/test_T2_explosive_sync.py
    python .claude/physics/validate_tests.py tests/ --summary
    python .claude/physics/validate_tests.py core/ --audit-code
    python .claude/physics/validate_tests.py --self-check
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - fallback for minimal CI images
    yaml = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[2]
INVARIANTS_PATH = REPO_ROOT / ".claude" / "physics" / "INVARIANTS.yaml"

PHYSICS_KEYWORDS = {
    "physics",
    "kuramoto",
    "sync",
    "ricci",
    "curvature",
    "free_energy",
    "thermo",
    "serotonin",
    "dopamine",
    "gaba",
    "inhibit",
    "kelly",
    "oms",
    "signalbus",
    "hpc",
    "cryptobiosis",
    "criticality",
}

MAGIC_THRESHOLD_RE = re.compile(
    r"assert\s+[^\n]*(?:<|>|<=|>=)\s*(?P<num>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
    re.IGNORECASE,
)
INV_RE = re.compile(r"INV-[A-Z0-9_-]+")


@dataclass
class Issue:
    level: str
    path: str
    line: int
    message: str


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)
    files_scanned: int = 0
    tests_scanned: int = 0
    grounded_tests: int = 0

    def add(self, level: str, path: Path, line: int, message: str) -> None:
        self.issues.append(Issue(level, str(path), line, message))

    def count(self, level: str) -> int:
        return sum(1 for issue in self.issues if issue.level == level)


def _simple_yaml_load(text: str) -> Any:
    if yaml is not None:
        return yaml.safe_load(text)
    raise RuntimeError("PyYAML unavailable; install pyyaml to load INVARIANTS.yaml")


def load_invariants() -> dict[str, dict[str, Any]]:
    data = _simple_yaml_load(INVARIANTS_PATH.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "id" in node and str(node["id"]).startswith("INV-"):
                out[str(node["id"])] = node
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data)
    return out


def _docstring(node: ast.AST) -> str:
    return ast.get_docstring(node) or ""


def _assert_messages(node: ast.AST) -> list[str]:
    messages: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Assert):
            if child.msg is None:
                messages.append("")
            elif isinstance(child.msg, ast.Constant):
                messages.append(str(child.msg.value))
            elif isinstance(child.msg, ast.JoinedStr):
                parts: list[str] = []
                for value in child.msg.values:
                    if isinstance(value, ast.Constant):
                        parts.append(str(value.value))
                messages.append("".join(parts))
            else:
                messages.append("<dynamic>")
    return messages


def _source_segment(path: Path, node: ast.AST) -> str:
    text = path.read_text(encoding="utf-8")
    return ast.get_source_segment(text, node) or ""


def _inv_ids(text: str) -> list[str]:
    return sorted(set(INV_RE.findall(text)))


def _test_functions(tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
    ]


def _expected_tokens(invariant: dict[str, Any]) -> set[str]:
    inv_type = str(invariant.get("type", "")).lower()
    test_type = str(invariant.get("test_type", "")).lower()
    tokens = {inv_type, test_type}
    if inv_type == "universal":
        tokens |= {"property", "bound", "always"}
    if inv_type == "asymptotic":
        tokens |= {"converge", "tail", "steady", "limit"}
    if inv_type == "monotonic":
        tokens |= {"monotonic", "non-increasing", "non-decreasing", "trajectory"}
    if inv_type == "statistical":
        tokens |= {"seed", "sample", "confidence", "ensemble"}
    if inv_type == "algebraic":
        tokens |= {"formula", "identity", "exact", "derivative"}
    if inv_type == "conditional":
        tokens |= {"condition", "when", "if"}
    return {token for token in tokens if token}


def _check_l3(path: Path, fn: ast.AST, invariant: dict[str, Any], report: ValidationReport) -> None:
    src = _source_segment(path, fn).lower()
    doc = _docstring(fn).lower()
    haystack = src + "\n" + doc
    tokens = _expected_tokens(invariant)
    if not tokens:
        return
    if not any(token in haystack for token in tokens):
        report.add(
            "L3",
            path,
            getattr(fn, "lineno", 1),
            f"test does not appear to exercise invariant type/test_type: {invariant.get('id')}",
        )


def _check_l4(path: Path, fn: ast.AST, report: ValidationReport) -> None:
    messages = _assert_messages(fn)
    if not messages:
        report.add("L4", path, getattr(fn, "lineno", 1), "test has no assert diagnostics")
        return
    required = {"expected", "observed"}
    for message in messages:
        lower = message.lower()
        if not message or not required <= set(re.findall(r"[a-z_]+", lower)):
            if "expected" not in lower or "observ" not in lower:
                report.add("L4", path, getattr(fn, "lineno", 1), "assert message lacks expected/observed fields")
                return


def _check_l5(path: Path, fn: ast.AST, report: ValidationReport) -> None:
    src = _source_segment(path, fn)
    for match in MAGIC_THRESHOLD_RE.finditer(src):
        num = match.group("num")
        context = src[max(0, match.start() - 120) : match.end() + 120].lower()
        if any(token in context for token in ("epsilon", "tolerance", "derived", "theory", "ulp", "machine")):
            continue
        report.add("L5", path, getattr(fn, "lineno", 1), f"possible magic threshold {num}")
        return


def _name_has_physics_keyword(name: str) -> bool:
    """Return True when a filename stem carries a physics token.

    This is intentionally token-aware. A raw substring check makes
    ``async`` match the ``sync`` keyword and turns ingestion tests into
    false-positive physics tests. This avoids false positives where non-physics substrings accidentally
    match physics keywords.
    """
    tokens = {token for token in re.split(r"[^a-z0-9]+", name.lower()) if token}
    for keyword in PHYSICS_KEYWORDS:
        if "_" in keyword:
            if keyword in name:
                return True
            continue
        if keyword == "sync":
            if tokens & {"sync", "synchrony", "synchronous", "synchronization", "synchronisation"}:
                return True
            continue
        if keyword in tokens:
            return True
        if keyword == "inhibit" and ("inhibition" in tokens or "inhibitory" in tokens):
            return True
    return False


def _is_physics_test_path(path: Path) -> bool:
    rel = path.relative_to(REPO_ROOT) if path.is_absolute() else path
    parts = {part.lower() for part in rel.parts}
    name = path.stem.lower()
    if "tests" not in parts:
        return False
    if "physics" in parts:
        return True
    if "validation" in parts and "validator" in name:
        return False
    return bool(parts & PHYSICS_KEYWORDS) or _name_has_physics_keyword(name)


def _is_physics_code_path(path: Path) -> bool:
    rel = path.relative_to(REPO_ROOT) if path.is_absolute() else path
    parts = {part.lower() for part in rel.parts}
    name = path.stem.lower()
    if "core" not in parts and "geosync_hpc" not in parts and "execution" not in parts:
        return False
    return bool(parts & PHYSICS_KEYWORDS) or _name_has_physics_keyword(name)


def validate_test_file(path: Path, invariants: dict[str, dict[str, Any]], report: ValidationReport) -> None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        report.add("L1", path, exc.lineno or 1, f"syntax error: {exc}")
        return

    fns = _test_functions(tree)
    if not fns:
        return
    report.files_scanned += 1
    for fn in fns:
        report.tests_scanned += 1
        doc = _docstring(fn)
        src = _source_segment(path, fn)
        ids = _inv_ids(doc + "\n" + src.split("assert", 1)[0])
        if not ids:
            report.add("L1", path, getattr(fn, "lineno", 1), "missing INV-* reference in test docstring/body preamble")
            continue
        valid_ids = [inv_id for inv_id in ids if inv_id in invariants]
        if not valid_ids:
            for inv_id in ids:
                report.add("L2", path, getattr(fn, "lineno", 1), f"unknown invariant id {inv_id}")
            continue
        report.grounded_tests += 1
        for inv_id in ids:
            if inv_id not in invariants:
                report.add("L2", path, getattr(fn, "lineno", 1), f"unknown invariant id {inv_id}")
                continue
            _check_l3(path, fn, invariants[inv_id], report)
        _check_l4(path, fn, report)
        _check_l5(path, fn, report)


def audit_code_file(path: Path, report: ValidationReport) -> None:
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return
    report.files_scanned += 1
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fname = ""
            if isinstance(node.func, ast.Name):
                fname = node.func.id
            elif isinstance(node.func, ast.Attribute):
                fname = node.func.attr
            if fname in {"clip", "minimum", "maximum", "clamp"}:
                line_start = max(0, getattr(node, "lineno", 1) - 3)
                lines = text.splitlines()
                context = "\n".join(lines[line_start : getattr(node, "lineno", 1) + 2])
                if not re.search(r"INV-|logging\.|logger\.|telemetry|bounds:", context):
                    report.add("C1", path, getattr(node, "lineno", 1), f"silent {fname} without INV/logging/bounds annotation")
            if fname in {"min", "max"} and len(node.args) >= 2:
                line_start = max(0, getattr(node, "lineno", 1) - 3)
                lines = text.splitlines()
                context = "\n".join(lines[line_start : getattr(node, "lineno", 1) + 2])
                if not re.search(r"INV-|bounds:|tolerance|epsilon|policy", context):
                    report.add("C2", path, getattr(node, "lineno", 1), f"bare {fname} bound without invariant/bounds annotation")


def iter_python_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(p for p in target.rglob("*.py") if ".git" not in p.parts and "__pycache__" not in p.parts)


def validate_tests(targets: list[Path], invariants: dict[str, dict[str, Any]]) -> ValidationReport:
    report = ValidationReport()
    for target in targets:
        for path in iter_python_files(target):
            rel = path.relative_to(REPO_ROOT) if path.is_absolute() else path
            if _is_physics_test_path(rel):
                validate_test_file(path, invariants, report)
    return report


def audit_code(targets: list[Path]) -> ValidationReport:
    report = ValidationReport()
    for target in targets:
        for path in iter_python_files(target):
            rel = path.relative_to(REPO_ROOT) if path.is_absolute() else path
            if _is_physics_code_path(rel):
                audit_code_file(path, report)
    return report


def print_report(report: ValidationReport, *, summary: bool) -> None:
    print("Physics validator report")
    print(f"  files scanned: {report.files_scanned}")
    print(f"  tests scanned: {report.tests_scanned}")
    print(f"  grounded tests: {report.grounded_tests}")
    for level in ["L1", "L2", "L3", "L4", "L5", "C1", "C2"]:
        count = report.count(level)
        if count:
            print(f"  [{level}] {count}")
    if not summary:
        for issue in report.issues:
            print(f"{issue.path}:{issue.line}: {issue.level}: {issue.message}")
    if not report.issues:
        print("All physics tests pass validation.")


def self_check(invariants: dict[str, dict[str, Any]]) -> int:
    if not invariants:
        print("ERROR: no invariants loaded", file=sys.stderr)
        return 1
    missing_id = [key for key, value in invariants.items() if value.get("id") != key]
    if missing_id:
        print(f"ERROR: invariant id mismatch: {missing_id[:5]}", file=sys.stderr)
        return 1
    print(f"Self-check PASS: {len(invariants)} invariants loaded from {INVARIANTS_PATH}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", type=Path, default=[Path("tests")])
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--audit-code", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)

    invariants = load_invariants()
    if args.self_check:
        return self_check(invariants)

    targets = [target if target.is_absolute() else REPO_ROOT / target for target in args.targets]
    if args.audit_code:
        report = audit_code(targets)
    else:
        report = validate_tests(targets, invariants)
    print_report(report, summary=args.summary)
    return 1 if report.issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
