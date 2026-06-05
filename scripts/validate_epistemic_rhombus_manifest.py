#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate the GeoSync Epistemic Rhombus manifest against its schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

DEFAULT_MANIFEST = Path("governance/epistemic_rhombus_manifest.jsonl")
DEFAULT_SCHEMA = Path("schemas/governance/epistemic_rhombus_manifest.schema.json")
WORKFLOW_ROOT = Path(".github/workflows")
WORKFLOW_SUFFIXES = {".yaml", ".yml"}
FORBIDDEN_PR_TARGET_TRIGGER = "pull_request" + "_target"


class ManifestReadError(ValueError):
    """Raised when the JSONL stream cannot be interpreted as manifest records."""


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield JSON objects from *path* without materializing the audit stream."""
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestReadError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ManifestReadError(f"line {line_number}: record must be a JSON object")
            yield line_number, value


def load_schema(path: Path) -> dict[str, Any]:
    """Load the JSON schema that defines the manifest contract."""
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"schema {path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(schema, dict):
        raise ValueError(f"schema {path}: root must be a JSON object")
    Draft202012Validator.check_schema(schema)
    return schema


def required_axis_order(schema: dict[str, Any]) -> list[str]:
    """Extract the canonical Rhombus axis order from the schema enum."""
    axis_schema = schema.get("properties", {}).get("axis", {})
    enum = axis_schema.get("enum")
    if not isinstance(enum, list) or not all(isinstance(axis, str) for axis in enum):
        raise ValueError("schema must define properties.axis.enum as a string array")
    return list(enum)


def required_axes(schema: dict[str, Any]) -> set[str]:
    """Extract required Rhombus axes from the schema enum to avoid drift."""
    return set(required_axis_order(schema))


def format_schema_error(error: ValidationError) -> str:
    """Return a compact, deterministic message for a schema validation error."""
    location = ".".join(str(part) for part in error.absolute_path)
    return f"{location}: {error.message}" if location else error.message


def command_target_exists(command: Iterable[str], repo_root: Path) -> bool:
    """Return whether the first command token resolves to a file or executable."""
    command_parts = list(command)
    if not command_parts:
        return False
    target = command_parts[0]
    if not isinstance(target, str):
        return False
    if target.startswith("/"):
        return Path(target).exists()
    if target.startswith("./") or "/" in target:
        return (repo_root / target).exists()
    return True


def _is_workflow_target(target: str) -> bool:
    path = Path(target)
    return (
        not path.is_absolute()
        and len(path.parts) >= 3
        and Path(*path.parts[:2]) == WORKFLOW_ROOT
        and path.suffix in WORKFLOW_SUFFIXES
    )


def _workflow_declares_trigger(text: str, trigger: str) -> bool:
    pattern = rf"(?m)^\s{{0,4}}{re.escape(trigger)}\s*:"
    return re.search(pattern, text) is not None


def _workflow_declares_readonly_contents(text: str) -> bool:
    return (
        re.search(r"(?m)^permissions\s*:\s*$", text) is not None
        and re.search(r"(?m)^\s{2}contents\s*:\s*read\s*$", text) is not None
    )


def validate_external_gate(command: list[str], repo_root: Path, prefix: str) -> list[str]:
    """Validate that an external gate points to a real fail-closed workflow."""
    target = command[0]
    errors: list[str] = []
    if not _is_workflow_target(target):
        return [
            f"{prefix}: external_gate command must target "
            f".github/workflows/*.yml or *.yaml, got: {target}"
        ]

    workflow_path = repo_root / target
    if not workflow_path.exists():
        return [f"{prefix}: enforcement command target does not exist: {target}"]

    text = workflow_path.read_text(encoding="utf-8")
    if _workflow_declares_trigger(text, FORBIDDEN_PR_TARGET_TRIGGER):
        errors.append(
            f"{prefix}: external gate must not use " f"{FORBIDDEN_PR_TARGET_TRIGGER}: {target}"
        )
    if not (
        _workflow_declares_trigger(text, "pull_request")
        or _workflow_declares_trigger(text, "merge_group")
    ):
        errors.append(
            f"{prefix}: external gate workflow must declare pull_request "
            f"or merge_group trigger: {target}"
        )
    if not _workflow_declares_readonly_contents(text):
        errors.append(
            f"{prefix}: external gate workflow must declare " f"permissions.contents=read: {target}"
        )
    return errors


def _record_prefix(line_number: int, record: dict[str, Any]) -> str:
    record_id_value = record.get("record_id")
    record_id = record_id_value if isinstance(record_id_value, str) else f"line {line_number}"
    return f"line {line_number} ({record_id})"


def _enforcement_target(record: dict[str, Any]) -> str | None:
    enforcement = record.get("enforcement")
    if not isinstance(enforcement, dict):
        return None
    command = enforcement.get("command")
    if not isinstance(command, list) or not command:
        return None
    target = command[0]
    return target if isinstance(target, str) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(manifest_path: Path, schema_path: Path, repo_root: Path) -> list[str]:
    """Validate manifest records and return all discovered errors."""
    schema = load_schema(schema_path)
    validator = Draft202012Validator(schema)
    axis_order = required_axis_order(schema)
    axis_to_record_id = {axis: f"ER-{index}" for index, axis in enumerate(axis_order, start=1)}
    axes_seen: dict[str, int] = {}
    control_gates_seen: dict[str, int] = {}
    record_ids_seen: dict[str, int] = {}
    errors: list[str] = []
    record_count = 0

    for line_number, record in iter_jsonl(manifest_path):
        record_count += 1
        prefix = _record_prefix(line_number, record)
        for error in sorted(
            validator.iter_errors(record),
            key=lambda item: list(item.absolute_path),
        ):
            errors.append(f"{prefix}: {format_schema_error(error)}")

        record_id_value = record.get("record_id")
        if isinstance(record_id_value, str):
            first_line = record_ids_seen.get(record_id_value)
            if first_line is None:
                record_ids_seen[record_id_value] = line_number
            else:
                errors.append(
                    f"{prefix}: duplicate record_id {record_id_value!r} "
                    f"(first seen on line {first_line})"
                )

        axis = record.get("axis")
        if isinstance(axis, str):
            expected_record_id = axis_to_record_id.get(axis)
            if isinstance(record_id_value, str) and record_id_value != expected_record_id:
                errors.append(
                    f"{prefix}: axis {axis!r} must use record_id "
                    f"{expected_record_id!r}, got {record_id_value!r}"
                )
            first_line = axes_seen.get(axis)
            if first_line is None:
                axes_seen[axis] = line_number
            else:
                errors.append(
                    f"{prefix}: duplicate axis {axis!r} " f"(first seen on line {first_line})"
                )

        control_gates = record.get("control_gates")
        if isinstance(control_gates, list):
            for gate in control_gates:
                if not isinstance(gate, str):
                    continue
                first_line = control_gates_seen.get(gate)
                if first_line is None:
                    control_gates_seen[gate] = line_number
                else:
                    errors.append(
                        f"{prefix}: duplicate control gate {gate!r} "
                        f"(first seen on line {first_line})"
                    )

        enforcement = record.get("enforcement")
        if isinstance(enforcement, dict):
            mode = enforcement.get("mode")
            command = enforcement.get("command")
            if isinstance(command, list) and all(isinstance(part, str) for part in command):
                if mode == "external_gate":
                    errors.extend(validate_external_gate(command, repo_root, prefix))
                elif not command_target_exists(command, repo_root):
                    errors.append(
                        f"{prefix}: enforcement command target does not exist: " f"{command[0]}"
                    )

    if record_count == 0:
        errors.append("manifest: no records found")

    missing_axes = sorted(set(axis_order) - set(axes_seen))
    if missing_axes:
        errors.append(f"manifest: missing axes: {', '.join(missing_axes)}")
    return errors


def build_report(
    manifest_path: Path,
    schema_path: Path,
    repo_root: Path,
    errors: list[str],
) -> dict[str, Any]:
    """Build a deterministic validation report for CI logs and audit bundles."""
    records = [
        {
            "axis": record.get("axis"),
            "control_gates": record.get("control_gates"),
            "enforcement_target": _enforcement_target(record),
            "line": line_number,
            "record_id": record.get("record_id"),
        }
        for line_number, record in iter_jsonl(manifest_path)
    ]
    return {
        "schema_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "schema_path": str(schema_path),
        "schema_sha256": _sha256(schema_path),
        "repo_root": str(repo_root),
        "record_count": len(records),
        "axes": [record["axis"] for record in records],
        "records": records,
        "errors": errors,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Write a deterministic JSON report, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", default=DEFAULT_MANIFEST, type=Path)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA, type=Path)
    parser.add_argument("--repo-root", default=Path.cwd(), type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional path for a deterministic JSON validation report.",
    )
    args = parser.parse_args(argv)

    try:
        errors = validate_manifest(args.manifest, args.schema, args.repo_root)
        if args.report is not None:
            write_report(
                args.report,
                build_report(args.manifest, args.schema, args.repo_root, errors),
            )
    except (OSError, ValueError) as exc:
        print(f"manifest validation failed: {exc}", file=sys.stderr)
        return 2

    if errors:
        print("manifest validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"validated Epistemic Rhombus manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
