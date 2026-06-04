#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate the GeoSync Epistemic Rhombus manifest against its schema."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

DEFAULT_MANIFEST = Path("governance/epistemic_rhombus_manifest.jsonl")
DEFAULT_SCHEMA = Path("schemas/governance/epistemic_rhombus_manifest.schema.json")


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield JSON objects from *path* without materializing the audit stream."""
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number}: record must be a JSON object")
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


def required_axes(schema: dict[str, Any]) -> set[str]:
    """Extract required Rhombus axes from the schema enum to avoid drift."""
    axis_schema = schema.get("properties", {}).get("axis", {})
    enum = axis_schema.get("enum")
    if not isinstance(enum, list) or not all(isinstance(axis, str) for axis in enum):
        raise ValueError("schema must define properties.axis.enum as a string array")
    return set(enum)


def format_schema_error(error: ValidationError) -> str:
    """Return a compact, deterministic message for a schema validation error."""
    location = ".".join(str(part) for part in error.absolute_path)
    prefix = f"{location}: " if location else ""
    return f"{prefix}{error.message}"


def command_target_exists(command: Iterable[str], repo_root: Path) -> bool:
    """Return whether the first command token resolves to a file or executable name.

    Manifest records may point to workflow files, repository-local commands, or a
    command name resolved by the execution environment. File paths are checked
    locally; bare command names are accepted as externally resolved.
    """
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


def validate_manifest(manifest_path: Path, schema_path: Path, repo_root: Path) -> list[str]:
    """Validate manifest records and return all discovered errors."""
    schema = load_schema(schema_path)
    validator = Draft202012Validator(schema)
    axes_required = required_axes(schema)
    axes_seen: dict[str, int] = {}
    record_ids_seen: dict[str, int] = {}
    errors: list[str] = []
    record_count = 0

    for line_number, record in iter_jsonl(manifest_path):
        record_count += 1
        record_id_value = record.get("record_id")
        record_id = record_id_value if isinstance(record_id_value, str) else f"line {line_number}"
        prefix = f"line {line_number} ({record_id})"

        for error in sorted(
            validator.iter_errors(record),
            key=lambda item: list(item.absolute_path),
        ):
            errors.append(f"{prefix}: {format_schema_error(error)}")

        if isinstance(record_id_value, str):
            first_line = record_ids_seen.get(record_id_value)
            if first_line is not None:
                errors.append(
                    f"{prefix}: duplicate record_id {record_id_value!r} "
                    f"(first seen on line {first_line})"
                )
            else:
                record_ids_seen[record_id_value] = line_number

        axis = record.get("axis")
        if isinstance(axis, str):
            first_line = axes_seen.get(axis)
            if first_line is not None:
                errors.append(
                    f"{prefix}: duplicate axis {axis!r} "
                    f"(first seen on line {first_line})"
                )
            else:
                axes_seen[axis] = line_number

        enforcement = record.get("enforcement")
        if isinstance(enforcement, dict):
            command = enforcement.get("command")
            if (
                isinstance(command, list)
                and all(isinstance(part, str) for part in command)
                and not command_target_exists(command, repo_root)
            ):
                errors.append(
                    f"{prefix}: enforcement command target does not exist: {command[0]}"
                )

    if record_count == 0:
        errors.append("manifest: no records found")

    missing_axes = sorted(axes_required - set(axes_seen))
    if missing_axes:
        errors.append(f"manifest: missing axes: {', '.join(missing_axes)}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", default=DEFAULT_MANIFEST, type=Path)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA, type=Path)
    parser.add_argument("--repo-root", default=Path.cwd(), type=Path)
    args = parser.parse_args(argv)

    try:
        errors = validate_manifest(args.manifest, args.schema, args.repo_root)
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
