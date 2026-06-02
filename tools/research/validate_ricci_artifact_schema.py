#!/usr/bin/env python3
"""Validate Ricci microstructure inference artifacts against the canonical schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn

from jsonschema import Draft202012Validator, FormatChecker

DEFAULT_SCHEMA = Path("schemas/research/ricci_microstructure_artifact.schema.json")
DEFAULT_ARTIFACT = Path("artifacts/runs/ricci_microstructure_v1/example_artifact.json")


def _die(message: str) -> NoReturn:
    print(f"ricci-artifact-schema: {message}", file=sys.stderr)
    raise SystemExit(1)


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _die(f"missing file: {path}")
    except json.JSONDecodeError as exc:
        _die(f"invalid JSON in {path}: {exc}")


def validate_artifact(schema_path: Path, artifact_path: Path) -> list[str]:
    """Return schema validation errors for one artifact path."""
    schema = _load_json(schema_path)
    artifact = _load_json(artifact_path)
    if not isinstance(schema, dict):
        _die(f"schema root must be an object: {schema_path}")

    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    return [
        f"{artifact_path}: {'/'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(artifact), key=lambda item: item.path)
    ]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--artifact",
        type=Path,
        action="append",
        default=None,
        help="Artifact JSON to validate. May be passed more than once.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    artifacts = args.artifact or [DEFAULT_ARTIFACT]

    failures: list[str] = []
    for artifact_path in artifacts:
        failures.extend(validate_artifact(args.schema, artifact_path))

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1

    print(f"Validated {len(artifacts)} Ricci artifact(s) against {args.schema}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
