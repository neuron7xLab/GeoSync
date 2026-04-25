#!/usr/bin/env python3
# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Fail-closed claim/evidence gate.

Validates ``docs/CLAIMS.yaml`` against the canonical schema and
verifies that every ``P0`` / ``P1`` claim's ``evidence_paths`` exist
in the working tree. Missing evidence on a P0 / P1 claim is a build
failure.

Run locally before push:

    python scripts/ci/check_claims.py

Exit codes:

    0  — all gated claims have intact evidence
    1  — schema violation OR missing evidence on a P0/P1 claim

The intent is the inverse of an aspirational README: the registry
states what the repository claims to be, and the CI gate refuses to
let those claims drift away from the artefacts that back them.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CLAIMS_PATH = ROOT / "docs" / "CLAIMS.yaml"

SCHEMA_VERSION_EXPECTED = 1
ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
PRIORITIES_GATED = frozenset({"P0", "P1"})
PRIORITIES_ALL = frozenset({"P0", "P1", "P2"})
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class Claim:
    """In-memory representation of a single registry entry."""

    id: str
    priority: str
    description: str
    evidence_paths: tuple[str, ...]
    added_utc: str


@dataclass(frozen=True)
class ValidationFailure:
    claim_id: str
    reason: str

    def __str__(self) -> str:
        return f"[{self.claim_id}] {self.reason}"


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing registry: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded: object = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(f"registry must be a YAML mapping, got {type(loaded).__name__}")
    return loaded


def _parse_claims(registry: dict[str, Any]) -> list[Claim]:
    if registry.get("schema_version") != SCHEMA_VERSION_EXPECTED:
        raise ValueError(
            f"schema_version={registry.get('schema_version')!r} "
            f"!= expected {SCHEMA_VERSION_EXPECTED}"
        )
    raw_claims = registry.get("claims")
    if not isinstance(raw_claims, list):
        raise ValueError("'claims' must be a list at the top level of CLAIMS.yaml")
    claims: list[Claim] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(raw_claims):
        if not isinstance(entry, dict):
            raise ValueError(f"claim #{index}: expected mapping, got {type(entry).__name__}")
        claim = _parse_one(entry, index)
        if claim.id in seen_ids:
            raise ValueError(f"duplicate claim id: {claim.id!r}")
        seen_ids.add(claim.id)
        claims.append(claim)
    return claims


def _parse_one(entry: dict[str, Any], index: int) -> Claim:
    required = {"id", "priority", "description", "evidence_paths", "added_utc"}
    missing = required - set(entry.keys())
    if missing:
        raise ValueError(f"claim #{index} missing fields: {sorted(missing)}")
    cid = entry["id"]
    if not isinstance(cid, str) or not ID_PATTERN.match(cid):
        raise ValueError(
            f"claim #{index}: id={cid!r} must be kebab-case "
            f"(lowercase, digits, hyphens, no leading/trailing hyphen)"
        )
    priority = entry["priority"]
    if priority not in PRIORITIES_ALL:
        raise ValueError(f"claim {cid}: priority={priority!r} not in {sorted(PRIORITIES_ALL)}")
    description = entry["description"]
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"claim {cid}: description must be a non-empty string")
    evidence = entry["evidence_paths"]
    if not isinstance(evidence, list) or not all(isinstance(p, str) for p in evidence):
        raise ValueError(f"claim {cid}: evidence_paths must be a list of strings")
    if not evidence:
        raise ValueError(f"claim {cid}: evidence_paths must contain at least one path")
    added = entry["added_utc"]
    if not isinstance(added, str) or not DATE_PATTERN.match(added):
        raise ValueError(f"claim {cid}: added_utc={added!r} must be 'YYYY-MM-DD'")
    return Claim(
        id=cid,
        priority=priority,
        description=description,
        evidence_paths=tuple(evidence),
        added_utc=added,
    )


def _validate_evidence(claim: Claim, root: Path) -> list[ValidationFailure]:
    """Return one failure per missing evidence path on a gated claim."""
    if claim.priority not in PRIORITIES_GATED:
        return []
    failures: list[ValidationFailure] = []
    for relative in claim.evidence_paths:
        target = root / relative
        if not target.exists():
            failures.append(
                ValidationFailure(
                    claim_id=claim.id,
                    reason=f"missing evidence path: {relative}",
                )
            )
    return failures


def main(argv: list[str] | None = None) -> int:
    del argv  # accept-no-args contract; flags would invite ad-hoc skipping.
    try:
        registry = _load_registry(CLAIMS_PATH)
        claims = _parse_claims(registry)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        print(f"CLAIMS.yaml schema violation: {exc}", file=sys.stderr)
        return 1

    failures: list[ValidationFailure] = []
    for claim in claims:
        failures.extend(_validate_evidence(claim, ROOT))

    gated_total = sum(1 for c in claims if c.priority in PRIORITIES_GATED)
    if failures:
        print(
            f"FAIL: {len(failures)} evidence violation(s) across {gated_total} gated claim(s):",
            file=sys.stderr,
        )
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1

    print(
        f"PASS: {gated_total} gated claim(s), {len(claims) - gated_total} P2; "
        f"all evidence paths present."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
