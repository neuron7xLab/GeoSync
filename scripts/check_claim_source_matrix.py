#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate GeoSync bibliography and claim-source matrix."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALID_TIERS = {
    "FACT",
    "MEASURED",
    "DERIVED",
    "SIMULATION",
    "SYNTHETIC_DEMO",
    "HYPOTHESIS",
    "SPECULATIVE",
    "RETIRED",
}


def die(msg: str) -> None:
    print(f"claim-source-matrix: ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def read(path: str) -> str:
    target = ROOT / path
    if not target.exists():
        die(f"missing required file: {path}")
    return target.read_text(encoding="utf-8")


def main() -> int:
    bib_md = read("docs/BIBLIOGRAPHY.md")
    claim_md = read("CLAIMS.md")
    try:
        matrix = json.loads(read("docs/claim_source_matrix.json"))
    except json.JSONDecodeError as exc:
        die(f"docs/claim_source_matrix.json is invalid JSON: {exc}")

    bib_ids = set(
        re.findall(r"^### (BIB-[A-Z0-9-]+)\b", bib_md, re.MULTILINE)
    )
    claim_ids = set(
        re.findall(r"^\| ([CR]-[A-Z0-9-]+) \|", claim_md, re.MULTILINE)
    )
    rows = matrix.get("entries")
    if not isinstance(rows, list) or not rows:
        die("matrix entries must be a non-empty list")

    seen: set[str] = set()
    for row in rows:
        claim_id = row.get("claim_id")
        tier = row.get("claim_tier")
        status = row.get("claim_status")
        sources = row.get("sources")
        evidence = row.get("local_evidence")
        boundary = row.get("boundary")
        if claim_id in seen:
            die(f"duplicate claim row: {claim_id}")
        seen.add(claim_id)
        if claim_id not in claim_ids:
            die(f"matrix claim missing from CLAIMS.md: {claim_id}")
        if tier not in VALID_TIERS:
            die(f"{claim_id}: invalid tier {tier}")
        if status not in {"active", "retired"}:
            die(f"{claim_id}: invalid status {status}")
        if status == "retired" and tier != "RETIRED":
            die(f"{claim_id}: retired status requires RETIRED tier")
        if not isinstance(sources, list):
            die(f"{claim_id}: sources must be a list")
        unknown = sorted(set(sources) - bib_ids)
        if unknown:
            die(f"{claim_id}: unknown BIB ids: {', '.join(unknown)}")
        if not isinstance(evidence, list) or not evidence:
            die(f"{claim_id}: local_evidence must be non-empty")
        if not isinstance(boundary, str) or len(boundary.strip()) < 24:
            die(f"{claim_id}: boundary is missing or too weak")

    missing = sorted(claim_ids - seen)
    if missing:
        die(f"CLAIMS.md ids missing from matrix: {', '.join(missing)}")
    print(
        "claim-source-matrix: OK "
        f"({len(bib_ids)} bibliography sources, {len(seen)} claim rows)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
