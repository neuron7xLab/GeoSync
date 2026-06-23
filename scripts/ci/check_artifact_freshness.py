#!/usr/bin/env python3
# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Deterministic evidence-artifact freshness gate (P0-7).

A governance repo must PROVE that committed evidence artifacts were regenerated
from their sources, not hand-edited. This gate regenerates each DETERMINISTIC
evidence artifact with its declared generator and asserts the result is
byte-identical to the committed file (the equivalent of a CI `git diff
--exit-code` over those paths) — and that ``artifacts/.manifest.json`` records
the matching sha256.

Only artifacts whose generators are byte-deterministic (no wall-clock, no RNG)
are covered. Nondeterministic artifacts (wheel_contract.json embeds a per-build
wheel hash; financial_verification_manifest.json regenerates with differing
structure) are DELIBERATELY excluded — covering them would make the gate flaky,
the opposite of its purpose.

Usage::

    python scripts/ci/check_artifact_freshness.py            # verify (fail-closed)
    python scripts/ci/check_artifact_freshness.py --write     # regenerate manifest

Exit codes: 0 fresh; 1 a committed artifact != regenerated, or manifest drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "artifacts" / ".manifest.json"

# (artifact, generator argv relative to ROOT, source files the generator reads).
DETERMINISTIC_ARTIFACTS: list[dict[str, object]] = [
    {
        "artifact": "artifacts/academy_fellow/distinguished_evidence_table.json",
        "generator": ["scripts/ci/gen_distinguished_evidence_table.py"],
        "sources": [".claude/neuro/NEURO_OPERATIONALIZATION_LEDGER.yaml"],
    },
    {
        "artifact": "artifacts/claims_hashes.json",
        "generator": ["scripts/generate_claim_hashes.py"],
        "sources": ["CLAIMS.md"],
    },
    {
        "artifact": "docs/architecture/claim_graph.json",
        "generator": ["scripts/generate_claim_graph.py"],
        "sources": ["CLAIMS.md"],
    },
]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run_generator(generator: list[str]) -> None:
    full_env = {**os.environ, "PYTHONPATH": str(ROOT)}
    subprocess.run(
        [sys.executable, *generator],
        cwd=ROOT,
        env=full_env,
        check=True,
        capture_output=True,
    )


def regenerated_sha(entry: dict[str, object]) -> tuple[str, str]:
    """Return (committed_sha, regenerated_sha); restores the committed bytes."""
    path = ROOT / str(entry["artifact"])
    committed = path.read_bytes()
    try:
        _run_generator([str(g) for g in entry["generator"]])  # type: ignore[union-attr]
        regenerated = path.read_bytes()
    finally:
        path.write_bytes(committed)  # never leave the tree dirty
    return _sha256(committed), _sha256(regenerated)


def build_manifest() -> dict[str, object]:
    rows = []
    for entry in DETERMINISTIC_ARTIFACTS:
        path = ROOT / str(entry["artifact"])
        rows.append(
            {
                "artifact": entry["artifact"],
                "generator_command": "python " + " ".join(str(g) for g in entry["generator"]),  # type: ignore[union-attr]
                "sources": entry["sources"],
                "artifact_sha256": _sha256(path.read_bytes()),
            }
        )
    return {
        "schema": "evidence_artifact_freshness@v1",
        "generated_at_policy": "deterministic; no wall-clock, no RNG",
        "artifacts": rows,
    }


def check() -> list[str]:
    errors: list[str] = []
    for entry in DETERMINISTIC_ARTIFACTS:
        path = ROOT / str(entry["artifact"])
        if not path.is_file():
            errors.append(f"missing artifact: {entry['artifact']}")
            continue
        committed_sha, regen_sha = regenerated_sha(entry)
        if committed_sha != regen_sha:
            errors.append(
                f"STALE: {entry['artifact']} committed sha {committed_sha[:12]} != "
                f"regenerated {regen_sha[:12]} (run its generator and commit)"
            )
    # Manifest must exist and match the committed artifacts.
    if not MANIFEST.is_file():
        errors.append(f"missing {MANIFEST.relative_to(ROOT)} (run --write)")
    else:
        committed_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if committed_manifest != build_manifest():
            errors.append("artifacts/.manifest.json drift — regenerate with --write")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the manifest")
    args = parser.parse_args(argv)

    if args.write:
        MANIFEST.write_text(json.dumps(build_manifest(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {MANIFEST.relative_to(ROOT)}")
        return 0

    errors = check()
    if errors:
        print("ARTIFACT FRESHNESS FAIL:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1
    print(
        f"ARTIFACT FRESHNESS: PASS — {len(DETERMINISTIC_ARTIFACTS)} deterministic artifacts fresh."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
