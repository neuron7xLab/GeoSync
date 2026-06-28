#!/usr/bin/env python3
# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Generate a repository readiness verdict from explicit evidence rows.

A row is PASS only when the expected artifact exists and, for command-backed
repository checks, the command has actually been executed with --execute. This
keeps report-only mode useful for artifact discovery while preventing command
claims from being treated as complete evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "production_readiness"
SUMMARY_PATH = RESULT_DIR / "READINESS_SUMMARY.json"
MATRIX_PATH = RESULT_DIR / "GATE_MATRIX.csv"
VERDICT_PATH = RESULT_DIR / "RELEASE_VERDICT.md"

PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class ReadinessClaim:
    claim_id: str
    scope: str
    check_command: str
    expected_artifact: str
    command_backed: bool = True


@dataclass
class ReadinessRow:
    claim_id: str
    scope: str
    check_command: str
    expected_artifact: str
    artifact_hash: str
    commit_sha: str
    verdict: str
    detail: str


CLAIMS: tuple[ReadinessClaim, ...] = (
    ReadinessClaim(
        "production-readiness-doc",
        "docs",
        "test -f docs/PRODUCTION_READINESS_STACK.md",
        "docs/PRODUCTION_READINESS_STACK.md",
        command_backed=False,
    ),
    ReadinessClaim(
        "release-verdict-protocol",
        "docs",
        "test -f docs/RELEASE_VERDICT_PROTOCOL.md",
        "docs/RELEASE_VERDICT_PROTOCOL.md",
        command_backed=False,
    ),
    ReadinessClaim(
        "readiness-verifier",
        "repository",
        "python scripts/verify_production_readiness.py --write-results",
        "scripts/verify_production_readiness.py",
    ),
    ReadinessClaim(
        "commit-acceptor",
        "repository",
        "test -f .claude/commit_acceptors/production-readiness-stack.yaml",
        ".claude/commit_acceptors/production-readiness-stack.yaml",
        command_backed=False,
    ),
    ReadinessClaim(
        "test-suite-command",
        "repository",
        "python -m pytest -q",
        "results/production_readiness/READINESS_SUMMARY.json",
    ),
    ReadinessClaim(
        "lint-command",
        "repository",
        "ruff check .",
        "results/production_readiness/READINESS_SUMMARY.json",
    ),
    ReadinessClaim(
        "type-check-command",
        "repository",
        "python -m mypy src tests --strict",
        "results/production_readiness/READINESS_SUMMARY.json",
    ),
    ReadinessClaim(
        "release-verdict-artifact",
        "artifact",
        "test -f results/production_readiness/RELEASE_VERDICT.md",
        "results/production_readiness/RELEASE_VERDICT.md",
        command_backed=False,
    ),
)


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def current_commit() -> str:
    code, out = _run(["git", "rev-parse", "HEAD"])
    if code == 0 and out:
        return out.splitlines()[0].strip()
    return "UNKNOWN"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def evaluate_claim(claim: ReadinessClaim, *, commit_sha: str, execute: bool) -> ReadinessRow:
    artifact = ROOT / claim.expected_artifact
    if not artifact.exists():
        return ReadinessRow(
            claim.claim_id,
            claim.scope,
            claim.check_command,
            claim.expected_artifact,
            "",
            commit_sha,
            BLOCKED,
            "expected artifact is absent",
        )

    artifact_hash = sha256_file(artifact)
    if not execute:
        if claim.command_backed:
            return ReadinessRow(
                claim.claim_id,
                claim.scope,
                claim.check_command,
                claim.expected_artifact,
                artifact_hash,
                commit_sha,
                BLOCKED,
                "command-backed claim was not executed in report-only mode",
            )
        return ReadinessRow(
            claim.claim_id,
            claim.scope,
            claim.check_command,
            claim.expected_artifact,
            artifact_hash,
            commit_sha,
            PASS,
            "artifact present",
        )

    code, output = _run(claim.check_command.split())
    if code == 0:
        return ReadinessRow(
            claim.claim_id,
            claim.scope,
            claim.check_command,
            claim.expected_artifact,
            artifact_hash,
            commit_sha,
            PASS,
            "command passed",
        )
    detail = output.splitlines()[-1][:200] if output else "command returned non-zero"
    return ReadinessRow(
        claim.claim_id,
        claim.scope,
        claim.check_command,
        claim.expected_artifact,
        artifact_hash,
        commit_sha,
        FAIL,
        detail,
    )


def build_summary(rows: list[ReadinessRow], *, commit_sha: str) -> dict[str, object]:
    counts = {PASS: 0, FAIL: 0, BLOCKED: 0}
    for row in rows:
        counts[row.verdict] = counts.get(row.verdict, 0) + 1
    release_ready = counts.get(FAIL, 0) == 0 and counts.get(BLOCKED, 0) == 0
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "commit_sha": commit_sha,
        "release_ready": release_ready,
        "verdict": PASS if release_ready else BLOCKED,
        "counts": counts,
        "rows": [asdict(row) for row in rows],
    }


def write_outputs(summary: dict[str, object], rows: list[ReadinessRow]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with MATRIX_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    lines = [
        "# Release Verdict",
        "",
        f"Commit: `{summary['commit_sha']}`",
        f"Generated: `{summary['generated_at']}`",
        f"Verdict: **{summary['verdict']}**",
        f"Release ready: `{summary['release_ready']}`",
        "",
        "| Claim | Scope | Verdict | Artifact |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row.claim_id}` | `{row.scope}` | `{row.verdict}` | `{row.expected_artifact}` |"
        )
    lines.append("")
    VERDICT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate GeoSync readiness verdict.")
    parser.add_argument("--execute", action="store_true", help="Run declared commands.")
    parser.add_argument("--write-results", action="store_true", help="Write JSON/CSV/Markdown outputs.")
    parser.add_argument("--enforce", action="store_true", help="Return non-zero unless every row is PASS.")
    args = parser.parse_args(argv)

    commit_sha = current_commit()
    rows = [evaluate_claim(claim, commit_sha=commit_sha, execute=args.execute) for claim in CLAIMS]
    summary = build_summary(rows, commit_sha=commit_sha)

    if args.write_results:
        write_outputs(summary, rows)

    print(json.dumps({"verdict": summary["verdict"], "counts": summary["counts"]}, sort_keys=True))
    if args.enforce and summary["verdict"] != PASS:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
