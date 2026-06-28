#!/usr/bin/env python3
# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Generate a repository readiness verdict from explicit evidence rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
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
        claim_id="production-readiness-doc",
        scope="docs",
        check_command="test -f docs/PRODUCTION_READINESS_STACK.md",
        expected_artifact="docs/PRODUCTION_READINESS_STACK.md",
        command_backed=False,
    ),
    ReadinessClaim(
        claim_id="release-verdict-protocol",
        scope="docs",
        check_command="test -f docs/RELEASE_VERDICT_PROTOCOL.md",
        expected_artifact="docs/RELEASE_VERDICT_PROTOCOL.md",
        command_backed=False,
    ),
    ReadinessClaim(
        claim_id="readiness-verifier",
        scope="repository",
        check_command="python scripts/verify_production_readiness.py --write-results",
        expected_artifact="scripts/verify_production_readiness.py",
    ),
    ReadinessClaim(
        claim_id="commit-acceptor",
        scope="repository",
        check_command="test -f .claude/commit_acceptors/production-readiness-stack.yaml",
        expected_artifact=".claude/commit_acceptors/production-readiness-stack.yaml",
        command_backed=False,
    ),
    ReadinessClaim(
        claim_id="test-suite-command",
        scope="repository",
        check_command="python -m pytest -q",
        expected_artifact="results/production_readiness/READINESS_SUMMARY.json",
    ),
    ReadinessClaim(
        claim_id="lint-command",
        scope="repository",
        check_command="ruff check .",
        expected_artifact="results/production_readiness/READINESS_SUMMARY.json",
    ),
    ReadinessClaim(
        claim_id="type-check-command",
        scope="repository",
        check_command="python -m mypy src tests --strict",
        expected_artifact="results/production_readiness/READINESS_SUMMARY.json",
    ),
    ReadinessClaim(
        claim_id="release-verdict-artifact",
        scope="artifact",
        check_command="test -f results/production_readiness/RELEASE_VERDICT.md",
        expected_artifact="results/production_readiness/RELEASE_VERDICT.md",
        command_backed=False,
    ),
)


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    return proc.returncode, output.strip()


def current_commit() -> str:
    code, output = _run(["git", "rev-parse", "HEAD"])
    if code == 0 and output:
        return output.splitlines()[0].strip()
    return "UNKNOWN"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def blocked_row(
    claim: ReadinessClaim,
    *,
    commit_sha: str,
    detail: str,
) -> ReadinessRow:
    return ReadinessRow(
        claim_id=claim.claim_id,
        scope=claim.scope,
        check_command=claim.check_command,
        expected_artifact=claim.expected_artifact,
        artifact_hash="",
        commit_sha=commit_sha,
        verdict=BLOCKED,
        detail=detail,
    )


def evaluate_claim(
    claim: ReadinessClaim,
    *,
    commit_sha: str,
    execute: bool,
) -> ReadinessRow:
    artifact = ROOT / claim.expected_artifact
    if not artifact.exists():
        return blocked_row(
            claim,
            commit_sha=commit_sha,
            detail="expected artifact is absent",
        )

    artifact_hash = sha256_file(artifact)
    if not execute:
        verdict = BLOCKED if claim.command_backed else PASS
        detail = (
            "command-backed claim was not executed"
            if claim.command_backed
            else "artifact present"
        )
        return ReadinessRow(
            claim_id=claim.claim_id,
            scope=claim.scope,
            check_command=claim.check_command,
            expected_artifact=claim.expected_artifact,
            artifact_hash=artifact_hash,
            commit_sha=commit_sha,
            verdict=verdict,
            detail=detail,
        )

    code, output = _run(claim.check_command.split())
    if code == 0:
        verdict = PASS
        detail = "command passed"
    else:
        verdict = FAIL
        detail = output.splitlines()[-1][:200] if output else "non-zero exit"

    return ReadinessRow(
        claim_id=claim.claim_id,
        scope=claim.scope,
        check_command=claim.check_command,
        expected_artifact=claim.expected_artifact,
        artifact_hash=artifact_hash,
        commit_sha=commit_sha,
        verdict=verdict,
        detail=detail,
    )


def build_summary(
    rows: list[ReadinessRow],
    *,
    commit_sha: str,
) -> dict[str, Any]:
    counts = {PASS: 0, FAIL: 0, BLOCKED: 0}
    for row in rows:
        counts[row.verdict] = counts.get(row.verdict, 0) + 1
    release_ready = counts[FAIL] == 0 and counts[BLOCKED] == 0
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "commit_sha": commit_sha,
        "release_ready": release_ready,
        "verdict": PASS if release_ready else BLOCKED,
        "counts": counts,
        "rows": [asdict(row) for row in rows],
    }


def write_outputs(summary: dict[str, Any], rows: list[ReadinessRow]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summary_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    SUMMARY_PATH.write_text(summary_text, encoding="utf-8")

    with MATRIX_PATH.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(asdict(rows[0]).keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
            "| "
            f"`{row.claim_id}` | `{row.scope}` | "
            f"`{row.verdict}` | `{row.expected_artifact}` |"
        )
    lines.append("")
    VERDICT_PATH.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GeoSync readiness verdict.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run declared commands.",
    )
    parser.add_argument(
        "--write-results",
        action="store_true",
        help="Write JSON/CSV/Markdown outputs.",
    )
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Return non-zero unless every row is PASS.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    commit_sha = current_commit()
    rows = [
        evaluate_claim(
            claim,
            commit_sha=commit_sha,
            execute=bool(args.execute),
        )
        for claim in CLAIMS
    ]
    summary = build_summary(rows, commit_sha=commit_sha)

    if args.write_results:
        write_outputs(summary, rows)

    console = {
        "counts": summary["counts"],
        "verdict": summary["verdict"],
    }
    print(json.dumps(console, sort_keys=True))
    if args.enforce and summary["verdict"] != PASS:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
