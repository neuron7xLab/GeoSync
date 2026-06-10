#!/usr/bin/env python3
# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Deterministic local mirror of the PR-gate — verify a changeset before pushing.

This runs the *exact* fast, deterministic gates the CI `pr-gate` workflow runs,
against the same changed-file set CI sees (``git diff <base>...HEAD``), so that a
green local run reproduces a green CI run. It exists because "passed locally but
failed in CI" (a skipped ``black``, an unbound commit-acceptor, an undeclared
inventory entry, a detect-secrets hit) is exactly the failure class that wastes a
review cycle — and because trust requires that what we *say* is verified actually
*was* verified.

Trust contract (fail-closed): the overall result is ``ok`` ONLY when every
applicable gate ran AND passed. A gate whose tool is missing is reported
``UNVERIFIED`` and fails the run — we never report success for something we could
not actually check.

Mirrored gates (CI job in parentheses):

* ruff / black --check / mypy on changed ``*.py``           (python-quality)
* bandit -ll on changed ``core|backtest|execution|src`` py  (secrets-supply-chain)
* detect-secrets-hook against the baseline                  (secrets-supply-chain)
* commit-acceptor diff-binding                              (commit-acceptor-gate)
* import-architecture ratchet                              (import-architecture-gate)
* INVENTORY sync + manifest hashes                         (repo-policy)
* docs-consistency (ADR-0024 canonical root)               (docs-consistency-gate)

Usage::

    python scripts/ci/verify_changeset.py                 # vs origin/main
    python scripts/ci/verify_changeset.py --base-ref HEAD~1
    python scripts/ci/verify_changeset.py --json artifacts/changeset_verify.json

Exit codes::

    0  — every applicable gate passed
    1  — at least one gate FAILED or was UNVERIFIED (fail-closed)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_DETECT_SECRETS_EXCLUDE = (
    r"^(INVENTORY\.json|\.github/detect-secrets\.baseline"
    r"|figures/disha_ba_correlation/repro_capsule/.*)$"
)
# CI's bandit job only scans security-critical first-party trees.
_BANDIT_PREFIXES = ("core/", "backtest/", "execution/", "src/")

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"
UNVERIFIED = "UNVERIFIED"


@dataclass
class GateResult:
    name: str
    status: str
    detail: str = ""
    command: list[str] = field(default_factory=list)


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    out = (proc.stdout + proc.stderr).strip()
    return proc.returncode, out


def _changed_files(base_ref: str) -> list[str]:
    code, out = _run(["git", "diff", "--name-only", f"{base_ref}...HEAD"])
    if code != 0:
        # Fall back to a two-dot diff if the base ref has no common ancestor view.
        code, out = _run(["git", "diff", "--name-only", base_ref, "HEAD"])
    files = sorted({line.strip() for line in out.splitlines() if line.strip()})
    # Only files that still exist in the working tree are checkable.
    return [f for f in files if (ROOT / f).is_file()]


def _tail(text: str, lines: int = 8) -> str:
    rows = [r for r in text.splitlines() if r.strip()]
    return "\n".join(rows[-lines:])


def _tool_gate(name: str, tool: str, args: list[str], *, applicable: bool) -> GateResult:
    if not applicable:
        return GateResult(name, SKIP, "not applicable to this changeset")
    if shutil.which(tool) is None and not (ROOT / ".venv" / "bin" / tool).exists():
        return GateResult(name, UNVERIFIED, f"tool '{tool}' not installed")
    cmd = [tool, *args]
    code, out = _run(cmd)
    status = PASS if code == 0 else FAIL
    return GateResult(name, status, _tail(out) if code != 0 else "", cmd)


def _script_gate(name: str, args: list[str], *, applicable: bool = True) -> GateResult:
    if not applicable:
        return GateResult(name, SKIP, "not applicable to this changeset")
    cmd = ["python", *args]
    code, out = _run(cmd)
    status = PASS if code == 0 else FAIL
    return GateResult(name, status, _tail(out) if code != 0 else "", cmd)


def verify(base_ref: str) -> list[GateResult]:
    changed = _changed_files(base_ref)
    py_files = [f for f in changed if f.endswith(".py")]
    bandit_files = [f for f in py_files if f.startswith(_BANDIT_PREFIXES)]
    has_py = bool(py_files)
    has_md = any(f.endswith(".md") for f in changed)

    results: list[GateResult] = [
        _tool_gate("ruff", "ruff", ["check", *py_files], applicable=has_py),
        _tool_gate("black", "black", ["--check", *py_files], applicable=has_py),
        _tool_gate("mypy", "mypy", py_files, applicable=has_py),
        _tool_gate(
            "bandit",
            "bandit",
            ["-ll", *bandit_files],
            applicable=bool(bandit_files),
        ),
        _tool_gate(
            "detect-secrets",
            "detect-secrets-hook",
            [
                "--baseline",
                ".github/detect-secrets.baseline",
                "--exclude-files",
                _DETECT_SECRETS_EXCLUDE,
                *changed,
            ],
            applicable=bool(changed),
        ),
        _script_gate(
            "commit-acceptor",
            [
                "tools/commit_acceptor/validate_commit_acceptor.py",
                "--base-ref",
                base_ref,
                "--head-ref",
                "HEAD",
                "--require-acceptor-for-code-change",
            ],
        ),
        _script_gate("import-architecture", ["scripts/ci/check_import_architecture.py"]),
        _script_gate("inventory-sync", ["scripts/ci/check_inventory_sync.py"]),
        _script_gate("manifest-hashes", ["scripts/ci/check_manifest_hashes.py"]),
        _script_gate(
            "docs-consistency",
            ["scripts/ci/check_docs_consistency.py"],
            applicable=has_md,
        ),
    ]
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic local PR-gate mirror.")
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="Base ref to diff against (default: origin/main).",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write the gate report as JSON to this path.",
    )
    args = parser.parse_args(argv)

    results = verify(args.base_ref)

    failed = [r for r in results if r.status in (FAIL, UNVERIFIED)]
    ok = not failed

    width = max(len(r.name) for r in results)
    icon = {PASS: "✓", FAIL: "✗", SKIP: "·", UNVERIFIED: "?"}
    print(f"Changeset gate vs {args.base_ref}:")
    for r in results:
        line = f"  {icon.get(r.status, '?')} {r.name.ljust(width)}  {r.status}"
        if r.detail and r.status in (FAIL, UNVERIFIED):
            line += f"  — {r.detail.splitlines()[0][:100]}"
        print(line)

    if args.json is not None:
        payload = {
            "version": 1,
            "base_ref": args.base_ref,
            "ok": ok,
            "gates": [{"name": r.name, "status": r.status, "detail": r.detail} for r in results],
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if not ok:
        print(
            "\nFAILED gates: "
            + ", ".join(f"{r.name}({r.status})" for r in failed)
            + "\nFix before pushing — these mirror the CI pr-gate exactly."
        )
        for r in failed:
            if r.detail:
                print(f"\n[{r.name}]\n{r.detail}")
        return 1

    print("\nAll applicable gates passed — safe to push.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
