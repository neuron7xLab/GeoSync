#!/usr/bin/env python3
# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Forbidden-terminology lint (IERD-PAI-FPS-UX-001 §3).

Scans Markdown surface text and Python docstrings for the IERD
forbidden-terminology list. In **warn** mode (Phase 0) it lists
offending lines and exits 0; in **strict** mode (Phase 5) it exits
non-zero so CI gates the merge.

Forbidden terms (literal, case-insensitive) — see directive §3:

    truth function
    real physics law
    thermodynamic invariant
    serotonin gain        (literal, including the underscore form)
    neuro truth
    universal intelligence law
    physics-aligned production-ready

Anchored use is allowed: a forbidden term in the same paragraph as a
matching exception marker (`[claim_id=…]`, `[INV-…]`, or an inline
citation like `[@AuthorYearKey]`) is skipped, because the surface
text is then traceable to a falsifying test or peer-reviewed source.

Usage:

    python scripts/ci/lint_forbidden_terms.py            # warn-only
    python scripts/ci/lint_forbidden_terms.py --strict   # fail-closed
    python scripts/ci/lint_forbidden_terms.py --paths README.md docs/

Phase 0 default is warn-only. Move to strict in Phase 5 per directive §7.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("truth function", re.compile(r"\btruth[\s\-_]+function\b", re.IGNORECASE)),
    ("real physics law", re.compile(r"\breal\s+physics\s+law\b", re.IGNORECASE)),
    (
        "thermodynamic invariant",
        re.compile(r"\bthermodynamic\s+invariant\b", re.IGNORECASE),
    ),
    ("serotonin gain", re.compile(r"\bserotonin[\s\-_]+gain\b", re.IGNORECASE)),
    ("neuro truth", re.compile(r"\bneuro[\s\-_]+truth\b", re.IGNORECASE)),
    (
        "universal intelligence law",
        re.compile(r"\buniversal\s+intelligence\s+law\b", re.IGNORECASE),
    ),
    (
        "physics-aligned production-ready",
        re.compile(r"\bphysics[\s\-]+aligned[\s\-]+production[\s\-]+ready\b", re.IGNORECASE),
    ),
)

EXCEPTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\[claim_id=[a-z0-9\-]+\]", re.IGNORECASE),
    re.compile(r"\[INV-[A-Z0-9\-]+\]"),
    re.compile(r"\[@[A-Za-z][A-Za-z0-9\-_]*\]"),
)

DEFAULT_INCLUDE_GLOBS: tuple[str, ...] = (
    "README.md",
    "docs/**/*.md",
    "docs/**/*.yaml",
    "reports/**/*.md",
)

# Paths that are allowed to quote forbidden terms (e.g. the directive
# itself, the audit findings, this lint, the IERD response).
ALLOWLIST_PATHS: frozenset[str] = frozenset(
    {
        "docs/governance/IERD-PAI-FPS-UX-001.md",
        "docs/audit/ierd_phase0_findings.md",
        "docs/yana-response.md",
        "docs/KNOWN_LIMITATIONS.md",
        "docs/CLAIMS.yaml",
        "scripts/ci/lint_forbidden_terms.py",
        "scripts/ci/check_claims.py",
        ".claude/physics/lint_forbidden_terms.py",
    }
)


@dataclass(frozen=True)
class Finding:
    path: Path
    line_no: int
    term: str
    excerpt: str

    def render(self) -> str:
        return f"{self.path}:{self.line_no}: '{self.term}' — {self.excerpt.strip()[:120]}"


def _iter_targets(root: Path, globs: Sequence[str]) -> Iterable[Path]:
    seen: set[Path] = set()
    for pattern in globs:
        for match in root.glob(pattern):
            if match.is_file() and match not in seen:
                seen.add(match)
                yield match


def _is_allowlisted(path: Path, root: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    return rel in ALLOWLIST_PATHS


def _has_exception(line: str) -> bool:
    return any(p.search(line) for p in EXCEPTION_PATTERNS)


def _scan_file(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return findings
    for line_no, line in enumerate(text.splitlines(), start=1):
        if _has_exception(line):
            continue
        for term, pattern in FORBIDDEN_PATTERNS:
            if pattern.search(line):
                findings.append(
                    Finding(
                        path=path,
                        line_no=line_no,
                        term=term,
                        excerpt=line,
                    )
                )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lint forbidden terminology per IERD-PAI-FPS-UX-001 §3."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Phase 5: exit non-zero on any finding (default warn).",
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="Override default include globs (relative to repo root).",
    )
    args = parser.parse_args(argv)

    globs = tuple(args.paths) if args.paths else DEFAULT_INCLUDE_GLOBS

    findings: list[Finding] = []
    scanned = 0
    for target in _iter_targets(ROOT, globs):
        if _is_allowlisted(target, ROOT):
            continue
        scanned += 1
        findings.extend(_scan_file(target))

    if not findings:
        print(f"PASS: {scanned} file(s) scanned, no forbidden terms found.")
        return 0

    label = "FAIL" if args.strict else "WARN"
    print(
        f"{label}: {len(findings)} forbidden-term finding(s) across {scanned} scanned file(s):",
        file=sys.stderr,
    )
    for f in findings:
        print(f"  {f.render()}", file=sys.stderr)

    if args.strict:
        print(
            "\nResolve by replacing per IERD §4.2 mapping or by adding "
            "an exception marker ([claim_id=…], [INV-…], [@CitationKey]).",
            file=sys.stderr,
        )
        return 1

    print(
        "\n[Phase 0 warn-only mode — these findings do not block CI. "
        "Strict mode lands in Phase 5.]",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
