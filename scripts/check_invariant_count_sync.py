# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Fail-closed CI gate: documentation invariant counts must match the registry.

Triggered by the 2026-04-30 external audit, which flagged four conflicting
counts ``57 / 66 / 67 / 87`` across the same repository:

* ``README.md`` (badge + Physics-Kernel section)
* ``BASELINE.md``
* ``CLAUDE.md`` invariant registry header

Authority is ``.claude/physics/INVARIANTS.yaml`` via :mod:`scripts.count_invariants`.
The gate is fail-closed: it exits non-zero on the first mismatch and never
auto-edits prose. Rewriting documentation is a human-reviewed action.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Final, Iterable

from count_invariants import REPO_ROOT, collect_invariant_ids

# Headline patterns only — historical/migration narrative ("the kernel had 34
# invariants", "this PR adds 10 invariants") describes a past state and is
# legitimate. The gate must flag ONLY load-bearing current-state claims.
_HEADLINE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    # **N invariants** / **N machine-checkable invariants**
    re.compile(r"\*\*(\d{2,4})\s+(?:machine-checkable\s+)?invariants?\*\*", re.IGNORECASE),
    # "N machine-checkable invariants" (subtitle/badge prose, no bold required)
    re.compile(r"\b(\d{2,4})\s+machine-checkable\s+invariants?\b", re.IGNORECASE),
    # "N invariants loaded by kernel self-check" (headline metric)
    re.compile(r"\b(\d{2,4})\s+invariants?\s+loaded\b", re.IGNORECASE),
    # Section header: "INVARIANT REGISTRY — N invariants"
    re.compile(r"INVARIANT\s+REGISTRY\s*[—\-]\s*(\d{2,4})\s+invariants?", re.IGNORECASE),
    # ASCII art block: "N INVARIANTS  ·"
    re.compile(r"\b(\d{2,4})\s+INVARIANTS\s*[·•]"),
)

DEFAULT_TARGETS: Final[tuple[str, ...]] = (
    "README.md",
    "BASELINE.md",
    "CLAUDE.md",
)

# Inline opt-out marker. Author appends this comment to a line that
# describes a HISTORICAL state (e.g. a 2026-04-06 migration milestone).
# Required wording is intentionally narrow so that someone cannot silence
# a real headline drift by accident.
_HISTORICAL_OPT_OUT: Final[str] = "count-sync:skip historical"


def find_count_claims(path: Path) -> list[tuple[int, int, str]]:
    """Return ``(line_number, claimed_count, full_line)`` triples found in ``path``.

    Matches only headline/load-bearing patterns; ignores migration narrative.
    """
    if not path.is_file():
        return []
    triples: list[tuple[int, int, str]] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if _HISTORICAL_OPT_OUT in raw_line:
            continue
        seen_counts: set[int] = set()
        for pattern in _HEADLINE_PATTERNS:
            for match in pattern.finditer(raw_line):
                count = int(match.group(1))
                if count in seen_counts:
                    continue
                seen_counts.add(count)
                triples.append((line_no, count, raw_line.strip()))
    return triples


def check(targets: Iterable[Path], expected: int) -> list[str]:
    """Return human-readable mismatch messages; empty list = success."""
    errors: list[str] = []
    for path in targets:
        for line_no, claimed, line in find_count_claims(path):
            if claimed != expected:
                rel = path.relative_to(REPO_ROOT) if path.is_absolute() else path
                errors.append(
                    f"{rel}:{line_no}: claims {claimed} invariants, "
                    f"registry has {expected} ({line})"
                )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="documentation files to scan (default: README.md, BASELINE.md, CLAUDE.md)",
    )
    args = parser.parse_args(argv)
    targets = (
        [REPO_ROOT / name for name in DEFAULT_TARGETS]
        if not args.paths
        else [p if p.is_absolute() else REPO_ROOT / p for p in args.paths]
    )
    expected = len(collect_invariant_ids())
    errors = check(targets, expected)
    if errors:
        print(
            f"INVARIANT COUNT DRIFT — registry has {expected} invariants but "
            f"documentation disagrees:",
            file=sys.stderr,
        )
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        print(
            "\nfix: regenerate the count or update the prose to match. "
            "single source of truth: .claude/physics/INVARIANTS.yaml.",
            file=sys.stderr,
        )
        return 1
    print(f"OK: {expected} invariants, all documentation surfaces in sync.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
