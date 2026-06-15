#!/usr/bin/env python3
# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Fail-closed product-category claim-boundary gate.

GeoSync's canonical positioning (``README.md``) is a **verification-first
quantitative research platform** — explicitly *not* a live-trading system,
*not* an alpha engine, and *not* investment advice. ``FORBIDDEN_CLAIMS.md``
already firewalls *status* wording (``validated alpha``, ``production
trading`` …). This gate closes the orthogonal gap: **product-category**
drift, where downstream prose quietly re-sells the platform as a
signal-emitting trading product and contradicts the README boundary.

Mechanism is not a claim. The repository legitimately ships an execution
realism harness (``interfaces/live_runner.py``), a ``Signal`` dataclass, a
``/v1/signals`` endpoint, and ``live/`` config directories — these are
internal substrate, not promises that GeoSync *is* a live-trading product.
This gate therefore follows the repository's existing ``detect-secrets``
baseline idiom: every boundary-sensitive phrase on the canonical doc
surface must either be reworded to the research boundary, or carry an
explicit, reasoned entry in ``.github/claim_boundary_allow.json``. A new,
unreviewed occurrence fails the build.

Run locally before push::

    python scripts/ci/check_claim_boundary.py

Exit codes::

    0  — no unreviewed product-category claims on the canonical surface
    1  — at least one unreviewed boundary-sensitive phrase, OR a stale
         allowlist entry that no longer matches anything (allowlist rot)
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST_PATH = ROOT / ".github" / "claim_boundary_allow.json"

# Canonical claim surface: top-level docs + the docs/ tree. This is where
# product-level positioning lives. Append-only *records* (audit findings
# that quote forbidden phrases as evidence, architectural decision records,
# and archived legacy reports) are excluded — scanning them would flag the
# very quotes that document the problem.
SCAN_GLOBS: tuple[str, ...] = ("*.md", "docs/**/*.md")
# Claim-bearing *code* surfaces: indicator modules whose docstrings and
# comments describe outputs to a reader. That prose is product-positioning
# just like the docs, so the same boundary patterns apply. Scanned with the
# identical normalise + allowlist + stale machinery. Note: identifier names
# (``entry_signal``, ``risk_multiplier`` …) do not match the phrase patterns
# below — only prose like "actionable trading signals" does — so the public
# API is unaffected by this gate.
CODE_SURFACE_FILES: tuple[str, ...] = (
    "core/indicators/kuramoto_ricci_composite.py",
    "core/indicators/market_state_contract.py",
)
EXCLUDE_DIR_PREFIXES: tuple[str, ...] = (
    "docs/archive/",  # frozen legacy reports
    "docs/audit/",  # audit findings quote banned phrasing as evidence
    "docs/adr/",  # architectural decision records (contextual rationale)
    "docs/releases/",  # immutable release notes (historical record)
)
# Policy / ledger files that enumerate banned phrasing by design.
EXCLUDE_FILES: frozenset[str] = frozenset(
    {
        "FORBIDDEN_CLAIMS.md",
        "CLAIMS.md",
        "PRODUCT_CATEGORY.md",  # canonical boundary statement (declares the boundary)
    }
)

# Boundary-sensitive phrases. Presence on the canonical surface is a
# violation UNLESS reworded or allowlisted. Phrases are matched on a
# whitespace- and unicode-dash-normalised, case-folded line so that
# line-wrapping and typographic dashes cannot smuggle a claim past the gate.
BOUNDARY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"live[ -]trading", "live-trading product framing"),
    (r"trading signals?", "trading-signal product output"),
    (r"actionable .{0,24}signals?", "actionable-signal promise"),
    (r"alpha[ -](engine|product|signal|generation)", "alpha-engine framing"),
    (r"signal generation", "signal-generation-as-product framing"),
)
_COMPILED: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(p), why) for p, why in BOUNDARY_PATTERNS
)

_UNICODE_DASHES = dict.fromkeys((0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212), ord("-"))


def _normalise(line: str) -> str:
    """Case-fold, NFKC-normalise, map unicode dashes to ASCII, collapse runs."""
    folded = unicodedata.normalize("NFKC", line).translate(_UNICODE_DASHES).casefold()
    return re.sub(r"\s+", " ", folded).strip()


@dataclass(frozen=True)
class AllowEntry:
    file: str
    match: str
    reason: str


@dataclass(frozen=True)
class Violation:
    file: str
    lineno: int
    why: str
    text: str


def _load_allowlist() -> list[AllowEntry]:
    if not ALLOWLIST_PATH.exists():
        return []
    payload = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    entries: list[AllowEntry] = []
    for item in payload.get("allow", []):
        entries.append(
            AllowEntry(
                file=item["file"],
                match=_normalise(item["match"]),
                reason=item.get("reason", ""),
            )
        )
    return entries


def _iter_surface() -> list[Path]:
    seen: set[Path] = set()
    files: list[Path] = []
    for glob in SCAN_GLOBS:
        for path in sorted(ROOT.glob(glob)):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT).as_posix()
            if rel in EXCLUDE_FILES:
                continue
            if any(rel.startswith(prefix) for prefix in EXCLUDE_DIR_PREFIXES):
                continue
            if path not in seen:
                seen.add(path)
                files.append(path)
    for rel in CODE_SURFACE_FILES:
        path = ROOT / rel
        if path.is_file() and path not in seen:
            seen.add(path)
            files.append(path)
    return files


def _scan() -> list[Violation]:
    violations: list[Violation] = []
    for path in _iter_surface():
        rel = path.relative_to(ROOT).as_posix()
        for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            norm = _normalise(raw)
            for pattern, why in _COMPILED:
                if pattern.search(norm):
                    violations.append(Violation(file=rel, lineno=lineno, why=why, text=norm))
                    break  # one finding per line is enough to require review
    return violations


def main() -> int:
    allow = _load_allowlist()
    violations = _scan()

    used_allow: set[AllowEntry] = set()
    unreviewed: list[Violation] = []
    for v in violations:
        matched = next(
            (e for e in allow if e.file == v.file and e.match in v.text),
            None,
        )
        if matched is None:
            unreviewed.append(v)
        else:
            used_allow.add(matched)

    stale = [e for e in allow if e not in used_allow]

    if unreviewed:
        print("ERROR: unreviewed product-category claims on the canonical surface.")
        print("GeoSync is a verification-first research platform — not a live-")
        print("trading or alpha product (see README.md / PRODUCT_CATEGORY.md).")
        print("Reword to the research boundary, or add a reasoned entry to")
        print(f"{ALLOWLIST_PATH.relative_to(ROOT).as_posix()}:\n")
        for v in unreviewed:
            print(f"  {v.file}:{v.lineno}: [{v.why}] {v.text[:120]}")
        print(f"\n{len(unreviewed)} unreviewed claim(s).")
        return 1

    if stale:
        print("ERROR: stale claim-boundary allowlist entries (no longer match).")
        print("Remove them so the allowlist stays an honest ledger:\n")
        for e in stale:
            print(f"  {e.file}: {e.match[:90]!r}")
        return 1

    print(
        f"Claim-boundary gate passed: {len(_iter_surface())} canonical surfaces "
        f"(docs + claim-bearing code) scanned, "
        f"{len(used_allow)} reviewed mechanism/record references."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
