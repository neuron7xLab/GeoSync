#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Architecture-debt inference for the repository.

Surfaces structural debt as a single machine-readable inference: each
debt class is measured deterministically from the tree, compared against
an explicit threshold, and assigned a severity. The point is not a score
— it is to make the debt *legible and addressable* rather than implicit.

Read-only, offline, deterministic. It classifies debt; it does not
repair it, and it never reports a measure it could not compute as zero
(a measure that fails resolves to UNKNOWN, not OK).

Severity ladder: HIGH > MEDIUM > LOW > INFO. A class is `over_threshold`
when its measured count exceeds `threshold`. ``by_design`` classes are
standing architectural choices, reported for visibility, never as debt.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]

HIGH, MEDIUM, LOW, INFO = "HIGH", "MEDIUM", "LOW", "INFO"
UNKNOWN = "UNKNOWN"


def _py_files() -> list[Path]:
    return [p for p in ROOT.rglob("*.py") if ".venv" not in p.parts]


def _grep_count(pattern: str) -> int:
    rx = re.compile(pattern)
    n = 0
    for f in _py_files():
        try:
            text = f.read_text(encoding="utf-8")
        except OSError:
            continue
        n += len(rx.findall(text))
    return n


def _acceptor_files() -> list[Path]:
    base = ROOT / ".claude" / "commit_acceptors"
    return sorted(base.glob("*.yaml")) if base.exists() else []


def m_empty_evidence_ledger() -> tuple[int, int]:
    """Acceptors whose inline evidence ledger is empty (`evidence: []`).

    Returns (empty, total). Per the populated-evidence-ledger invariant,
    an empty inline ledger is unbacked governance.
    """
    files = _acceptor_files()
    empty = sum(1 for f in files if re.search(r"^evidence:\s*\[\]\s*$", f.read_text(), re.M))
    return empty, len(files)


def m_type_ignore() -> int:
    return _grep_count(r"type:\s*ignore")


def m_noqa() -> int:
    return _grep_count(r"\bnoqa\b")


def m_test_skips() -> int:
    rx = re.compile(r"pytest\.mark\.skip(if)?\b")
    return sum(
        1
        for f in (ROOT / "tests").rglob("*.py")
        if (ROOT / "tests").exists() and rx.search(f.read_text(encoding="utf-8"))
    )


def m_test_slow() -> int:
    rx = re.compile(r"pytest\.mark\.slow\b")
    return sum(
        1
        for f in (ROOT / "tests").rglob("*.py")
        if (ROOT / "tests").exists() and rx.search(f.read_text(encoding="utf-8"))
    )


def m_todo_markers() -> int:
    return _grep_count(r"\b(TODO|FIXME|HACK|XXX)\b")


def m_dual_physics_layers() -> int:
    a = ROOT / ".claude" / "physics"
    b = ROOT / "physics_contracts"
    return int(a.exists() and b.exists())


@dataclass(frozen=True)
class DebtClass:
    id: str
    title: str
    severity: str
    threshold: int
    measure: Callable[[], int]
    note: str
    by_design: bool = False


# The catalogue. Thresholds are explicit budgets; exceeding them flags debt.
CLASSES: tuple[DebtClass, ...] = (
    DebtClass(
        "empty_evidence_ledger",
        "Commit-acceptors with an empty inline evidence ledger",
        HIGH,
        threshold=0,
        measure=lambda: m_empty_evidence_ledger()[0],
        note="Per the populated-evidence-ledger invariant, evidence: [] is "
        "unbacked governance — the acceptor promises but carries no proof.",
    ),
    DebtClass(
        "type_ignore_suppressions",
        "`type: ignore` suppressions across the Python surface",
        MEDIUM,
        threshold=400,
        measure=m_type_ignore,
        note="Each suppression is a hole in the --strict type contract; "
        "concentrated in tests but not exclusively.",
    ),
    DebtClass(
        "noqa_suppressions",
        "`noqa` lint suppressions across the Python surface",
        LOW,
        threshold=400,
        measure=m_noqa,
        note="Lint exceptions; track the trend, do not let it grow silently.",
    ),
    DebtClass(
        "test_skip_files",
        "Test files carrying skip/skipif markers",
        MEDIUM,
        threshold=30,
        measure=m_test_skips,
        note="Skipped tests are coverage the suite claims but does not run.",
    ),
    DebtClass(
        "test_slow_files",
        "Test files marked slow (excluded from fast lanes)",
        LOW,
        threshold=40,
        measure=m_test_slow,
        note="Slow-marked tests do not run in the fast gate; drift hides here.",
    ),
    DebtClass(
        "todo_markers",
        "TODO/FIXME/HACK/XXX markers in Python",
        LOW,
        threshold=30,
        measure=m_todo_markers,
        note="Inline deferrals; low count is healthy.",
    ),
    DebtClass(
        "dual_physics_contract_layers",
        "Parallel physics-contract layers (.claude/physics + physics_contracts)",
        INFO,
        threshold=1,
        measure=m_dual_physics_layers,
        note="Sanctioned duality (keep both); reported for visibility only.",
        by_design=True,
    ),
)


def infer() -> dict[str, object]:
    findings: list[dict[str, object]] = []
    worst = INFO
    order = {INFO: 0, LOW: 1, MEDIUM: 2, HIGH: 3, UNKNOWN: 4}
    for c in CLASSES:
        try:
            count = c.measure()
            status = UNKNOWN if count < 0 else ("DEBT" if count > c.threshold else "OK")
        except Exception:  # noqa: BLE001 - a measure that crashes is UNKNOWN, not OK
            count, status = -1, UNKNOWN
        effective = (
            c.severity
            if (status == "DEBT" and not c.by_design)
            else (UNKNOWN if status == UNKNOWN else INFO)
        )
        if order[effective] > order[worst]:
            worst = effective
        extra: dict[str, object] = {}
        if c.id == "empty_evidence_ledger":
            empty, total = m_empty_evidence_ledger()
            extra = {"total_acceptors": total, "ratio": round(empty / total, 3) if total else None}
        findings.append(
            {
                "id": c.id,
                "title": c.title,
                "severity": c.severity,
                "count": count,
                "threshold": c.threshold,
                "status": status,
                "by_design": c.by_design,
                "note": c.note,
                **extra,
            }
        )
    return {
        "worst_severity": worst,
        "debt_classes": findings,
        "actionable": [f["id"] for f in findings if f["status"] == "DEBT" and not f["by_design"]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if any actionable HIGH debt class is over threshold",
    )
    args = parser.parse_args(argv)
    result = infer()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for f in result["debt_classes"]:  # type: ignore[union-attr]
            flag = "·" if f["status"] == "OK" else f["status"]
            print(
                f"[{f['severity']:<6}] {flag:<7} {f['count']:>5} " f"(>{f['threshold']}) {f['id']}"
            )
        print(f"worst_severity: {result['worst_severity']}")
        print(f"actionable: {', '.join(result['actionable']) or 'none'}")  # type: ignore[arg-type]
    has_high = any(
        f["status"] == "DEBT" and f["severity"] == HIGH and not f["by_design"]
        for f in result["debt_classes"]  # type: ignore[union-attr]
    )
    return 1 if args.strict and has_high else 0


if __name__ == "__main__":
    raise SystemExit(main())
