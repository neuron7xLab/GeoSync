#!/usr/bin/env python3
# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Debt-baseline monotonicity meta-gate — closes the ``--write`` laundering hole.

``check_code_hygiene.py`` / ``check_skip_ratchet.py`` freeze debt and fail on
new violators — but their own ``--write`` regenerates the baseline from the
current tree with NO comparison to the prior baseline. So the cheapest
response to a red gate is: add the debt, run ``--write``, commit the larger
baseline; the next ``verify`` reports "held" while debt grew. In a
deadline-driven, many-agent repo that escape converges the ratchet to a
rubber stamp.

This meta-gate removes the escape. It diffs the committed baseline *totals*
in the working tree against ``--base-ref`` (default ``origin/main``) and
fails if ANY dimension total increased. ``--write`` that pays debt DOWN
(totals shrink) passes; ``--write`` that launders debt UP cannot land. The
per-dimension hygiene/skip ratchets still catch identity-level changes; this
gate guarantees the aggregate can only fall.

Fail-closed: if the base ref or a base baseline cannot be read, the gate
fails (it cannot prove monotonicity) rather than passing vacuously.

Usage::

    python scripts/ci/check_debt_baseline_monotonic.py
    python scripts/ci/check_debt_baseline_monotonic.py --base-ref origin/main

Exit codes::

    0  — every baseline total <= the base ref's total
    1  — a baseline total grew, or the base ref could not be read
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Each tracked baseline -> how to reduce its JSON payload to per-dimension totals.
HYGIENE = "docs/CODE_DEBT_BASELINE.json"
SKIP = "docs/SKIP_RATCHET_BASELINE.json"


def _hygiene_totals(payload: dict[str, object]) -> dict[str, int]:
    out: dict[str, int] = {}
    sym = payload.get("symbol_debt", {})
    assert isinstance(sym, dict)
    for dim, entries in sym.items():
        out[f"symbol:{dim}"] = len(entries)
    fc = payload.get("file_count_debt", {})
    assert isinstance(fc, dict)
    for dim, counts in fc.items():
        out[f"count:{dim}"] = sum(counts.values())
    return out


def _skip_totals(payload: dict[str, object]) -> dict[str, int]:
    skips = payload.get("skips", {})
    assert isinstance(skips, dict)
    return {"skips": sum(skips.values())}


_REDUCERS = {HYGIENE: _hygiene_totals, SKIP: _skip_totals}


def _read_working(rel: str) -> dict[str, object]:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def _read_base(rel: str, base_ref: str) -> dict[str, object] | None:
    proc = subprocess.run(
        ["git", "show", f"{base_ref}:{rel}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return json.loads(proc.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Debt-baseline monotonicity meta-gate.")
    parser.add_argument("--base-ref", default="origin/main")
    args = parser.parse_args(argv)

    failed = False
    for rel, reducer in _REDUCERS.items():
        base_payload = _read_base(rel, args.base_ref)
        if base_payload is None:
            # New baseline absent from base ref: any positive total is fresh
            # debt that must be justified, not silently admitted. Treat the
            # base as all-zero so the gate fails closed on a non-empty ledger.
            base_totals: dict[str, int] = {}
            print(f"NOTE: {rel} not found at {args.base_ref}; treating base totals as 0.")
        else:
            base_totals = reducer(base_payload)
        cur_totals = reducer(_read_working(rel))

        for dim, cur in sorted(cur_totals.items()):
            base = base_totals.get(dim, 0)
            if cur > base:
                failed = True
                print(f"ERROR: {rel} [{dim}] grew {base} -> {cur} (baseline totals only shrink).")

    if failed:
        print(
            "A baseline total increased. New debt must be refactored away, not "
            "frozen: do NOT --write to admit it. If this is a legitimate scope "
            "expansion, it needs explicit review, not a silent re-baseline."
        )
        return 1

    print(f"Debt-baseline monotonicity held: no baseline total grew vs {args.base_ref}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
