#!/usr/bin/env python3
# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Package-boundary ratchet — fail-closed, monotone-down wheel-surface ledger.

GeoSync's canonical wheel namespace is ``geosync*``. Today the wheel still ships
16 non-``geosync`` top-level packages (``core``, ``application``, ``scripts``,
``tools``, ...): nine console-script entry points still resolve into them, so a
one-shot ``packages.find = geosync*`` would break the installed CLI. The
canonical fix is the ADR-0024 migration (re-home runtime code under
``geosync.*``, repoint entry points), which lands incrementally.

This ratchet makes that migration *safe and monotone* — the same doctrine the
import-architecture ratchet applies to ``src.*`` imports:

* a NEW non-``geosync`` top-level package entering the wheel surface fails the
  build — the leak set may never grow while migration is in flight;
* a package re-homed out of the surface also fails, with a one-line fix
  (``--write``) — paying down must tighten the ledger so it trends to empty.

Source of truth is ``[tool.setuptools.packages.find].include`` in
``pyproject.toml`` — the exact knob that controls what the wheel ships. The
release gate's ``B.wheel`` probe is the *aspirational* target (GREEN only at
zero leak); this ratchet is the *anti-regression floor* that protects the path
to it. Migration target: leak set empty, ``B.wheel`` GREEN.

Usage::

    python scripts/ci/check_package_boundary.py            # verify
    python scripts/ci/check_package_boundary.py --write     # tighten ledger

Exit codes::

    0  — current leak set == baseline (no new leak, no stale entry)
    1  — a new non-geosync package entered, or a re-homed entry is stale
    2  — pyproject.toml is missing the packages.find include list
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
BASELINE_PATH = ROOT / ".github" / "package_boundary_baseline.json"


def _shipped_leak() -> set[str]:
    """Non-geosync top-level packages the wheel currently ships.

    Derived from ``[tool.setuptools.packages.find].include`` (the knob that
    controls the wheel surface) intersected with real top-level directories.
    """
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    include = data["tool"]["setuptools"]["packages"]["find"]["include"]
    tops = {pattern.split(".", 1)[0] for pattern in include}
    return {top for top in tops if not top.startswith("geosync") and (ROOT / top).is_dir()}


def _load_baseline() -> set[str]:
    if not BASELINE_PATH.exists():
        return set()
    payload = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return set(payload.get("non_geosync_packages", []))


def _write_baseline(leak: set[str]) -> None:
    existing = {}
    if BASELINE_PATH.exists():
        existing = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    payload = {
        "_doc": existing.get(
            "_doc",
            "Package-boundary debt ledger (RATCHET). Non-geosync top-level "
            "packages still shipped in the wheel surface. This set may ONLY "
            "shrink; check_package_boundary.py enforces it. Target: empty "
            "(single geosync* namespace, release-gate B.wheel GREEN).",
        ),
        "version": 1,
        "non_geosync_packages": sorted(leak),
    }
    BASELINE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Package-boundary ratchet.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Regenerate the baseline from the current tree (tighten the ledger).",
    )
    args = parser.parse_args(argv)

    try:
        cur = _shipped_leak()
    except KeyError:
        print(
            "ERROR: pyproject.toml has no [tool.setuptools.packages.find].include list",
            file=sys.stderr,
        )
        return 2

    if args.write:
        _write_baseline(cur)
        print(f"Baseline tightened: {len(cur)} non-geosync packages on the wheel surface.")
        return 0

    base = _load_baseline()
    new = sorted(cur - base)
    fixed = sorted(base - cur)
    failed = False

    if new:
        failed = True
        print("ERROR: new package-boundary debt (the geosync* wheel ratchet only shrinks).")
        for pkg in new:
            print(f"  + non-geosync package on wheel surface: {pkg}")
        print(
            "Fix: re-home the module under the canonical `geosync` namespace; "
            "do not add new top-level packages to packages.find.include."
        )

    if fixed:
        failed = True
        print(
            f"Boundary paid down ({len(fixed)} package(s) re-homed) but the ledger "
            "is stale. Tighten it so it stays honest:"
        )
        print("  python scripts/ci/check_package_boundary.py --write")

    if not failed:
        print(
            f"Package-boundary ratchet held: {len(cur)} non-geosync packages on the "
            "wheel surface (target 0). No new debt."
        )
        if not cur:
            print("Ledger empty — single canonical geosync* wheel achieved.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
