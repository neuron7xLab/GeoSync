#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail-closed inventory reconciliation for managed repository scopes."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = ROOT / "INVENTORY.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_tracked(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", rel],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _tracked_files_in_scope(scope: str) -> set[str]:
    scope_path = ROOT / scope
    if scope_path.is_file():
        return {scope} if _is_tracked(scope_path) else set()
    if not scope_path.exists():
        return set()
    files: set[str] = set()
    for path in scope_path.rglob("*"):
        if path.is_file() and _is_tracked(path):
            files.add(path.relative_to(ROOT).as_posix())
    return files


def main() -> int:
    if not INVENTORY_PATH.exists():
        print(f"ERROR: missing inventory file: {INVENTORY_PATH}")
        return 1

    payload = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    scopes: list[str] = payload.get("scopes", [])
    declared = payload.get("files", [])
    declared_paths = {item["path"] for item in declared}
    declared_hash = {item["path"]: item["sha256"] for item in declared}

    expected: set[str] = set()
    for scope in scopes:
        expected |= _tracked_files_in_scope(scope)

    missing = sorted(expected - declared_paths)
    orphan = sorted(declared_paths - expected)
    mismatched = sorted(
        path for path in (expected & declared_paths) if _sha256(ROOT / path) != declared_hash[path]
    )

    if missing or orphan or mismatched:
        if missing:
            print("ERROR: INVENTORY missing tracked files:")
            for path in missing:
                print(f"  - {path}")
        if orphan:
            print("ERROR: INVENTORY contains orphan files:")
            for path in orphan:
                print(f"  - {path}")
        if mismatched:
            print("ERROR: INVENTORY sha256 mismatch:")
            for path in mismatched:
                print(f"  - {path}")
        return 1

    print(f"Inventory sync passed ({len(expected)} files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
