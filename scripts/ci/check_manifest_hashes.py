#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify SHA-256 integrity for contract manifests with artifact digests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATHS = [
    ROOT / "stakeholders" / "manifest.json",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    failures: list[str] = []
    checked = 0

    for manifest in MANIFEST_PATHS:
        if not manifest.exists():
            failures.append(f"missing manifest: {manifest.relative_to(ROOT).as_posix()}")
            continue
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        artifacts = payload.get("artifacts", [])
        for artifact in artifacts:
            rel = artifact["path"]
            expected = artifact["sha256"]
            target = ROOT / rel
            if not target.exists():
                failures.append(f"{manifest.name}: missing artifact {rel}")
                continue
            actual = _sha256(target)
            checked += 1
            if actual != expected:
                failures.append(
                    f"{manifest.name}: checksum mismatch for {rel} (expected {expected}, got {actual})"
                )

    if failures:
        print("Manifest hash verification failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(f"Manifest hash verification passed ({checked} artifacts).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
