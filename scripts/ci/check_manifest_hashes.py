#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify SHA-256 integrity for contract manifests with artifact digests."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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


def _discover_manifest_paths() -> list[Path]:
    manifests: list[Path] = []
    for path in ROOT.rglob("manifest.json"):
        if ".git/" in path.as_posix():
            continue
        if _is_tracked(path):
            manifests.append(path)
    return sorted(manifests)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    failures: list[str] = []
    checked = 0

    for manifest in _discover_manifest_paths():
        if not manifest.exists():
            failures.append(f"missing manifest: {manifest.relative_to(ROOT).as_posix()}")
            continue
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        artifacts = payload.get("artifacts", [])
        if not artifacts:
            continue
        for artifact in artifacts:
            rel = artifact.get("path")
            expected = artifact.get("sha256")
            if not isinstance(rel, str) or not rel:
                failures.append(f"{manifest.name}: artifact entry missing valid path")
                continue
            if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
                failures.append(f"{manifest.name}: invalid sha256 for {rel}")
                continue
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
