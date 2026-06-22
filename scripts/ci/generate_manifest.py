#!/usr/bin/env python3
# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Regenerate MANIFEST.sha256 — the source-integrity cold-verify manifest.

The release gate's ``D.manifest`` probe cold-verifies that every entry in
``MANIFEST.sha256`` matches the working tree byte-for-byte. The manifest is a
supply-chain integrity proof: it certifies the cryptographic fingerprint of
every *source/config* file under version control.

Scope policy (explicit and reproducible, so the manifest is not a hand-rolled
snapshot that silently rots):

* Cover every git-tracked file …
* … EXCEPT ``MANIFEST.sha256`` itself (a file cannot hash its own hash), and
* … EXCEPT the ``artifacts/`` tree, which holds machine-generated evidence
  that the gate regenerates on every ``--deep`` run (e.g. the clean-clone
  probe records a temp install path). Pinning volatile machine output would
  make the integrity manifest self-defeating — it would go RED on the very
  next gate run. Artifact integrity is instead carried by each artifact's own
  ``artifact_sha256`` self-hash (see ``scripts/ci/proof_common.py``).

Run::

    python scripts/ci/generate_manifest.py            # write MANIFEST.sha256
    python scripts/ci/generate_manifest.py --check     # verify, exit 1 on drift

Regenerating is a deliberate act (the manifest declares "this is the tree at
this commit"); commit the result alongside the change that moved the tree.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "MANIFEST.sha256"
MANIFEST_REL = "MANIFEST.sha256"
EXCLUDE_PREFIXES: tuple[str, ...] = ("artifacts/",)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tracked_files() -> list[str]:
    proc = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _in_scope(rel: str) -> bool:
    if rel == MANIFEST_REL:
        return False
    return not any(rel.startswith(p) for p in EXCLUDE_PREFIXES)


def build_manifest() -> list[str]:
    """Return sorted ``<sha256>  ./<rel>`` lines for every in-scope tracked file."""
    lines: list[str] = []
    for rel in _tracked_files():
        if not _in_scope(rel):
            continue
        target = ROOT / rel
        if not target.is_file():  # deleted-but-staged / symlink to nowhere
            continue
        lines.append(f"{_sha256(target)}  ./{rel}")
    return sorted(lines)


def write() -> int:
    lines = build_manifest()
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST_REL} with {len(lines)} entries.")
    return 0


def check() -> int:
    if not MANIFEST.exists():
        print("MANIFEST.sha256 absent")
        return 1
    want = MANIFEST.read_text(encoding="utf-8").strip().splitlines()
    have = build_manifest()
    if want == have:
        print(f"MANIFEST.sha256 cold-verify clean ({len(have)} entries).")
        return 0
    want_set = set(want)
    have_set = set(have)
    added = len(have_set - want_set)
    removed = len(want_set - have_set)
    print(f"MANIFEST.sha256 drift: {added} new/changed, {removed} stale/removed entries.")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify instead of writing")
    args = parser.parse_args(argv)
    return check() if args.check else write()


if __name__ == "__main__":
    raise SystemExit(main())
