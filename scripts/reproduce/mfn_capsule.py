#!/usr/bin/env python3
# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Reproducible MFN integration capsule — build and verify.

This is the first *real* hash-pinned artifact graph in the repository. Until it
existed, ``scripts/ci/prove_repo_integrity.sh`` honestly reported
``MANIFEST_PROOF: NOT_PROVEN`` because a vacuous pass over zero tracked artifacts
is not proof of integrity.

The capsule is the byte-deterministic output of the dependency-light MFN
gateway (``geosync.mfn.cli``) under a pinned seed and ``SOURCE_DATE_EPOCH``. It
is **not** an empirical evidence artifact: the MFN bundle self-labels
``claim_tier: INSTRUMENTED``, ``decision: OBSERVE``, ``falsification_status:
BLOCKED``. What it proves is *integrity and reproducibility*: a committed
artifact whose SHA-256 in ``manifest.json`` matches the file on disk, and which
regenerates bit-for-bit from its declared command.

Two modes::

    python scripts/reproduce/mfn_capsule.py --build    # (re)generate + write manifest
    python scripts/reproduce/mfn_capsule.py --verify    # prove on-disk == regenerated

``--verify`` is the CI contract: it regenerates the bundle into a temp dir and
fails closed if any committed file's digest drifts, or if the manifest digests
do not match the committed files. Stdlib-only by design so it runs in the
dependency-light MFN lane.

Exit codes::

    0  — capsule is reproducible and every manifest digest matches
    1  — drift detected (non-reproducible bundle or stale manifest)
    2  — usage error
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAPSULE_DIR = ROOT / "artifacts" / "reproducible_capsules" / "mfn_integration_seed7_p256"
BUNDLE_DIR = CAPSULE_DIR / "bundle"
MANIFEST_PATH = CAPSULE_DIR / "manifest.json"

# Pinned reproducibility parameters. SOURCE_DATE_EPOCH matches the value the
# MFN release gate uses (.github/workflows/mfn-release-gate.yml) so the capsule
# and that gate agree on the byte-exact bundle.
SEED = 7
POINTS = 256
SOURCE_DATE_EPOCH = "1700000000"
GENERATOR = (
    "SOURCE_DATE_EPOCH=1700000000 python -m geosync.mfn.cli "
    "--out <dir> --seed 7 --points 256 run"
)
SCHEMA_VERSION = "geosync.reproducible_capsule.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generate_bundle(out_dir: Path) -> None:
    """Run the MFN gateway deterministically into *out_dir*."""
    env = dict(os.environ)
    env["SOURCE_DATE_EPOCH"] = SOURCE_DATE_EPOCH
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "geosync.mfn.cli",
            "--out",
            str(out_dir),
            "--seed",
            str(SEED),
            "--points",
            str(POINTS),
            "run",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def _bundle_files(bundle_dir: Path) -> list[Path]:
    return sorted(p for p in bundle_dir.iterdir() if p.is_file())


def build() -> int:
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "bundle"
        _generate_bundle(staging)
        shutil.copytree(staging, BUNDLE_DIR)

    artifacts = []
    for path in _bundle_files(BUNDLE_DIR):
        artifacts.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "reproducible_integration_capsule",
        "claim_tier": "INSTRUMENTED",
        "evidence_class": "integrity_and_reproducibility",
        "is_empirical_evidence": False,
        "honesty_note": (
            "Byte-deterministic output of the dependency-light MFN gateway. This "
            "proves artifact integrity and reproducibility, not market alpha or "
            "empirical inference. The bundle self-labels INSTRUMENTED / OBSERVE / "
            "BLOCKED; it makes no falsifiable scientific claim."
        ),
        "generator": GENERATOR,
        "seed": SEED,
        "points": POINTS,
        "source_date_epoch": int(SOURCE_DATE_EPOCH),
        "verify_command": "python scripts/reproduce/mfn_capsule.py --verify",
        "artifacts": artifacts,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"capsule built: {len(artifacts)} artifacts -> {MANIFEST_PATH.relative_to(ROOT)}")
    return 0


def verify() -> int:
    failures: list[str] = []

    if not MANIFEST_PATH.exists():
        print(f"missing manifest: {MANIFEST_PATH.relative_to(ROOT)}", file=sys.stderr)
        return 1
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", [])
    if not artifacts:
        failures.append("manifest declares zero artifacts (vacuous proof)")

    # 1. On-disk committed files match their declared digests.
    declared: dict[str, str] = {}
    for entry in artifacts:
        rel, expected = entry.get("path"), entry.get("sha256")
        declared[rel] = expected
        target = ROOT / rel
        if not target.exists():
            failures.append(f"declared artifact missing on disk: {rel}")
            continue
        actual = _sha256(target)
        if actual != expected:
            failures.append(f"manifest digest stale for {rel}: {expected} != {actual}")

    # 2. The committed bundle regenerates bit-for-bit from its declared command.
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp) / "bundle"
        _generate_bundle(fresh)
        committed = {p.name: p for p in _bundle_files(BUNDLE_DIR)} if BUNDLE_DIR.exists() else {}
        regenerated = {p.name: p for p in _bundle_files(fresh)}
        if committed.keys() != regenerated.keys():
            failures.append(
                f"bundle file set drifted: committed={sorted(committed)} "
                f"regenerated={sorted(regenerated)}"
            )
        for name in sorted(committed.keys() & regenerated.keys()):
            if _sha256(committed[name]) != _sha256(regenerated[name]):
                failures.append(f"non-reproducible bundle file: bundle/{name}")

    if failures:
        print("MFN capsule verification FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(
        f"MFN capsule verification PASSED: {len(artifacts)} artifacts reproduced "
        "and digests match."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true", help="(re)generate capsule + manifest")
    group.add_argument("--verify", action="store_true", help="prove on-disk == regenerated")
    args = parser.parse_args(argv)
    if args.build:
        return build()
    return verify()


if __name__ == "__main__":
    raise SystemExit(main())
