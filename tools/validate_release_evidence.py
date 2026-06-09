#!/usr/bin/env python3
# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Validate release evidence required files and SHA256SUMS."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REQUIRED_FILES = (
    "FINAL_STATUS.md",
    "SHA256SUMS",
    "structural_audit.md",
    "blocker_matrix.json",
    "commands.log",
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, nargs="?", default=Path("artifacts/release_evidence"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    bundle = args.bundle
    errors: list[str] = []
    for rel in REQUIRED_FILES:
        target = bundle / rel
        if not target.is_file() or target.stat().st_size == 0:
            errors.append(f"missing-or-empty: {rel}")
    sums = bundle / "SHA256SUMS"
    if sums.is_file():
        for line in sums.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                errors.append(f"malformed-sha-line: {line}")
                continue
            expected, rel = parts
            target = bundle / rel.strip()
            if not target.is_file():
                errors.append(f"sha-target-missing: {rel.strip()}")
                continue
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual != expected:
                errors.append(f"sha-mismatch: {rel.strip()}")
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"release_evidence_valid={bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
