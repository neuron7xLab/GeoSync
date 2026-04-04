#!/usr/bin/env python3
"""Validate CI toolchain package versions against lock contracts."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
from pathlib import Path


def _parse_pins(lock_path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw_line in lock_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            continue
        name, version = line.split("==", 1)
        pins[name.strip()] = version.strip()
    return pins


def _read_pytest_pin(dev_lock_path: Path) -> str:
    for raw_line in dev_lock_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("pytest=="):
            return line.split("==", 1)[1].strip()
    raise SystemExit(f"pytest pin not found in {dev_lock_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bootstrap-lock",
        default=".github/config/python-bootstrap.lock",
        help="Path to pinned bootstrap packages lockfile.",
    )
    parser.add_argument(
        "--dev-lock",
        default="requirements-dev.lock",
        help="Path to dev lockfile used for pytest pin.",
    )
    parser.add_argument(
        "--require-pytest",
        action="store_true",
        help="Fail if installed pytest does not match requirements-dev.lock pin.",
    )
    args = parser.parse_args()

    bootstrap_pins = _parse_pins(Path(args.bootstrap_lock))
    if not bootstrap_pins:
        raise SystemExit(f"No pinned entries found in {args.bootstrap_lock}")

    for package, expected in sorted(bootstrap_pins.items()):
        installed = metadata.version(package)
        if installed != expected:
            raise SystemExit(f"{package} version mismatch: {installed} != {expected}")
        print(f"{package} pinned: {installed}")

    if args.require_pytest:
        expected_pytest = _read_pytest_pin(Path(args.dev_lock))
        installed_pytest = metadata.version("pytest")
        if installed_pytest != expected_pytest:
            raise SystemExit(
                f"pytest version mismatch: {installed_pytest} != {expected_pytest}"
            )
        print(f"pytest pinned: {installed_pytest}")


if __name__ == "__main__":
    main()
