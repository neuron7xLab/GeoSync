#!/usr/bin/env python3
# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Fail-closed dependency determinism gate.

This gate closes the gaps that ordinary pip-audit cannot see:

* CI bootstrap tooling must not install vulnerable pip/setuptools/wheel before
  security constraints are applied.
* scan-only manifests must not carry weaker floors than the security constraints
  they are supposed to validate.
* critical transitive security packages must be explicit in the scan manifest so
  scanner resolution cannot silently omit them.

Stdlib-only by design: this runs before dependency installation is trusted.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ACTION = ROOT / ".github" / "actions" / "setup-geosync" / "action.yml"
CONSTRAINTS = ROOT / "constraints" / "security.txt"
REQ_DEV = ROOT / "requirements-dev.txt"
REQ_SCAN = ROOT / "requirements-scan.txt"

_VERSION = r"([0-9][0-9A-Za-z.+\-_!]*)"
_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]+\])?")
_FLOOR = re.compile(rf"(?:==|>=)\s*{_VERSION}")

CRITICAL_SCAN_PACKAGES = frozenset(
    {
        "aiohttp",
        "cryptography",
        "fastapi",
        "jinja2",
        "pandera",
        "pyarrow",
        "pydantic",
        "pydantic-settings",
        "pyjwt",
        "python-multipart",
        "pyyaml",
        "requests",
        "sqlalchemy",
        "starlette",
        "strawberry-graphql",
        "tornado",
    }
)


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _version_tuple(text: str) -> tuple[int, ...]:
    parts = tuple(int(part) for part in re.findall(r"\d+", text)[:4])
    return parts or (0,)


def _read_bounds(path: Path) -> dict[str, str]:
    bounds: dict[str, str] = {}
    if not path.exists():
        return bounds
    for raw in path.read_text(encoding="utf-8").splitlines():
        code = raw.split("#", 1)[0].strip()
        if not code or code.startswith("-r ") or code.startswith("-c "):
            continue
        name_match = _NAME.match(code)
        floor_match = _FLOOR.search(code)
        if not name_match or not floor_match:
            continue
        bounds[_normalize_name(name_match.group(1))] = floor_match.group(1)
    return bounds


def _read_action_exact_pins(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    pins: dict[str, str] = {}
    for package in ("pip", "setuptools", "wheel"):
        matches = re.findall(rf"\b{package}=={_VERSION}", text)
        unique = sorted(set(matches))
        if len(unique) == 1:
            pins[package] = unique[0]
    return pins


def check(repo_root: Path = ROOT) -> list[str]:
    action = repo_root / ACTION.relative_to(ROOT)
    constraints_path = repo_root / CONSTRAINTS.relative_to(ROOT)
    req_dev_path = repo_root / REQ_DEV.relative_to(ROOT)
    req_scan_path = repo_root / REQ_SCAN.relative_to(ROOT)

    errors: list[str] = []
    constraints = _read_bounds(constraints_path)
    req_dev = _read_bounds(req_dev_path)
    req_scan = _read_bounds(req_scan_path)
    action_pins = _read_action_exact_pins(action)

    bootstrap_policy = {
        "pip": req_dev.get("pip"),
        "setuptools": constraints.get("setuptools"),
        "wheel": constraints.get("wheel"),
    }
    for package, floor in bootstrap_policy.items():
        if not floor:
            errors.append(f"missing security floor for CI bootstrap package: {package}")
            continue
        pin = action_pins.get(package)
        if not pin:
            errors.append(f"{action}: missing exact bootstrap pin for {package}")
            continue
        if _version_tuple(pin) < _version_tuple(floor):
            errors.append(
                f"{action}: {package}=={pin} is below required floor {floor}; "
                "CI would bootstrap with an untrusted packaging toolchain"
            )

    for package in sorted(CRITICAL_SCAN_PACKAGES):
        scan_floor = req_scan.get(package)
        if not scan_floor:
            errors.append(
                f"{req_scan_path}: missing explicit scan floor for security-critical "
                f"package {package}"
            )
            continue
        security_floor = constraints.get(package)
        if security_floor and _version_tuple(scan_floor) < _version_tuple(security_floor):
            errors.append(
                f"{req_scan_path}: {package}>={scan_floor} is below security "
                f"constraint floor {security_floor}; scan-only resolution can regress"
            )

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)

    errors = check(args.repo_root)
    if errors:
        print("DEPENDENCY OPERATIONAL DETERMINISM FAIL:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    print("DEPENDENCY OPERATIONAL DETERMINISM: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
