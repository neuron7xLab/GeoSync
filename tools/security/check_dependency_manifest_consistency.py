# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Security-floor consistency gate across dependency manifests.

``constraints/security.txt`` is the single source of truth for the patched
floor of every security-critical package. A manifest that declares a *lower*
lower-bound silently re-admits a vulnerable version: ``pip install .`` resolves
against ``pyproject.toml``, so a stale ``pyproject`` floor bypasses the floors
raised in ``requirements.txt``.

This gate parses the declared lower bound for each security-critical package in
each manifest and fails closed if any manifest admits a version *below* the
``constraints/security.txt`` floor.

Scope (issue #1107 / dependency policy):
    strict   : pyproject.toml — enforced (exit 1 on violation). This is the
               ``pip install .`` source of truth.
    advisory : requirements.txt / requirements-backend.txt — reported only.
               Their security floors are owned by the security-sweep lane
               (PR #1111); promoting these to strict here would create a
               cross-branch write conflict. Tighten to strict once #1111 lands
               (see docs/security/dependency_policy.md).

Run::

    python tools/security/check_dependency_manifest_consistency.py
"""

from __future__ import annotations

import argparse
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]
CONSTRAINTS = ROOT / "constraints" / "security.txt"
PYPROJECT = ROOT / "pyproject.toml"
REQUIREMENTS = ROOT / "requirements.txt"
REQUIREMENTS_BACKEND = ROOT / "requirements-backend.txt"

#: Packages whose floor is a vulnerability boundary. Only these are gated; a
#: lower bound below the constraints floor on any of them re-admits a CVE.
SECURITY_CRITICAL: frozenset[str] = frozenset(
    {
        "pyjwt",
        "cryptography",
        "aiohttp",
        "starlette",
        "tornado",
        "python-multipart",
        "fastapi",
    }
)


def _normalize(name: str) -> str:
    """PEP 503 normalization, dropping any extras (``PyJWT[crypto]`` -> ``pyjwt``)."""
    base = name.split("[", 1)[0]
    return re.sub(r"[-_.]+", "-", base).strip().lower()


def _floor(spec: SpecifierSet) -> Version | None:
    """Lowest version the specifier admits via a ``>=``/``==``/``~=`` bound."""
    candidates = [Version(s.version) for s in spec if s.operator in (">=", "==", "~=")]
    return min(candidates) if candidates else None


@dataclass(frozen=True)
class Violation:
    manifest: str
    package: str
    declared_floor: str
    required_floor: str

    def __str__(self) -> str:
        return (
            f"{self.manifest}: {self.package} admits >={self.declared_floor} "
            f"but security policy requires >={self.required_floor}"
        )


def _parse_constraint_floors() -> dict[str, Version]:
    floors: dict[str, Version] = {}
    for raw in CONSTRAINTS.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            req = Requirement(line)
        except InvalidRequirement:
            continue
        name = _normalize(req.name)
        if name not in SECURITY_CRITICAL:
            continue
        floor = _floor(req.specifier)
        if floor is not None:
            floors[name] = floor
    return floors


def _parse_pyproject_floors() -> dict[str, Version]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    deps: list[str] = list(data.get("project", {}).get("dependencies", []))
    for group in data.get("project", {}).get("optional-dependencies", {}).values():
        deps.extend(group)
    return _floors_from_specs(deps)


def _parse_requirements_floors(path: Path) -> dict[str, Version]:
    if not path.is_file():
        return {}
    specs: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            specs.append(line)
    return _floors_from_specs(specs)


def _floors_from_specs(specs: list[str]) -> dict[str, Version]:
    floors: dict[str, Version] = {}
    for spec in specs:
        try:
            req = Requirement(spec)
        except InvalidRequirement:
            continue
        name = _normalize(req.name)
        if name not in SECURITY_CRITICAL:
            continue
        floor = _floor(req.specifier)
        if floor is not None:
            floors[name] = min(floor, floors[name]) if name in floors else floor
    return floors


def check_manifest(
    manifest_label: str, manifest_floors: dict[str, Version], policy: dict[str, Version]
) -> list[Violation]:
    """Floors below the security policy for any security-critical package."""
    out: list[Violation] = []
    for pkg, declared in sorted(manifest_floors.items()):
        required = policy.get(pkg)
        if required is not None and declared < required:
            out.append(Violation(manifest_label, pkg, str(declared), str(required)))
    return out


def evaluate() -> tuple[list[Violation], list[Violation]]:
    """Return (strict_violations, advisory_violations)."""
    policy = _parse_constraint_floors()
    strict = check_manifest("pyproject.toml", _parse_pyproject_floors(), policy)
    advisory: list[Violation] = []
    advisory += check_manifest("requirements.txt", _parse_requirements_floors(REQUIREMENTS), policy)
    advisory += check_manifest(
        "requirements-backend.txt",
        _parse_requirements_floors(REQUIREMENTS_BACKEND),
        policy,
    )
    return strict, advisory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-strict",
        action="store_true",
        help="Treat requirements manifests as strict too (use after PR #1111 lands).",
    )
    args = parser.parse_args(argv)

    strict, advisory = evaluate()
    for v in advisory:
        marker = "ERROR" if args.all_strict else "ADVISORY"
        print(f"[{marker}] {v}")
    for v in strict:
        print(f"[ERROR] {v}")

    failed = strict + (advisory if args.all_strict else [])
    if failed:
        print(f"\nFAIL: {len(failed)} security-floor inconsistency(ies).")
        return 1
    print(
        "PASS: pyproject.toml security floors are consistent with "
        "constraints/security.txt"
        + (" (requirements treated advisory; see dependency_policy.md)" if advisory else "")
        + "."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
