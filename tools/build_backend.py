# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Dependency-free PEP 517/660 build backend for GeoSync.

The backend exists to keep the repository import and CLI smoke contracts
machine-verifiable in clean environments where setuptools is not preinstalled
and network access is unavailable. It intentionally implements only the wheel
and editable-wheel hooks used by pip for local installs.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import tomllib
import zipfile
from email.message import Message
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"


def _load_project() -> dict[str, object]:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _normalise_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def _normalise_version(version: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.!+-]+", ".", version)


def _project_name(doc: dict[str, object]) -> str:
    project = doc["project"]
    assert isinstance(project, dict)
    return str(project["name"])


def _project_version(doc: dict[str, object]) -> str:
    project = doc["project"]
    assert isinstance(project, dict)
    version = project.get("version")
    if isinstance(version, str):
        return version
    scm = doc.get("tool", {})
    if isinstance(scm, dict):
        setuptools_scm = scm.get("setuptools_scm", {})
        if isinstance(setuptools_scm, dict):
            fallback = setuptools_scm.get("fallback_version")
            if isinstance(fallback, str):
                return fallback
    return "0.1.0"


def _dist_info_name(doc: dict[str, object]) -> str:
    return f"{_normalise_distribution(_project_name(doc))}-{_normalise_version(_project_version(doc))}.dist-info"


def _wheel_name(doc: dict[str, object]) -> str:
    return f"{_normalise_distribution(_project_name(doc))}-{_normalise_version(_project_version(doc))}-py3-none-any.whl"


def _metadata(doc: dict[str, object]) -> bytes:
    project = doc["project"]
    assert isinstance(project, dict)
    msg = Message()
    msg["Metadata-Version"] = "2.4"
    msg["Name"] = str(project["name"])
    msg["Version"] = _project_version(doc)
    if description := project.get("description"):
        msg["Summary"] = str(description)
    if requires_python := project.get("requires-python"):
        msg["Requires-Python"] = str(requires_python)
    if license_value := project.get("license"):
        if isinstance(license_value, dict) and "text" in license_value:
            msg["License"] = str(license_value["text"])
    dependencies = project.get("dependencies", [])
    if isinstance(dependencies, list):
        for dependency in dependencies:
            msg["Requires-Dist"] = str(dependency)
    return bytes(msg)


def _wheel_metadata() -> bytes:
    return (
        "Wheel-Version: 1.0\n"
        "Generator: geosync.tools.build_backend\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n\n"
    ).encode("utf-8")


def _entry_points(doc: dict[str, object]) -> bytes:
    project = doc["project"]
    assert isinstance(project, dict)
    scripts = project.get("scripts", {})
    lines = ["[console_scripts]"]
    if isinstance(scripts, dict):
        for name in sorted(scripts):
            lines.append(f"{name} = {scripts[name]}")
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def _sha256_record(payload: bytes) -> str:
    digest = hashlib.sha256(payload).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _write_member(
    archive: zipfile.ZipFile,
    records: list[tuple[str, str, str]],
    arcname: str,
    payload: bytes,
) -> None:
    archive.writestr(arcname, payload)
    records.append((arcname, _sha256_record(payload), str(len(payload))))


def _iter_package_files(doc: dict[str, object]) -> Iterable[Path]:
    include_roots = _package_include_roots(doc)
    for root in include_roots:
        base = PROJECT_ROOT / root
        if not base.exists():
            continue
        if base.is_file():
            yield base
            continue
        for pattern in ("*.py", "py.typed"):
            yield from base.rglob(pattern)


def _package_include_roots(doc: dict[str, object]) -> list[Path]:
    tool = doc.get("tool", {})
    roots: list[Path] = []
    if isinstance(tool, dict):
        setuptools = tool.get("setuptools", {})
        if isinstance(setuptools, dict):
            packages = setuptools.get("packages", {})
            if isinstance(packages, dict):
                find = packages.get("find", {})
                if isinstance(find, dict):
                    include = find.get("include", [])
                    if isinstance(include, list):
                        for item in include:
                            if not isinstance(item, str) or item.endswith(".*"):
                                continue
                            roots.append(Path(item.replace(".", os.sep)))
    return sorted(set(roots))


def _write_dist_info(
    archive: zipfile.ZipFile,
    records: list[tuple[str, str, str]],
    doc: dict[str, object],
) -> str:
    dist_info = _dist_info_name(doc)
    _write_member(archive, records, f"{dist_info}/METADATA", _metadata(doc))
    _write_member(archive, records, f"{dist_info}/WHEEL", _wheel_metadata())
    _write_member(archive, records, f"{dist_info}/entry_points.txt", _entry_points(doc))
    return dist_info


def _finish_record(
    archive: zipfile.ZipFile, records: list[tuple[str, str, str]], dist_info: str
) -> None:
    record_path = f"{dist_info}/RECORD"
    rows = [*records, (record_path, "", "")]
    lines: list[str] = []
    for row in rows:
        output = []
        for field in row:
            if any(char in field for char in ',"\n'):
                output.append('"' + field.replace('"', '""') + '"')
            else:
                output.append(field)
        lines.append(",".join(output))
    archive.writestr(record_path, "\n".join(lines) + "\n")


def build_wheel(
    wheel_directory: str, config_settings: object = None, metadata_directory: str | None = None
) -> str:
    doc = _load_project()
    wheel = _wheel_name(doc)
    destination = Path(wheel_directory) / wheel
    records: list[tuple[str, str, str]] = []
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in _iter_package_files(doc):
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            _write_member(archive, records, rel, path.read_bytes())
        dist_info = _write_dist_info(archive, records, doc)
        _finish_record(archive, records, dist_info)
    return wheel


def build_editable(
    wheel_directory: str,
    config_settings: object = None,
    metadata_directory: str | None = None,
) -> str:
    doc = _load_project()
    wheel = _wheel_name(doc)
    destination = Path(wheel_directory) / wheel
    records: list[tuple[str, str, str]] = []
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        _write_member(archive, records, "geosync_editable.pth", f"{PROJECT_ROOT}\n".encode("utf-8"))
        dist_info = _write_dist_info(archive, records, doc)
        _finish_record(archive, records, dist_info)
    return wheel


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: object = None,
) -> str:
    doc = _load_project()
    dist_info = _dist_info_name(doc)
    target = Path(metadata_directory) / dist_info
    target.mkdir(parents=True, exist_ok=True)
    (target / "METADATA").write_bytes(_metadata(doc))
    (target / "WHEEL").write_bytes(_wheel_metadata())
    (target / "entry_points.txt").write_bytes(_entry_points(doc))
    return dist_info


def prepare_metadata_for_build_editable(
    metadata_directory: str,
    config_settings: object = None,
) -> str:
    return prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def get_requires_for_build_wheel(config_settings: object = None) -> list[str]:
    return []


def get_requires_for_build_editable(config_settings: object = None) -> list[str]:
    return []


def get_requires_for_build_sdist(config_settings: object = None) -> list[str]:
    return []
